import cv2
import fitz
import numpy as np
import os
import math
import io
from PIL import Image
from typing import Optional
from dotenv import load_dotenv
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()

origins =[
    'http://127.0.0.1:5500/',
    'http://localhost',
    'http://localhost:3000',
    'http://localhost:8080',
    'http://opticost.ai',
    'https://opticost.ai',
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# API & key for 1st prediction (for Area Detection) 
endpoint_id = os.getenv("ENDPOINT_ID1")
api_key = os.getenv("API_KEY1")
# Api & key for 2nd prediction (for Feature Detection)
endpoint_id2 = os.getenv("ENDPOINT_ID2")
api_key2 = os.getenv("API_KEY2")

def convert_to_png_and_resize(file: UploadFile, max_pixels=1000000):
    try:
        file_extension = file.filename.split(".")[-1].lower()
        images = []

        if file_extension == 'pdf':
            with fitz.open(stream=io.BytesIO(file.file.read())) as doc:
                if len(doc) !=1:
                        raise ValueError('Only Single-page PDFs are allowed')
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                current_pixels = img.width * img.height
                if current_pixels > max_pixels:
                    scaling_factor = math.sqrt(max_pixels / current_pixels)
                    new_width = int(img.width * scaling_factor)
                    new_height = int(img.height * scaling_factor)

                    # Ensures the resized image is as large as possible without exceeding max_pixels limit
                    while (new_width * new_height) > max_pixels:
                        new_width -= 1
                        new_height = int(new_width / (img.width / img.height))
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                images.append(img)
        else:
            img = Image.open(io.BytesIO(file.file.read()))
            current_pixels = img.width * img.height
            if current_pixels > max_pixels:
                scaling_factor = math.sqrt(max_pixels / current_pixels)
                new_width = int(img.width * scaling_factor)
                new_height = int(img.height * scaling_factor)
                
                # Ensures the resized image is as large as possible without exceeding max_pixels limit
                while (new_width * new_height) > max_pixels:
                    new_width -= 1
                    new_height = int(new_width / (img.width / img.height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            images.append(img)

        return images
    except Exception as e:
        # Handle exceptions
        print(f"Error converting file: {e}")    
        return []

    
@app.get('/')
def read_root():
    return{'Nothing to see here'}

@app.post("/analyze")
def analyze(response: Response, file: UploadFile = File(...), user_entered_scale: float = Form(...), user_entered_height: Optional[float] = Form(None), user_entered_slope_factor: float = Form(...)):
    # user_entered_width: int = Form(...), user_entered_height: int = Form(...)
    print("Request received...")
    response_dict = {"Processing": True}

    # Load image & run conversion
    user_uploaded_file = convert_to_png_and_resize(file)
     
    # after conversion get the converted image
    converted_image = user_uploaded_file[0]
    
    # Convert the image to gray scale
    gray_pre = converted_image.convert('L')

    # Prep grayscale from PIL to np array image
    gray = np.array(gray_pre)

    # Convert the grayscale image to strict black and white using cv2.threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_BGR2RGB)

    # Run inference
    predictor = Predictor(endpoint_id, api_key=api_key)
    first_predictions = predictor.predict(gray)
    first_predictions_overlay = overlay_predictions(first_predictions, binary_rgb)
    # Turn image to numpy array
    first_predictions_overlay_np = np.array(first_predictions_overlay)

    #first_predictions_overlay.show()

    ########################################################################
    ## Process Converted Image to Isolate Roof Area & Remove Surroundings ##
    ########################################################################

    # Load the two mask images 
    mask_area_of_interest = cv2.cvtColor(first_predictions_overlay_np, cv2.COLOR_RGB2BGR)

    color_ranges = {
        '1st': (np.array([216, 238, 243]), np.array([220, 242, 247])),
        '2nd': (np.array([215, 222, 206]), np.array([219, 226, 210]))
    }
    # Load the image using OpenCV to get a NumPy array
    image = mask_area_of_interest

    def analyze_border_for_dominant_color(image, border_width, color_ranges):
        color_counts = {key: 0 for key in color_ranges.keys()}
        borders = [image[:border_width, :], image[-border_width:, :], image[:, :border_width], image[:, -border_width:]]

        def increment_color_count(border_segment, color_name, lower_bound, upper_bound):
            mask = cv2.inRange(border_segment, lower_bound, upper_bound)
            color_counts[color_name] += cv2.countNonZero(mask)

        for color_name, (lower, upper) in color_ranges.items():
            for border in borders:
                increment_color_count(border, color_name, lower, upper)

        dominant_color_name = max(color_counts, key=color_counts.get)
        return dominant_color_name
    
    border_width = 20  # Adjust based on your image size
    dominant_color_name = analyze_border_for_dominant_color(image, border_width, color_ranges)

    # Determine the non-dominant color name
    non_dominant_color_name = '1st' if dominant_color_name == '2nd' else '2nd'

    # Create mask for non-dominant color
    non_dominant_lower, non_dominant_upper = color_ranges[non_dominant_color_name]
    non_dominant_mask = cv2.inRange(image, non_dominant_lower, non_dominant_upper)

    # Shrink the mask by eroding it by 10 pixels
    kernel_shrink = np.ones((10, 10), np.uint8)  # The kernel size determines how much the mask will be shrunk
    eroded_mask = cv2.erode(non_dominant_mask, kernel_shrink, iterations=1)

    # Dilate the mask to expand it by roughly 40 pixels
    kernel = np.ones((200, 200), np.uint8)  # Adjust the kernel size as needed for your specific requirement
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)


    # Optionally, apply the mask to the original image to visualize the result
    result_image = cv2.bitwise_and(gray, gray, mask=dilated_mask)

    result_image2 = cv2.bilateralFilter(result_image,5,150,150)

    ####################################################################################
    ## RUN 2nd PREDICTIONS FOR FEATURES USING MODIFIED ROOF IMAGE FROM 1st PREDICTION ##
    ####################################################################################


    # Run inference for 2nd prediction (for Feature Detection)
    predictor2 = Predictor(endpoint_id2, api_key=api_key2)

    predictions2 = predictor2.predict(result_image2)
    color_map = {
    "Ridge": "red",
    "Hip": "green",
    "Valley": "orange",
    "Eave": "blue",
    "Rake": "purple",
    "IGNORE": "white"
    }
    options = {
        "color_map": color_map
    }
    
    
    overlayed_image3 = overlay_predictions(predictions2, result_image2, options=options)
    
    # Convert the PIL Image to a numpy array, then ensure format
    overlayed_image3_np = np.array(overlayed_image3)

    overlayed_image3_np = overlayed_image3_np.astype(np.uint8)

    #print('second prediction with overlay, this is image 2')
    #overlayed_image3.show()

    ##################################################
    ## PIXEL COUNT FOR ROOF AREA & SURROUNDING AREA ##
    ##################################################

    # Calculate the number of pixels for non-dominant mask (pixel_count_color1)
    pixel_count_color1 = cv2.countNonZero(non_dominant_mask)

    # Create a dominant mask by inverting the non-dominant mask
    dominant_mask = cv2.bitwise_not(non_dominant_mask)

    # Calculate the number of pixels for dominant mask (pixel_count_color2)
    pixel_count_color2 = cv2.countNonZero(dominant_mask)

    # Now you have the pixel counts directly from the masks without needing to analyze HSV color space

    print(f"Pixel Count for Non-Dominant Color/Feature: {pixel_count_color1}")
    print(f"Pixel Count for Dominant Color/Feature: {pixel_count_color2}")

    #####################################
    ## BASE CALCULATIONS AND VARIABLES ##
    #####################################
    
    # Create page size selections (e.g. 24" x 36" default)
    ## TBD - Option to choose between different page size options later, to be used if autosizing is either unavailable or incorrect. 

    # Define the default input image size (24" high by 36" wide is standard, but should be similar 2:3 proportional sizes, such as 12"x18" - rarely may be another size, such as 8.5" x 11")
    if user_entered_height:
        default_height_inches = user_entered_height
    else: 
        default_height_inches = 24
    #default_width_inches = 36  #user_entered_width along with file upload above
    
    # Define the default scale (ex. 1/4" on the image = 1 foot in real life)
    default_scale_factor = round(user_entered_scale, 3)  # Adjust as needed

    # Define the desired height and width for final calculation purposes
    desired_height_inches = 24  # Adjust as needed
    desired_width_inches = 36   # Adjust as needed

    # Rise is equal to the numerator float in the slope fraction -- Ex. If slope is [5.0 / 12], rise is [5.0] and slope_multiplier would then be equal to [1.08333]
    # slope_multiplier would typically apply to the AREA, RAKE, HIP, and VALLEY
    slope_multiplier = ((user_entered_slope_factor ** 2 + 144) ** 0.5) / 12


    #################################################################
    ## DEFINE CALCS / MEASUREMENTS TO BE USED W/ ADJUSTED IMAGE(S) ##
    #################################################################
    
    # Calculate the scaling factors for height and width
    #height_scale_factor = gray.shape[0] / default_height_inches
    #width_scale_factor = gray.shape[1] / default_width_inches

    # Adjust the default scale based on the desired size
    #adjusted_scale_factor = default_scale_factor * (height_scale_factor + width_scale_factor) / 2

    # Count the total number of pixels in the image
    total_pixels = gray.shape[0] * gray.shape[1]

    # Calculate the square footage
    square_feet_per_pixel =  (((desired_width_inches) * (1/default_scale_factor)) * ((desired_height_inches) * (1/default_scale_factor))) / (total_pixels)

    # Calculate the real-world width represented by each pixel
    page_inches_per_pixel_desired_width = (desired_width_inches) / (gray.shape[1]) # 24 inch div by 720 pixels, and 36 inch div by 1080 pixels = same answer
    page_inches_per_pixel_desired_height = (desired_height_inches) / (gray.shape[0]) # 24 inch div by 720 pixels, and 36 inch div by 1080 pixels = same answer
    print(f"Page Inches per Pixel (Width): {page_inches_per_pixel_desired_width} in/px W")
    print(f"Page Inches per Pixel (Height): {page_inches_per_pixel_desired_height} in/px H")

    # Calculate the measurements in square feet
    measurement_color1 = pixel_count_color1 * square_feet_per_pixel

    # Calculate the total square footage in the image
    total_square_feet = pixel_count_color1 * square_feet_per_pixel * slope_multiplier

    #-TBD

    # Display results
    #print(f"Assumed Scale Factor: 1/{int(1/adjusted_scale_factor)} inch per foot")
    #print(f"Linear feet per pixel: {real_linear_feet_per_pixel} lf/pixel")
    print(f"Square Feet per Pixel: {square_feet_per_pixel:.6f} sq. ft/pixel")
    print(f"Total Pixels in the Image: {total_pixels}")
    print(f"Total Square Feet on the Roof: {total_square_feet:.2f} sq. ft")

    # Display the adjusted scale
    #print(f"Adjusted Scale Factor: 1/{int(1/adjusted_scale_factor)} inch per foot")

    # print(f"Measurement of Void: {measurement_color2:.2f} sq. ft.")  # Change label for Color 1
    print(f"Measurement of Roof Area - Plane: {measurement_color1:.2f} sq. ft.")  # Change label for Color 2


    ################################################################################
    ## DETECT & ISOLATE THE FEATURES USING COLOR CODES, THEN DEFINE LINE SEGMENTS ##
    ################################################################################


    # Define the BGR color codes for each roof feature on np image
    ## Final version will have slight hue adjustments to the color code due to landingai processing differences
    color_codes = {
        "Ridge": ([254, 191, 191], [255, 193, 193]),  
        "Hip": ([191, 223, 191], [193, 225, 193]),  
        "Valley": ([254, 232, 191], [255, 234, 193]), 
        "Eave": ([191, 191, 254], [193, 193, 255]), 
        "Rake": ([224, 192, 224], [224, 192, 224]),
                
        # Add other colors for features as needed in the future (Rake, Roof to Wall, Step Flashing, etc.)
    }

    # Features to apply the slope multiplier to
    features_to_scale = ["Rake", "Valley", "Hip"]

    # Tolerance value
    tolerance = 20
    response_dict['measurement_of_roof_area'] = round(measurement_color1, 2)
    response_dict['total_number_of_pixel_of_image'] = total_pixels
    scale_factor = ((default_height_inches)/(1/8))/(gray.shape[0])

    # Initialize the dictionary to store the lengths for each feature
    feature_lengths_dict = {}

    ###############################################################################
    ## LOOP THROUGH FEATURE DETECTION PROCESS FOR MEASURING EACH DESIRED FEATURE ##
    ###############################################################################

    # Define the tolerance for RGB values
    ## This is what works; keep unless other values/ranges tested further

    # Process each color code
    for feature, (lower_color, upper_color) in color_codes.items():
        # Ensure the color bounds are lists with 3 values (RGB)
        if not (isinstance(lower_color, list) and isinstance(upper_color, list) and len(lower_color) == 3 and len(upper_color) == 3):
            print(f"Invalid color definition for feature {feature}: {lower_color}, {upper_color}")
            continue

        # Adjust the bounds for each color using the tolerance
        adjusted_lower_bound = [max(c - tolerance, 0) for c in lower_color]
        adjusted_upper_bound = [min(c + tolerance, 255) for c in upper_color]

        # Convert the adjusted bounds to numpy arrays
        lower_bound = np.array(adjusted_lower_bound)
        upper_bound = np.array(adjusted_upper_bound)

        # Create a binary mask for the color
        mask = cv2.inRange(overlayed_image3_np, lower_bound, upper_bound)
        
        #Kernels
        kernel_size = 3  # Adjust the size to remove smaller objects
        kernel = np.ones((kernel_size, kernel_size), np.uint8) #Remove Noise
        
        reduced_mask = cv2.erode(mask, kernel, iterations=1)
        median = cv2.medianBlur(reduced_mask, 3)
        
        
        ##Before applying Hough Line Transform, preprocess the mask
        kernel = np.ones((3,3), np.uint8)  # 5x5 kernel for dilation, adjust size as needed
    
        dilated_mask = cv2.dilate(median, kernel, iterations = 4)  # You can adjust iterations if needed

        # Find contours from the binary masks
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize list for storing contour perimeters for each feature
        feature_lengths = []
        
        for contour in contours:
            # Calculate and store the perimeter of each object contour within the mask
            perimeter = cv2.arcLength(contour, True)  # True indicates that the contour is closed
            feature_lengths.append(perimeter)
        
        
        # Sum the perimeters of all contours to get total length
        total_feature_length = sum(feature_lengths)
        # Adjust length based on your scale factor or specific needs
        scaled_length = total_feature_length * scale_factor * 0.5 # Adjust scale_factor accordingly; ## 0.5 is to cut the contour length in half, which ends up being the true length
        
        # Store the length in the dictionary
        feature_lengths_dict[feature] = scaled_length

        response_dict[f"{feature}"] = round(scaled_length, 2)

        # Check if the specified features should have the multiplier applied
        if feature in features_to_scale:
            scaled_length *= slope_multiplier

        # Store the new adjusted lengths in the dictionary    
        feature_lengths_dict[feature] = scaled_length
        response_dict[f"{feature}"] = round(scaled_length, 2)

    print(response_dict)
    return response_dict