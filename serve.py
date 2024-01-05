import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
#import boto3
from PIL import Image
import io
from matplotlib import pyplot as plt
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

from typing import Union
from fastapi import FastAPI, Response, File
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins =[
    'http://127.0.0.1:5500/',
    'http://localhost',
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
## NEXT 5 Clusters are for Preparing S3 images saved for loading to EC2 instance prior to running the remainder. 
## To be adjusted later based on workflow requirements - this is for demo usage only until we finalize

# Initialize a Boto3 S3 client
#s3 = boto3.client('s3')

# Specify your S3 bucket and the image keys
# bucket_name = 'opticostdatabase'
# image_key_Base = 'Test CV Images/roof112.jpg'
# image_key_Processed = 'Test CV Images/roof lines AI demo3.PNG'
# image_key_PreProcessed = 'Test CV Images/Roof Area Prediction.png'

# # Retrieve and read the first image
# obj1 = s3.get_object(Bucket=bucket_name, Key=image_key_Base)
# image_content1 = obj1['Body'].read()
# image1 = Image.open(io.BytesIO(image_content1))

# # Retrieve and read the second image
# obj2 = s3.get_object(Bucket=bucket_name, Key=image_key_Processed)
# image_content2 = obj2['Body'].read()
# image2 = Image.open(io.BytesIO(image_content2))

# # Retrieve and read the third image
# obj3 = s3.get_object(Bucket=bucket_name, Key=image_key_PreProcessed)
# image_content3 = obj3['Body'].read()
# image3 = Image.open(io.BytesIO(image_content3))

@app.post("/analyze")
def analyze(response: Response, file: bytes = File(...)):
    print("Request received...")
    response_dict = {"Success":False}
    # Enter your API Key and Secret (for Area Detection)
    endpoint_id = "54ab4c09-ef9f-4ddd-bc84-71361a0d16d7"
    api_key = "land_sk_h4VB3GlORHrUrd8MVsihDJjcKy9x128xwCKD8YcV1gzKjnJ7cZ"

    # Load your image
    user_uploaded_image = Image.open(io.BytesIO(file))
    print('user uploaded image',user_uploaded_image)
    user_uploaded_image.show()
    # Convert PIL Image to OpenCV format
    image1_np = np.array(user_uploaded_image)

    # Convert the image from RGB to BGR (OpenCV uses BGR)
    image1_bgr = cv2.cvtColor(image1_np, cv2.COLOR_RGB2BGR)
    #image2_bgr = cv2.cvtColor(image2_np, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    converted_to_gray_scale = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)

    # Run inference
    #predictor = Predictor(endpoint_id, api_key=api_key)
    #ai_landing_predictions = predictor.predict(converted_to_gray_scale)
    #ai_landing_predictions_overlay = overlay_predictions(ai_landing_predictions, user_uploaded_image) #result of this is image 3
    #print('first prediction with overlay, this is image 3',#ai_landing_predictions_overlay)
    #ai_landing_predictions_overlay.show()
    # Enter your API Key (for Colored Segment Detection)
    #endpoint_id2 = "aa131064-09f2-476f-b7e7-704887b5b1b0"
    #api_key2 = "land_sk_39Z470ya52y2sJ0gzYlzFNZuif8XdGr0Z1t8AAdnsbeJd1B1wu"

    # Run inference
    #predictor2 = Predictor(endpoint_id2, api_key=api_key2)
    #predictions2 = predictor2.predict(converted_to_gray_scale) #result of this is image 2
    #predictions2_overlay = overlay_predictions(predictions2, #user_uploaded_image)
    #print('second prediction with overla, this is image 2',#predictions2_overlay)
    predictions2_overlay.show() 
    # Example of processing image1 and getting an image with predictions
    #image_with_preds = ai_landing_predictions 

    # continue the above example to save
    # image_with_preds = overlay_predictions(predictions, image)
    # image_with_preds.save(r"C:\Users\LPlazola\Desktop\Roof Area Prediction.png")
    #image_with_preds.save("/home/ec2-user/Roof_Area_Prediction.png")


    # Convert image_with_preds to OpenCV format for further processing
    image_with_preds_np = np.array(ai_landing_predictions_overlay) #feed image3 or result of first prediction overlay here
    #print('image with pred np',image_with_preds_np)
    image_with_preds_cv2 = image_with_preds_np[:, :, ::-1]

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image_with_preds_cv2, cv2.COLOR_BGR2HSV)

    # Calculate the histogram of the Hue channel (H)
    hue_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])

    # Find peaks in the histogram (you can adjust the threshold)
    peaks = np.where(hue_hist > 2000)[0]

    # Determine color ranges based on the peaks
    color_ranges = []
    start = peaks[0]
    for i in range(1, len(peaks)):
        if peaks[i] != peaks[i-1] + 1:
            end = peaks[i-1]
            color_ranges.append((start, end))
            start = peaks[i]

    # Add the last color range
    end = peaks[-1]
    color_ranges.append((start, end))

    # Define a mapping of Hue values to color names
    color_mapping = {
        (1, 86): "Void",  # Change label for Color 1
        (87, 88): "Other Color",  # Adjust Label (if needed) 
        (89, 148): "Roof Area - Plane",  # Change label for Color 2
        (149, 150): "Green",
        (151, 210): "Cyan",
        (211, 270): "Blue",
        (271, 330): "Magenta",
    }

    # Print the determined color ranges and their corresponding color names
    for i, (start, end) in enumerate(color_ranges):
        color_name = None
        for (range_start, range_end), name in color_mapping.items():
            if range_start <= start <= range_end and range_start <= end <= range_end:
                color_name = name
                break
        print(f"Color Range {i+1}: H={start}-{end} ({color_name})")

    # Calculate the proportion of Color 1 (Hue 5-50, VOID)
    color1_proportion = 0
    for start, end in color_ranges:
        if 1 <= start <= 50 and 1 <= end <= 50:
            color1_proportion += (end - start + 1)  # Length of the color range
    color1_proportion /= hsv_image.size

    # Calculate the proportion of Color 2 (Hue 51-99, ROOF)
    color2_proportion = 0
    for start, end in color_ranges:
        if 51 <= start <= 150 and 51 <= end <= 150:
            color2_proportion += (end - start + 1)  # Length of the color range
    color2_proportion /= hsv_image.size

    # Print color proportions
    ##print(f"Proportion of Void: {color1_proportion:.2f}")  # Change label for Color 1
    ##print(f"Proportion of Plane Roof Area: {color2_proportion:.2f}")  # Change label for Color 2

    # Calculate the number of pixels for each color range
    pixel_count_color1 = 0
    pixel_count_color2 = 0

    for (start, end) in color_ranges:
        for i in range(start, end + 1):
            if 1 <= i <= 50:
                pixel_count_color1 += np.sum(hsv_image[:, :, 0] == i)
            elif 51 <= i <= 150:
                pixel_count_color2 += np.sum(hsv_image[:, :, 0] == i)


    ## Create page size selections (e.g. 24" x 36" default)

    # Define the default input image size (24" high by 36" wide is standard, but could be similar 2:3 proportional sizes, such as 12"x18", or odd sizes such as 8.5" x 11")
    default_height_inches = 24
    default_width_inches = 36

    # Define the default scale (1/4" on the image = 1 foot in real life)
    default_scale_factor = (3) / (8.0)  # Adjust as needed

    # Define the desired height and width for final calculation purposes
    desired_height_inches = 24  # Adjust as needed
    desired_width_inches = 36   # Adjust as needed

    # Calculate the scaling factors for height and width
    height_scale_factor = desired_height_inches / default_height_inches
    width_scale_factor = desired_width_inches / default_width_inches

    # Adjust the default scale based on the desired size
    adjusted_scale_factor = default_scale_factor * (height_scale_factor + width_scale_factor) / 2

    # Convert PIL Image to NumPy array (in RGB format) this is a duplicated code with line 89
    image_with_preds_np = np.array(ai_landing_predictions_overlay)

    # Convert from RGB to BGR format for OpenCV
    image_with_preds_bgr = cv2.cvtColor(image_with_preds_np, cv2.COLOR_RGB2BGR)

    # Assuming width_scale_factor and height_scale_factor are defined
    adjusted_image = cv2.resize(image_with_preds_bgr, None, fx=width_scale_factor, fy=height_scale_factor)


    ## Perform calculations or measurements using the adjusted image

    # Calculate the scaling factors for height and width
    height_scale_factor = image_with_preds_np.shape[0] / default_height_inches
    width_scale_factor = image_with_preds_np.shape[1] / default_width_inches

    # Count the total number of pixels in the image
    total_pixels = image_with_preds_np.shape[0] * image_with_preds_np.shape[1]

    # Calculate the square footage per pixel
    square_feet_per_pixel =  (((desired_width_inches) * (1/default_scale_factor)) * ((desired_height_inches) * (1/default_scale_factor))) / (total_pixels)

    # Define the conversion factor (square feet per pixel)
    #conversion_factor = 0.00385  # Example: 0.01 square feet per pixel

    # Calculate the measurements in square feet
    measurement_color1 = pixel_count_color1 * square_feet_per_pixel
    measurement_color2 = pixel_count_color2 * square_feet_per_pixel

    # Calculate the total square footage in the image
    total_square_feet = pixel_count_color1 * square_feet_per_pixel

    ## Apply roof slope as a multiplier to the roof area measurement

    #-TBD

    ## Create page scale options (e.g. 1/4" = 1', 1/2" = 1', 3/8" = 1', etc.)

    #-TBD

    ## Print measurements based on user selections

    #-TBD

    ## Create the main window

    #-TBD

    # Display results
    # print(f"Assumed Scale Factor: 1/{int(1/adjusted_scale_factor)} inch per foot")
    # print(f"Square Feet per Pixel: {square_feet_per_pixel:.6f} sq. ft/pixel")
    # print(f"Total Pixels in the Image: {total_pixels}")
    # print(f"Total Square Feet on the Roof: {total_square_feet:.2f} sq. ft")

    # Display the adjusted scale
    print(f"Adjusted Scale Factor: 1/{int(1/adjusted_scale_factor)} inch per foot")

    # Print flat measurements of Roof and Void Areas

    # print(f"Measurement of Void: {measurement_color2:.2f} sq. ft.")  # Change label for Color 1
    print(f"Measurement of Roof Area - Plane: {measurement_color1:.2f} sq. ft.")  # Change label for Color 2

    # Add the square foot result to the image
    # result_text = f"Total Roof Area: {total_square_feet:.2f} sq. ft"
    # cv2.putText(image_with_preds, result_text, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    # Save or display the updated image with the result
    #1# cv2.imwrite("output_image.jpg", image_with_preds)  # Save the image with the Total Area result
    # cv2.namedWindow("Image with Result", cv2.WINDOW_NORMAL) 
    #1# cv2.imshow("Image with Result", image_with_preds)  # Display the image with the result
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    ## Display the adjusted image
    #cv2.imshow("Adjusted Image", adjusted_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    response_dict['measurement_color'] = measurement_color1


    # Load the image
    ## This should be same as the OUTPUT image of the previous "roof area prediction.png" @ line 28
    ## Currently using the locally saved and uploaded version to shortcut until we can save the initial output and feed it directly here.
    predictions2_overlay_np = np.array(predictions2_overlay)
    predictions2_overlay_bgr = cv2.cvtColor(predictions2_overlay_np, cv2.COLOR_RGB2BGR)
    img = predictions2_overlay_bgr 
    #?# img = cv2.cvtColor(image2_bgr, cv2.COLOR_RGB2BGR)

    # if img is None:
    #     print("Failed to load the image.")
    #     exit()

    #imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Define the color codes for each roof feature
    ## Final version will have slight hue adjustments to the color code due to landingai processing differences
    color_codes = {
        "Ridge": ([55, 55, 183], [90, 90, 218]),  # Red
        "Eave": ([223, 223, 95], [231, 231, 97]),  # Light Blue
        "Hip": ([55, 183, 55], [123, 251, 123]),  # Green
        "Valley": ([63, 146, 191], [124, 207, 252]),  # Orange
        # Add other colors for features as needed in the future (Rake, Roof to Wall, Step Flashing, etc.)
    }


    # Function to skeletonize the image (reduce to single pixel width)
    def skeletonize(img):
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        while True:
            open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open_img)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    # Define the tolerance for RGB values
    ## This is what works; keep unless other values/ranges tested further
    tolerance = 40

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
        mask = cv2.inRange(img, lower_bound, upper_bound)
        
        #Kernels
        kernel_size = 5  # Adjust the size to remove smaller objects
        kernel = np.ones((kernel_size, kernel_size), np.uint8)#Remove Noise
        
        reduced_mask = cv2.erode(mask, kernel, iterations=1)
        median = cv2.medianBlur(reduced_mask, 5)
        
        # Apply morphological opening

        median2 = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
        
        blur = cv2.bilateralFilter(median2,9,150,150)
        
        ##Before applying Hough Line Transform, preprocess the mask
        kernel = np.ones((3,3), np.uint8)  # 5x5 kernel for dilation, adjust size as needed
        #
    
        
        # Skeletonize the mask
        #mask = skeletonize(dilated_mask)
        #
        eroded_image = cv2.erode(blur, kernel, iterations = 1)
        #
        median3 = cv2.medianBlur(eroded_image, 1)
        opened_image = cv2.morphologyEx(median3, cv2.MORPH_OPEN, kernel)
        opened_image2 = cv2.morphologyEx(opened_image, cv2.MORPH_OPEN, kernel) 
        
        dilated_mask = cv2.dilate(opened_image2, kernel, iterations = 2)  # You can adjust iterations if needed
        
    
        #Edge Detection
        #EdgeMask = cv2.Canny(mask, 5, 9)
        
        #
        #blurred_mask = cv2.GaussianBlur(dilated_mask, (3, 3), 1)  # 5x5 Gaussian kernel, adjust size as needed
        
        #closing = cv2.morphologyEx(opened_image2, cv2.MORPH_CLOSE, median)
        
        #Re-Skeletonized the mask
        skel_mask = skeletonize(opened_image2) #was not use in code
        
        ## These are the small graphical output tiles @ below the cell after processing
        #plt.subplot(121),plt.imshow(dilated_mask),plt.title('dilated_mask')
        #plt.xticks([]), plt.yticks([])
        #plt.subplot(122),plt.imshow(skel_mask),plt.title('skel_mask')
        #plt.xticks([]), plt.yticks([])
        #plt.show()
        
        # Invert the mask to make it printer-friendly
        ## mask = cv2.bitwise_not(EdgeMask)
        
        # Display the mask
        #cv2.imshow(f"Mask for {feature}", eroded_image) 
        #cv2.imshow(f"Mask for {feature}", dilated_mask)
        #cv2.imwrite(f"mask_for_{feature}.png", dilated_mask)  # This will save the masks with filenames like "mask_for_ridge.png", "mask_for_eave.png", etc., based on the feature
        #cv2.imwrite(f"mask_for_{feature}_skelly.png", skel_mask) THIS ONE SUCKS; NEEDS FURTHER CORRECTION BEFORE USING. USE DILATED FOR NOW 
        #cv2.waitKey(0)

        # Initialize the dictionary to store the lengths
        lengths = {}
        
        # Compute the length of the chain
        ## Uses Dilated Mask outputs previously generated
        length = cv2.countNonZero(dilated_mask)
        
        # Store the length in the dictionary
        lengths[feature] = length
        
        # Calculate and print the total number of pixels
        total_pixels = img.shape[0] * img.shape[1]
        print(f"Total number of pixels in the image: {total_pixels}")
        
        # Show total length of features 
        print(f"{feature} length: {length} pixels")
        
        # Calculate the linear feet total of each feature
        print(f"{feature} = {length*.00125} LF")
        #print([feature])