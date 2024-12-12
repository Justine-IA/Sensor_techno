import cv2 as cv
import sys
import os
import numpy as np 
image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\lab5\pics_Lab5\classB\b1.tiff"
image_path1 = r"C:\Users\Jean\Documents\Suede\Sensor_techno\lab5\pics_Lab5\classB\b2.tiff"
image_path2 = r"C:\Users\Jean\Documents\Suede\Sensor_techno\lab5\pics_Lab5\classB\b3.tiff"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img0 = cv.imread(image_path)
img1 = cv.imread(image_path1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)

class_A =[img0,img1,img2]

for i in class_A:
    if i is None:
        sys.exit("Error: Could not read the image file.")

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)


ksize = (9,9)
kernel =cv.getStructuringElement(cv.MORPH_CROSS,ksize)

# Create SIFT object
sift = cv.SIFT_create()

for idx, img in enumerate(class_A):

    # img_erode = cv.erode(img, kernel)

    # img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    # boundary_img = cv.subtract(img_opening, img_erode)

    # blurred = cv.GaussianBlur(boundary_img, ksize,1.4)

    # edges = cv.Canny(blurred, 50 , 150)


    # Detect keypoints
    keypoints = sift.detect(img, None)

    # Draw keypoints on the image
    output_img = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image with keypoints
    cv.imshow(f"Keypoints {idx+1}", output_img)
    cv.waitKey(0)

    # Save the output image (optional)
    cv.imwrite(f"sift_keypoints_{idx+1}.png", output_img)

cv.destroyAllWindows()