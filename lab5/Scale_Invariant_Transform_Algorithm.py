import cv2 as cv
import sys
import os
import numpy as np 
image_path = r"H:\Image_analy\Sensor_techno\lab5\pics_Lab5\testImg.jpg"
image_path1 = r"H:\Image_analy\Sensor_techno\lab5\pics_Lab5\classB\b2.tiff"
image_path2 = r"H:\Image_analy\Sensor_techno\lab5\pics_Lab5\classB\b3.tiff"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img0 = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
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

for img in class_A:

    # img_erode = cv.erode(img, kernel)

    # img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    # boundary_img = cv.subtract(img_opening, img_erode)

    # blurred = cv.GaussianBlur(boundary_img, ksize,1.4)

    # edges = cv.Canny(img, 50 , 150)

    _, tresh = cv.threshold(img, 100, 230, cv.THRESH_BINARY_INV)    
    cv.imshow("idk",tresh)
    cv.waitKey(0)
    keypoints = sift.detect(tresh, None)# Detect keypoints

    # Draw keypoints on the image
    output_img = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image with keypoints
    cv.imshow(f"Keypoints ", output_img)
    cv.waitKey(0)

cv.destroyAllWindows()