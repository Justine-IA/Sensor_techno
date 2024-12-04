import cv2 as cv
import sys
import os
import numpy as np 
image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\lab5\pics_Lab5\classA\a1.tiff"
image_path1 = r"C:\Users\Jean\Documents\Suede\Sensor_techno\lab5\pics_Lab5\classA\a2.tiff"
image_path2 = r"C:\Users\Jean\Documents\Suede\Sensor_techno\lab5\pics_Lab5\classA\a3.tiff"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img0 = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
img1 = cv.imread(image_path1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)

class_A =[img0]

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

img_tresh = []

for i in class_A:
    img_erode = cv.erode(i, kernel)

    img_opening = cv.morphologyEx(i, cv.MORPH_OPEN, kernel)

    boundary_img = cv.subtract(img_opening, img_erode)
    cv.imshow("boundary Image", boundary_img)
    cv.waitKey(0)

    # blurred_image = cv.GaussianBlur(boundary_img, (5, 5), 0)
    blurred_image = cv.bilateralFilter(boundary_img, 9, 75, 75)

    # ret, tresh = cv.threshold(blurred_image, 30,255,cv.THRESH_BINARY)
    tresh = cv.adaptiveThreshold(
    blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
)
    tresh = cv.bitwise_not(tresh)

    img_tresh.append(tresh)
    cv.imshow("test",tresh)
    cv.waitKey(0)

    tresh_median = cv.medianBlur(tresh, 9)


    close_img = cv.morphologyEx(tresh_median, cv.MORPH_CLOSE, kernel)
    close_img = cv.morphologyEx(tresh_median, cv.MORPH_CLOSE, kernel)

    median_opening = cv.medianBlur(close_img,3,0)

    cv.imshow("idk", median_opening)
    cv.waitKey(0)








