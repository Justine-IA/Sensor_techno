import cv2 as cv
import numpy as np
import os
import sys

image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\machine_vision_lab1\Images_Lab_1\piece05.png"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

img = resize_image(img, 20)


def img_show(img):
    cv.imshow("original", img )
    cv.waitKey(0)
    cv.destroyAllWindows()

img_show(img)


blurred = cv.GaussianBlur(img, (9,9), 0)
# Apply a gaussian blur that smooth the images to reduce noise and details

treshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,2)
#Apply an adaptative treshhold to convert the images to a black and white image to better detect forms and edge

median = cv.medianBlur(treshold, 5)
img_show(median)
#Median is very useful, to remove little noise like salt pepper noise that we have in the image
#even after the adaptative treshhold to remove any non wanted pixels

edges = cv.Canny(median, 50, 150)
img_show(edges)
#We use it to detect edges, by identifying area where the intensity changes

kernel = np.ones((2, 2), np.uint8)
dilated = cv.dilate(edges, kernel, iterations=1)
#we use dilation to expand white region to have the edges discoverd by canny more visible


contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#finally we use find contours to detects and draw the boundary of object in the binary images that we have now 


output = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(output, contours, -1, (0, 255, 0), 2)
img_show(output)



###########ANSWER###########
#Filter explanation : 
#   Canny : Best for detecting precise edge when gradient are changing abruptly
#   Adaptative tresholidng :best to binarize the image with uneven lightning condition
#   Which filter to choose : the best is to use canny after blur like gaussian blur, and to use 
#   Adapatative tresholding to separate object from background with different light and brightness 
#
#To improve image quality when aquiring it would be to :
#   Have good lightning to minimize shadow because for example in our final image the object on top mistake the shade for the edge
#   Have a good focus on what we want to analyze later
#   Have good contrast between object and the background
#   Using a high resoltuion camera for more precision
############################