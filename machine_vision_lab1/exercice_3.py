import cv2 as cv
import numpy as np
import os
import sys

image_path = r"H:\Image_analy\Sensor_techno\machine_vision_lab1\Images_Lab_1\piece03.png"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

def img_show(img):
    cv.imshow("original", img )
    cv.waitKey(0)
    cv.destroyAllWindows()

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

img = resize_image(img, 10)


different_tresh= []
def diff_imread(img):
    ret, tresh = cv.threshold(img,70,100,cv.THRESH_BINARY)
    different_tresh.append(tresh)
    ret, tresh = cv.threshold(img,70,100,cv.THRESH_BINARY_INV)
    different_tresh.append(tresh)
    ret, tresh = cv.threshold(img,100,255,cv.THRESH_TRUNC)
    different_tresh.append(tresh)
    ret, tresh = cv.threshold(img,100,255,cv.THRESH_TOZERO)
    different_tresh.append(tresh)
    ret, tresh = cv.threshold(img,100,255,cv.THRESH_TOZERO_INV)
    different_tresh.append(tresh)

diff_imread(img)



for img in different_tresh:
    #img = resize_image(img, 50)
    img_show(img)

stacked_vertically = np.vstack(different_tresh)

stacked_horizontally = np.hstack(different_tresh)

cv.imshow('Vertical Stack', stacked_vertically)
cv.waitKey(0)

cv.imshow('Horizontal Stack', stacked_horizontally)
cv.waitKey(0)

cv.destroyAllWindows()

###########ANSWER#########
#After stacking vertically or horizontally, 
#the function imshow reads and display the picture horizontally,
#the 5 picture in a row, same for vertically but in vertical 
#
#
#
##########################