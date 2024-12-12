import cv2 as cv 
import numpy as np
import os
import sys

image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\machine_vision_lab1\Images_Lab_1\piece03.png"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)


def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

img = resize_image(img, 10)

def nothing(x):
    pass

cv.namedWindow("Trackbars")

cv.createTrackbar("Lower", "Trackbars", 0, 255, nothing)
cv.createTrackbar("Upper", "Trackbars", 0, 255, nothing)

while True : 
    
    lower = cv.getTrackbarPos("Lower", "Trackbars")
    upper = cv.getTrackbarPos("Upper", "Trackbars")

    _, binary = cv.threshold(img, lower, upper, cv.THRESH_BINARY)
    _, binary_inv = cv.threshold(img, lower, upper, cv.THRESH_BINARY_INV)
    _, trunc = cv.threshold(img, lower, upper, cv.THRESH_TRUNC)
    _, tozero = cv.threshold(img, lower, upper, cv.THRESH_TOZERO)
    _, tozero_inv = cv.threshold(img, lower, upper, cv.THRESH_TOZERO_INV)

    stacked = np.hstack([binary, binary_inv, trunc, tozero, tozero_inv])

    cv.imshow("stack with track", stacked)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

<<<<<<< HEAD
cv.imshow('Vertical Stack', stacked_vertically)
cv.waitKey(0)

cv.imshow('Horizontal Stack', stacked_horizontally)
=======

>>>>>>> b84ec606b220cf6159c7794bf6a733c956017eff
cv.waitKey(0)

cv.destroyAllWindows()

###########ANSWER#########
<<<<<<< HEAD
#After stacking vertically or horizontally, 
#the function imshow reads and display the picture horizontally,
#the 5 picture in a row, same for vertically but in vertical 
#
#
#
=======
#after stacking vertically or horizontally, 
#the function imshow reads and display the picture horizontally,
#the 5 picture in a row, same for vertically but in vertical 
#Treshold value range for all object to be visible : lower 62 to 160, upper 30 to 255
#Because grayscale is only from 0 to 255, 
#everything outside this range will not be useful as it will go back to 0 or 255
>>>>>>> b84ec606b220cf6159c7794bf6a733c956017eff
##########################