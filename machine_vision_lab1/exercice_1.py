import cv2 as cv
import numpy as np
import os
import sys

image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\machine_vision_lab1\Images_Lab_1\original.png"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img = cv.imread(image_path)

def img_show(img):
    cv.imshow("original", img )
    cv.waitKey(0)
    cv.destroyAllWindows()


img_show(img)
print(img.shape)
print(img.dtype)

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

print(img.shape)
print(img.dtype)


########ANSWER#########
#image height : 540
#image widht : 1050
#image dtype : uint8
#The image is in RGB, it has 3 color channel instead of 1 for gray scale
#######################
