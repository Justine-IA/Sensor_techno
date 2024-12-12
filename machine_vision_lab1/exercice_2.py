import cv2 as cv
import numpy as np
import os
import sys

image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\machine_vision_lab1\Images_Lab_1\colormap.jpg"

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

valuePix1=img[25,400]
print("pixel 1",valuePix1)

valuePix2=img[130,230]
print("pixel 2",valuePix2)

valuePix3=img[300,400]
print("pixel 3",valuePix3)

valuePix4=img[340,140]
print("pixel 4",valuePix4)

img[25,400] = [0,0,0]
img[130,230] = [0,0,0]
img[300,400] = [0,0,0]
img[340,140] = [0,0,0]

img_show(img)


##########ANSwER#############
#first pixel coordinate: 25, 400 
# RGB Value : 0,0,201 (opencv BGR value:201,0,0)
#second pixel coordinate: 130, 230, 
# RGB value :0,254,255 (opencv BGR value :255,254,0)
#third pixel coordinate:300, 400, 
# RGB value :255,103,154 (opencv BGR value: 154,103,255)
#fourth pixel coordinate:340, 140, 
# RGB value :152,203,0 (opencv BGRvalue :0, 203,152)
##############################