import cv2 as cv
import numpy as np
import os
import sys

image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\machine_vision_lab1\Images_Lab_1\colormap.jpg"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img = cv.imread(image_path)

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

img = resize_image(img, 70)


def img_show(img):
    cv.imshow("original", img )
    cv.waitKey(0)
    cv.destroyAllWindows()

#GRAYSCALE CHANNEL
img_canal_0 = img[:,:,0]
img_canal_1 = img[:,:,1]
img_canal_2 = img[:,:,2]

stacked_grey = np.hstack([img_canal_2, img_canal_1, img_canal_0])

cv.imshow("grey", stacked_grey)
cv.waitKey(0)
cv.destroyAllWindows()


#COLOR CHANNEL
color_channel_0 = np.zeros_like(img)
color_channel_1 = np.zeros_like(img)
color_channel_2 = np.zeros_like(img)

color_channel_0[:, :, 0] = img[:, :, 0]  # Blue channel
color_channel_1[:, :, 1] = img[:, :, 1]  # Green channel
color_channel_2[:, :, 2] = img[:, :, 2]  # Red channel

stacked_color = np.hstack([color_channel_2, color_channel_1, color_channel_0])


img_show(stacked_color)
