import cv2 as cv
import sys
import numpy as np 
Gauss_img = cv.imread(cv.samples.findFile("gaussianNoiseImg.tif"))
if Gauss_img is None:
    sys.exit("Could not read the image.")
Pepper_img = cv.imread(cv.samples.findFile("peppersaltImg.tif"))
if Pepper_img is None:
    sys.exit("Could not read the image.")

Gauss_Img=cv.imread("gaussianNoiseImg.tif")
cv.imshow("Gauss", Gauss_Img)
cv.waitKey(0)

PepperSalt_Img=cv.imread("peppersaltImg.tif")
cv.imshow("Pepper salt noise", PepperSalt_Img)
cv.waitKey(0)

gaussian_blur_range = [5]
for i in gaussian_blur_range:
    Blur_Gauss_Img = cv.medianBlur(Gauss_Img,i,0)
    cv.imshow(f" Gauss median Blur {i}", Blur_Gauss_Img)
    Blur_PepperSalt_Img = cv.medianBlur(PepperSalt_Img,i,0)
    cv.imshow(f"Pepper-Salt median Blur {i}", Blur_PepperSalt_Img)
    cv.waitKey(0)

cv.destroyAllWindows()