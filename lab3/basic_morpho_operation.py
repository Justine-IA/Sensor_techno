import cv2 as cv
import sys
import numpy as np 
img1 = cv.imread(cv.samples.findFile("img1.tif"))
if img1 is None:
    sys.exit("Could not read the image.")
img2 = cv.imread(cv.samples.findFile("img2.tif"))
if img2 is None:
    sys.exit("Could not read the image.")


img1=cv.imread("img1.tif")
cv.imshow("img1", img1)
cv.waitKey(0)

img2=cv.imread("img2.tif")
cv.imshow("img2", img2)
cv.waitKey(0)

print(img1.shape)


ksize = (5, 5)
kernel = cv.getStructuringElement(cv.MORPH_CROSS,ksize)
x = [img1]

for i in x:
    img_erode = cv.erode(i, kernel)
    cv.imshow("Eroded Image", img_erode)
    cv.waitKey(0)

    img_dilate = cv.dilate(i, kernel)
    cv.imshow("Dilated Image", img_dilate)
    cv.waitKey(0)

    img_opening = cv.morphologyEx(i, cv.MORPH_OPEN, kernel)
    cv.imshow("Opening Image", img_opening)
    cv.waitKey(0)

    img_closing = cv.morphologyEx(i, cv.MORPH_CLOSE, kernel)
    cv.imshow("Closing Image", img_closing)
    cv.waitKey(0)

    img_hitormiss = cv.morphologyEx(i, cv.MORPH_HITMISS,kernel)
    cv.imshow("hit or miss Image", img_hitormiss)
    cv.waitKey(0)

    boundary_img = cv.subtract(i, img_erode)
    cv.imshow("boundary Image", boundary_img)
    cv.waitKey(0)


cv.destroyAllWindows()