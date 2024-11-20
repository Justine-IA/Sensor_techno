import cv2 as cv
import sys
img = cv.imread(cv.samples.findFile("img.tif"))
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey (0)
if k == ord("s"):
    cv.imwrite("savedImg.tif", img)

yourImg=cv.imread("img.tif")
cv.imshow("first", yourImg)
cv.waitKey(0)

Inv_img = cv.bitwise_not(yourImg)
cv.imshow("inv", Inv_img)
cv.waitKey(0)

cv.destroyAllWindows()