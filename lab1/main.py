import cv2 as cv
import sys
img = cv.imread(cv.samples.findFile("img.tif"))
if img is None :
    sys.exit("Could not read the image .")
cv.imshow("Display window",img)
k = cv.waitKey(0)
if k == ord("s") :
    cv.imwrite(" savedImg.tif" ,img )
yourImg = cv.imread("img.tif")
print(yourImg.shape)
print(yourImg.dtype)


yourImg[0,0] = [0]

valuePix1=yourImg[0,0,0] = 255
yourImg[0,0] = valuePix1
print(valuePix1)
cv.imshow("blala", yourImg)
cv.waitKey(0)
ROI=yourImg[50:100,50:100]
cv.imshow('ROI',ROI)
cv.waitKey(0)
yourImg[50:100,50:100] = 255
cv.imshow("modify field in image", yourImg)
cv.waitKey(0)
cv.destroyAllWindows()