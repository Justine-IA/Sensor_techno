import cv2 as cv
import sys
import numpy as np 
img = cv.imread(cv.samples.findFile("img.tif"))
if img is None:
    sys.exit("Could not read the image.")
k = cv.waitKey (0)
if k == ord("s"):
    cv.imwrite("savedImg.tif", img)

yourImg=cv.imread("img.tif")
cv.imshow("first", yourImg)
cv.waitKey(0)
# valuePix1=yourImg[0,0]
print(yourImg.shape)
print(yourImg.dtype)
print(np.max(img))


float_img = np.float32(img)
m= np.max(float_img)
c = 255/(np.log(1+m))

log_img = yourImg

log_img = c * np.log(1 + float_img)  

log_img_normalized = cv.normalize(log_img, None, 0, 255, cv.NORM_MINMAX)
log_img_uint8 = np.uint8(log_img_normalized)

cv.imshow("first", yourImg)
cv.imshow("log", log_img_uint8)
cv.waitKey(0)

cv.destroyAllWindows()