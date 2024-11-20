import cv2 as cv
import sys
import numpy as np 
img = cv.imread(cv.samples.findFile("img.tif"))
if img is None:
    sys.exit("Could not read the image.")
# cv.imshow("Display window", img)
k = cv.waitKey (0)
if k == ord("s"):
    cv.imwrite("savedImg.tif", img)

yourImg=cv.imread("img.tif")
cv.imshow("first", yourImg)
cv.waitKey(0)

normalized_img = img / 255.0
gamma_range = [0.1,0.5,1.5,2]
for i in gamma_range:
    gamma = i
    gamma_corrected = np.array(255*(yourImg/255)**gamma)
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    cv.imshow("gamma", gamma_corrected)
    cv.waitKey(0)

cv.destroyAllWindows()