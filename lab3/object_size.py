import cv2 as cv
import sys
import numpy as np 
img = cv.imread(cv.samples.findFile("img2.tif"))
if img is None:
    sys.exit("Could not read the image.")

# cv.imshow("img", img)
# cv.waitKey(0)
print(img.shape)

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

img = resize_image(img, 50)  
cv.imshow("Original Image", img)
cv.waitKey(0)

ret, img_tresh = cv.threshold(img,127,255,cv.THRESH_BINARY)
cv.imshow("tresh img", img_tresh)
cv.waitKey(0)

ksize = (5, 5)
kernel = cv.getStructuringElement(cv.MORPH_CROSS,ksize)
x = [img_tresh]

for i in x:
    img_erode = cv.erode(i, kernel)
    cv.imshow("Eroded Image", img_erode)
    cv.waitKey(0)

    img_opening = cv.morphologyEx(i, cv.MORPH_OPEN, kernel)
    cv.imshow("Opening Image", img_opening)
    cv.waitKey(0)

    boundary_img = cv.subtract(img_opening, img_erode)
    cv.imshow("boundary Image", boundary_img)
    cv.waitKey(0)

indices = np.where(boundary_img == 255)
rows, cols = indices[0], indices[1]

leftmost = np.min(cols)
rightmost = np.max(cols)
topmost = np.min(rows)
bottommost = np.max(rows)

width = rightmost - leftmost
height = bottommost - topmost

centroid_row = np.mean(rows)
centroid_col = np.mean(cols)

print(f"Width: {width} pixels")
print(f"Height: {height} pixels")
print(f"Centroid: ({centroid_row:.2f}, {centroid_col:.2f})")

output_img = boundary_img
cv.rectangle(output_img, (leftmost, topmost), (rightmost, bottommost), (0, 255, 0), 2)  # Draw bounding box
cv.circle(output_img, (int(centroid_col), int(centroid_row)), 5, (0, 0, 255), -1)  # Draw centroid
cv.imshow("Melt Pool Analysis", output_img)


cv.waitKey(0)
cv.destroyAllWindows()