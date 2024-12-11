import cv2 as cv
import numpy as np
import os
import sys

image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\machine_vision_lab1\Images_Lab_1\star.png"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img = cv.imread(image_path)

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

img = resize_image(img, 200)

def img_show(img):
    cv.imshow("original", img )
    cv.waitKey(0)
    cv.destroyAllWindows()

img_show(img)

kernel = (5,5)

blurred = cv.GaussianBlur(img, (5, 5), 0)

edges = cv.Canny(blurred, 50, 150)

edges = cv.dilate(edges,kernel )


img_show(edges)

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv.contourArea)


leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])  # Min x
rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])  # Max x
topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])  # Min y
bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])  # Max y
top_left = tuple(largest_contour[np.argmin(largest_contour[:, :, 0] + largest_contour[:, :, 1])][0])

output = img.copy()

summit_points = [leftmost, rightmost, topmost, bottommost, top_left]
central_x = int(sum([p[0] for p in summit_points]) / len(summit_points))
central_y = int(sum([p[1] for p in summit_points]) / len(summit_points))
central_point = (central_x, central_y)

cv.circle(output, central_point, 5, (0, 255, 255), -1)
cv.circle(output, leftmost, 5, (255, 0, 0), -1)  
cv.circle(output, rightmost, 5, (0, 255, 0), -1)  
cv.circle(output, topmost, 5, (0, 0, 255), -1)  
cv.circle(output, bottommost, 5, (255, 255, 0), -1)  
cv.circle(output, top_left, 5, (255, 0, 255), -1)

cv.imshow("Extremum Points", output)
cv.waitKey(0)
cv.destroyAllWindows()

print("Central Point of Summits:", central_point)


########BINARIZATION############

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


kernel = (5,5)

_, binary = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV )

img_show(binary)

contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv.contourArea)


leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])  # Min x
rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])  # Max x
topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])  # Min y
bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])  # Max y
top_left = tuple(largest_contour[np.argmin(largest_contour[:, :, 0] + largest_contour[:, :, 1])][0])



output = img.copy()

summit_points = [leftmost, rightmost, topmost, bottommost, top_left]
central_x = int(sum([p[0] for p in summit_points]) / len(summit_points))
central_y = int(sum([p[1] for p in summit_points]) / len(summit_points))
central_point = (central_x, central_y)

cv.circle(output, central_point, 5, (0, 255, 255), -1)
cv.circle(output, leftmost, 5, (255, 0, 0), -1)  
cv.circle(output, rightmost, 5, (0, 255, 0), -1)  
cv.circle(output, topmost, 5, (0, 0, 255), -1)  
cv.circle(output, bottommost, 5, (255, 255, 0), -1)  
cv.circle(output, top_left, 5, (255, 0, 255), -1)

cv.imshow("Extremum Points", output)
cv.waitKey(0)
cv.destroyAllWindows()

print("Central Point of Summits:", central_point)




#########ANSWER############
#Canny : 
#while using canny filter,
#i got: 176,227 coordinate, 
# Firstl i applied gaussian blurr then canny, and then dilation, 
# then to have th coordinate I  used cv.contours and then finding the corner of the star 
#after that I found the middle of all the corner by averaging the sum of the X coordinate then y coordinate and that gave us the middle of the star
#
#
#Binarization : I did the same thing as canny but with the binarizatin and without any other filters, 
#got as coordinate : 177,226
#
#
###########################