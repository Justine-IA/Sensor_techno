import cv2 as cv
import numpy as np
import os
import sys
import math

image_path = r"C:\Users\Jean\Documents\Suede\Sensor_techno\machine_vision_lab1\Images_Lab_1\tree.png"

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


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

########CANNY#########

edges = cv.Canny(gray, 50,150)
edges = cv.dilate(edges, (3,3))
img_show(edges)


contour_canny, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest_contour = max(contour_canny, key = cv.contourArea)

bottom_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

x_coords = largest_contour[:, :, 0].astype(float)
y_coords = largest_contour[:, :, 1].astype(float)

x_normalized = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
y_normalized = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())

# Compute a weighted score for each point
alpha = 0.6  
beta = 0.4   
scores = alpha * x_normalized + beta * y_normalized

# Find the index of the point with the highest score
best_point_idx = np.argmax(scores)
bottom_right = tuple(largest_contour[best_point_idx][0])


dx = bottom_right[0] - bottom_point[0]
dy = bottom_point[1] - bottom_right[1]  

angle_radians = math.atan2(dy, dx)  
angle_degrees = math.degrees(angle_radians)  

if angle_degrees < 0:
    angle_degrees += 360

output = img.copy()

cv.drawContours(output, [largest_contour], -1, (255, 0, 0), 2)  
cv.line(output, bottom_point, bottom_right, (0, 255, 0), 2)  
cv.circle(output, bottom_point, 5, (255, 255, 0), -1)  
cv.circle(output, bottom_right, 5, (0, 255, 255), -1)  

cv.imshow("Bounding Box and Orientation", output)
cv.waitKey(0)
cv.destroyAllWindows()

print(f"Angle for canny: {angle_degrees:.2f} degrees")


########binarization#########

_, binary = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV )

img_show(binary)

contour_canny, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest_contour = max(contour_canny, key = cv.contourArea)

bottom_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

x_coords = largest_contour[:, :, 0].astype(float)
y_coords = largest_contour[:, :, 1].astype(float)

x_normalized = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
y_normalized = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())

# Compute a weighted score for each point
alpha = 0.6  
beta = 0.4   
scores = alpha * x_normalized + beta * y_normalized

# Find the index of the point with the highest score
best_point_idx = np.argmax(scores)
bottom_right = tuple(largest_contour[best_point_idx][0])


dx = bottom_right[0] - bottom_point[0]
dy = bottom_point[1] - bottom_right[1]  

angle_radians = math.atan2(dy, dx)  
angle_degrees = math.degrees(angle_radians) 

if angle_degrees < 0:
    angle_degrees += 360


output = img.copy()

cv.drawContours(output, [largest_contour], -1, (255, 0, 0), 2)  
cv.line(output, bottom_point, bottom_right, (0, 255, 0), 2)  
cv.circle(output, bottom_point, 5, (255, 255, 0), -1)  
cv.circle(output, bottom_right, 5, (0, 255, 255), -1)  

cv.imshow("Bounding Box and Orientation", output)
cv.waitKey(0)
cv.destroyAllWindows()

print(f"Angle for binarization {angle_degrees:.2f} degrees")



#########ANSWER#########
#For the canny filter we found 21.21 degree
#We obtain that by applying a canny filter, then normalize x and y
#we weight a score because to find the bottom right corner i needed to or it would have taken 
#the bottom corner again or the most farthest point right
#then we found the point with the highest score combination fro x and y and take it 
#then draw a line between this one and the bottom one and 
#we use the math library to find the angle with the x axis and this new line
#
#For binarization we found 21.01 degreen, we did the same as with canny but with binarization
#I used chatgpt to help me on this one because i was struggling a lot to find the point on the bottom right of the branch
########################