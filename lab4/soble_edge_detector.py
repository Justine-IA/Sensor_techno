import cv2 as cv
import sys
import os
import numpy as np 
image_path = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\car.png"
image_path1 = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\coins.png"
image_path2 = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\dog.png"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img0 = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
img1 = cv.imread(image_path1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)

x =[img0,img1,img2]

for i in x:
    if i is None:
        sys.exit("Error: Could not read the image file.")

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

x[0] = resize_image(x[0],30)

for idx, img in enumerate(x):
    window_name = f"Resized Image {idx + 1}"
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyWindow(window_name)


ksize = (5,5)
kernel = cv.getStructuringElement(cv.MORPH_CROSS, ksize)

img_boundary = []
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
    img_boundary.append(boundary_img)

    cv.destroyAllWindows()

kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

for img in img_boundary:
    grad_x = cv.Sobel(img, -1, 1, 0, ksize)
    grad_y = cv.Sobel(img, -1,0,1, ksize)
    sobel = cv.add(np.abs(grad_x), np.abs(grad_y))
    cv.imshow("Original Image", img)
    cv.waitKey(0)
    cv.imshow("sobel X (Vertical Edges)", grad_x)
    cv.waitKey(0)
    cv.imshow("sobel Y (Horizontal Edges)", grad_y)
    cv.waitKey(0)
    cv.imshow("sobel Edge Detection", sobel)
    cv.waitKey(0)
    cv.destroyAllWindows()

cv.destroyAllWindows()


