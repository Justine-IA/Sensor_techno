import cv2 as cv
import sys
import os
import numpy as np 
image_path = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\icons01.png"
image_path1 = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\piece03.png"
image_path2 = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\meltPool.tif"
image_path3 = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\car.png"
image_path4 = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\coins.png"
image_path5 = r"H:\Image_analy\Sensor_techno\lab4\lab4_pics\dog.png"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img0 = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
img1 = cv.imread(image_path1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)
img3 = cv.imread(image_path3, cv.IMREAD_GRAYSCALE)
img4 = cv.imread(image_path4, cv.IMREAD_GRAYSCALE)
img5 = cv.imread(image_path5, cv.IMREAD_GRAYSCALE)

x =[img0,img1,img2,img3,img4,img5]

for i in x:
    if i is None:
        sys.exit("Error: Could not read the image file.")

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

x[0] = resize_image(x[0],30)
x[1] = resize_image(x[1],30)
x[2] = resize_image(x[2],50)
x[3] = resize_image(x[3],30)

ksize = (5,5)
kernel = cv.getStructuringElement(cv.MORPH_CROSS, ksize)


for img in x:
    Canny = cv.Canny(img, 70, 150)
    
    lines = cv.HoughLines(Canny, 1, np.pi / 180, 100)  
    
    img_with_lines = img.copy()
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Convert from polar coordinates to Cartesian coordinates
            x1 = int(rho * np.cos(theta) + 1000 * (-np.sin(theta)))  # Adding large value to ensure line visibility
            y1 = int(rho * np.sin(theta) + 1000 * (np.cos(theta)))
            x2 = int(rho * np.cos(theta) - 1000 * (-np.sin(theta)))
            y2 = int(rho * np.sin(theta) - 1000 * (np.cos(theta)))
            
            # Draw the line on the image
            cv.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Red line with thickness of 2
    
    cv.imshow("Original Image", img)
    cv.waitKey(0)
    
    cv.imshow("Hough Lines", img_with_lines)
    cv.waitKey(0)
    cv.destroyAllWindows()

cv.destroyAllWindows()
