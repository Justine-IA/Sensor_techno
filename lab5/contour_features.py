import cv2 as cv
import sys
import os
import numpy as np 
image_path = r"H:\Image_analy\Sensor_techno\lab5\pics_Lab5\testImg.jpg"
image_path1 = r"H:\Image_analy\Sensor_techno\lab5\pics_Lab5\classA\a2.tiff"
image_path2 = r"H:\Image_analy\Sensor_techno\lab5\pics_Lab5\classA\a2.tiff"

if not os.path.exists(image_path):
    
    sys.exit(f"Error: File not found at {image_path}")

img0 = cv.imread(image_path)
img1 = cv.imread(image_path1)
img2 = cv.imread(image_path2)

class_A =[img0,img1,img2]

for i in class_A:
    if i is None:
        sys.exit("Error: Could not read the image file.")

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)


ksize = (9,9)
kernel =cv.getStructuringElement(cv.MORPH_CROSS,ksize)

def canny_filter(image, lower_threshold=50, upper_threshold=150):
    # Apply Gaussian Blur to reduce noise
    blurred = cv.GaussianBlur(image, (5, 5), 1.4)

    # Apply Canny edge detection
    edges = cv.Canny(blurred, lower_threshold, upper_threshold)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return edges, contours

img_tresh = []

for i in class_A:
    edges, contours = canny_filter(i,50,250)

    output = i.copy()

  
    for cnt in contours:
        # Compute Moments
        M = cv.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  # Centroid X
            cy = int(M['m01'] / M['m00'])  # Centroid Y
        else:
            cx, cy = 0, 0

        # Compute various features
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        hull = cv.convexHull(cnt)
        is_convex = cv.isContourConvex(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int32(box)
        aspect_ratio = float(w) / h
        equi_diameter = np.sqrt(4 * area / np.pi)
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        # Draw contours and bounding boxes
        cv.drawContours(output, [cnt], -1, (0, 255, 0), 2)  # Contours in green
        cv.drawContours(output, [box], 0, (0,0,255), 2 )
        if len(cnt)>5:
            ellipse = cv.fitEllipse(cnt)
            cv.ellipse(output, ellipse, (255,0,0),2 )
        # Show the image with contours and features



        # Mark the centroid
        cv.circle(output, (cx, cy), 5, (255, 0, 255), -1)


        # Print contour features
        print(f"Contour Features:")
        print(f" - convex: {is_convex}")
        print(f" - Area: {area}")
        print(f" - Perimeter: {perimeter}")
        print(f" - Aspect Ratio: {aspect_ratio}")
        print(f" - Equivalent Diameter: {equi_diameter}")
        print(f" - Centroid: ({cx}, {cy})")
        print(f" - Bounding Box: x={x}, y={y}, w={w}, h={h}")
        print(f" - Leftmost Point: {leftmost}")
        print(f" - Rightmost Point: {rightmost}")
        print(f" - Topmost Point: {topmost}")
        print(f" - Bottommost Point: {bottommost}")

    # Show the image with contours and features
    cv.imshow("Contours and Features", output)
    cv.waitKey(0)

    # Optional: Display the Canny edges
    cv.imshow("Canny Edges", edges)
    cv.waitKey(0)











