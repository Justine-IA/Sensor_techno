import cv2
import numpy as np


#creating class with the two filters in it to then analyze and filter images 
#to isolate object with the main file 
class MyDetectionMethods:
    @staticmethod
    def canny_filter(image_data, lower_threshold=50, upper_threshold=150):
        # Convert to grayscale
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    @staticmethod
    def binarization(image_data, threshold_value=127):
        # Convert to grayscale
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Apply binary thresholding
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours


