import cv2
import numpy as np

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

    @staticmethod
    def analyze_camera_feed():

        cap = cv2.VideoCapture(0)  # Open the default camera

        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from the camera.")
                break

            # Apply Canny filter and binarization to detect contours
            canny_contours = MyDetectionMethods.canny_filter(frame)
            binary_contours = MyDetectionMethods.binarization(frame)

            # Draw contours on the original frame
            canny_frame = frame.copy()
            binary_frame = frame.copy()
            cv2.drawContours(canny_frame, canny_contours, -1, (0, 255, 0), 2)  # Green contours
            cv2.drawContours(binary_frame, binary_contours, -1, (255, 0, 0), 2)  # Blue contours

            # Show the frames
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Canny Contours", canny_frame)
            cv2.imshow("Binary Contours", binary_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

