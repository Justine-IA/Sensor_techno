import cv2
import cv2.aruco as aruco
from MyDetectionMethods import MyDetectionMethods

def main():
    
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Define ArUco dictionary and parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    aruco_size = None
    pixel_to_cm_ratio = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        canny_contours = MyDetectionMethods.canny_filter(frame)
        canny_frame = frame.copy()
        cv2.drawContours(frame, canny_contours, -1, (0, 0, 255), 2)


        # Detect ArUco markers in the frame
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # Draw bounding boxes around detected markers
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

        print(corners)
        if ids is not None and len(corners) > 0 and len(corners[0]) > 0:
            top_left_corner = corners[0][0][0]
            top_right_corner = corners[0][0][1]

            print("top left : ", top_left_corner)
            print("top right : ", top_right_corner)

            aruco_size = abs(top_left_corner[0] - top_right_corner[0] )
            print("aruco size", aruco_size)

            pixel_to_cm_ratio = 10/aruco_size

            for contour in canny_contours:
                x, y, w, h = cv2.boundingRect(contour)  # Get bounding box for each contour
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle
                centroid = (x + w // 2, y + h // 2)
                cv2.circle(frame, centroid, radius=2, color=(0, 0, 255), thickness=-1)
                width = round(w*pixel_to_cm_ratio, 1)
                height = round(h*pixel_to_cm_ratio,1)


                text = f"Width:{width}, Height:{height}"
                cv2.putText(frame, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        

        # Show the frame
        cv2.imshow("ArUco Marker Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
