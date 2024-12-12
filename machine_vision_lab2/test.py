            # for contour in canny_contours:

            #     x, y, w, h = cv2.boundingRect(contour)  # Get bounding box for each contour
            #     width = round(w*pixel_to_cm_ratio, 1)
            #     height = round(h*pixel_to_cm_ratio,1)
            #     centroid = (x + w // 2, y + h // 2)
            #     if width>1 and height>4:
            #         cv2.circle(frame, centroid, radius=2, color=(0, 0, 255), thickness=-1)
            #         cv2.rectangle(frame,(x,y),(x+w, y+h), (255, 0, 0), 2)  # Draw rectangle
            #         text = f"Width:{width:.1f}, Height:{height:.1f}"
            #         cv2.putText(frame, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            