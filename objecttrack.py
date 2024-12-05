from ultralytics import YOLO
import cv2
import os
import random

# Initialize YOLO Model
model = YOLO("best.pt")
classname = ["", "Tree"]  # Note: Class index starts at 1 if your list begins with an empty string

# Initialize Video Capture
cap = cv2.VideoCapture('media/1448735-uhd_4096_2160_24fps.mp4')

# Tracker Setup
tracker_name = 'MedianFlow'
OPENCV_TRACKER_TYPES = {
    'Boosting': cv2.TrackerBoosting.create(),
    'MIL': cv2.TrackerMIL(),
    'KCF': cv2.TrackerKCF(),
    'TLD': cv2.TrackerTLD(),
    'MedianFlow': cv2.TrackerMedianFlow(),
    'GOTURN': cv2.TrackerGOTURN
}
tracker_type = OPENCV_TRACKER_TYPES[tracker_name]
trackers = []  # To hold all active trackers
detect_interval = 5  # Detect new objects every 5 frames

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # **Detection Interval**
    if frame_index % detect_interval == 0:
        # Predict the frame
        predictions = model.predict(frame)

        # **Reset Trackers for New Detections**
        trackers = []

        for prediction in predictions:
            detection = []
            for data in prediction.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = data
                x1, y2, x2, y2, score, class_id = int(x1), int(y1), int(x2), int(y2), score, int(class_id)

                # Initialize Tracker for Each Detection
                ok, bbox = True, (x1, y1, x2 - x1, y2 - y1)  # x, y, w, h
                tracker = tracker_type().init(frame, bbox)
                trackers.append(tracker)

                # Draw Detection Rectangle (Before Tracking)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, (f"{score:.2f},{classname[class_id]}"), (int(x1 + 5), int(y1) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 225), 2)

    else:
        # **Update Trackers**
        for i, tracker in enumerate(trackers):
            ok, bbox = tracker.update(frame)
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                # Optional: Add text indicating the tracker ID or class
                # cv2.putText(frame, f"Tracker {i}", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            else:
                # Remove failed trackers
                trackers.pop(i)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()