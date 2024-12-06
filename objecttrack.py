from ultralytics import YOLO
import cv2
import random
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("custom-tree.pt")

# Initialize Deep SORT tracker with parameters
tracker = DeepSort(
    max_age=10,  # Max frames to keep a track alive without detections
    max_iou_distance=0.7,  # Max IoU distance for association
    max_cosine_distance=0.2,  # Max cosine distance for appearance matching
    nn_budget=None,  # Size of the embedding buffer; None means unlimited
    embedder="mobilenet",  # Model used for generating appearance embeddings
    half=True,  # Use half-precision floating-point calculations for efficiency
    bgr=True,  # Process input frames as BGR color format (as used by OpenCV)
    embedder_gpu=True  # Use GPU for the embedding model for faster processing
)

# Class names and colors
classname = ["Tree"]
#colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# Load video capture
cap = cv2.VideoCapture('media/1448735-uhd_4096_2160_24fps.mp4')

# Set to keep track of unique tree IDs
unique_track_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # Predict the frame using YOLO
    predictions = model.predict(frame)

    # Process detections for Deep SORT
    detections = []
    for prediction in predictions:
        for data in prediction.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = data
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            detections.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])


    # Update tracks using Deep SORT
    tracks = tracker.update_tracks(detections, frame=frame)
    # Draw tracks and detections on the frame
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        unique_track_ids.add(track_id)  # Add to unique track IDs set
        ltrb = track.to_ltrb(orig=True)  # Use original detection coordinates if available
        x1, y1, x2, y2 = map(int, ltrb) #left, top, right, bottom
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the total number of unique trees
    cv2.putText(frame, f"Total Unique Trees: {len(unique_track_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()