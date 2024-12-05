#Read a video and show
from ultralytics import YOLO
import cv2
import os
import random

model = YOLO("best.pt")

#Tracker
tracker = cv2.TrackerMOSSE.create()


classname = ["","Tree"]

cap = cv2.VideoCapture('media/1448735-uhd_4096_2160_24fps.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    #resize the frame
    frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)
    #predict the frame
    prediction = model.predict(frame)

    #Object Detection
    for prediction in prediction:
        detection = []
        for data in prediction.boxes.data.tolist():
            x, y, w, h, score , class_id = data
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            class_id = int(class_id)
            detection.append([x, y, w, h])
            cv2.rectangle(frame, (x,y),(w,h), (0,255,0),2)
            cv2.putText(frame, (f"{score:.2f},{classname[class_id]}"), (int(x+5), int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 225), 2)

    #Object Tracking

    #Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()