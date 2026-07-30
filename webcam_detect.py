"""
Real-Time Object Detection via Webcam
Standalone script (no Flask needed) that opens your webcam and draws
live bounding boxes around detected objects using OpenCV's DNN module
with a pre-trained MobileNet-SSD model.

Run this with:  python webcam_detect.py
Press 'q' to quit.

Author: Surjya Kanta Saha
"""

import cv2
import numpy as np

PROTOTXT = "models/MobileNetSSD_deploy.prototxt"
MODEL = "models/MobileNetSSD_deploy.caffemodel"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

CONFIDENCE_THRESHOLD = 0.4

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
cap = cv2.VideoCapture(0)

print("Starting webcam... press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 200, 0), 2)
            cv2.putText(frame, label, (startX, max(startY - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    cv2.imshow("Real-Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
