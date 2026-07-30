"""
Object Detection Using TensorFlow/OpenCV
A Flask web app that lets a user upload an image (or use a live webcam feed
via the standalone script) and runs real-time object detection using a
pre-trained MobileNet-SSD deep learning model through OpenCV's DNN module.

Author: Surjya Kanta Saha
"""

from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pre-trained MobileNet-SSD model files (see README for download instructions)
PROTOTXT = "models/MobileNetSSD_deploy.prototxt"
MODEL = "models/MobileNetSSD_deploy.caffemodel"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

CONFIDENCE_THRESHOLD = 0.4

net = None
if os.path.exists(PROTOTXT) and os.path.exists(MODEL):
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)


def detect_objects(image_path, output_path):
    """Run object detection on an image and save the annotated result."""
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    found_objects = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
            found_objects.append(CLASSES[idx])

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 200, 0), 2)
            cv2.putText(image, label, (startX, max(startY - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    cv2.imwrite(output_path, image)
    return found_objects


@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detected = []
    error = None

    if request.method == "POST":
        if net is None:
            error = "Model files not found. Please download them first (see README)."
        else:
            file = request.files.get("image")
            if file:
                input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
                output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
                file.save(input_path)
                detected = detect_objects(input_path, output_path)
                result_image = output_path

    return render_template("index.html", result_image=result_image, detected=detected, error=error)


if __name__ == "__main__":
    app.run(debug=True)
