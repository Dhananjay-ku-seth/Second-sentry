import cv2
import numpy as np

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_model.xml")

# Load label names
labels = np.load("models/labels.npy", allow_pickle=True).item()

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def recognize_faces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    results = []

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 80:
            name = labels[label]
        else:
            name = "UNKNOWN"

        results.append((x,y,w,h,name))

    return results
