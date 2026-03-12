import cv2
import os
import numpy as np

dataset_path = "models/dataset"

faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1


recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(labels))

recognizer.save("models/face_model.xml")

np.save("models/labels.npy", label_map)

print("Training complete")
