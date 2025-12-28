"""
Smart Face Recognition Attendance System

Author: Rohini Kasarapu
Description: Face recognition based attendance system using Python and OpenCV.
"""

import cv2
import numpy as np
import os
import csv
from datetime import datetime

DATASET_PATH = "dataset"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("❌ Haar cascade not loaded")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
label_id = 0


# LOAD DATASET (MULTI IMAGE PER PERSON)

for folder in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = folder

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer.train(faces, labels)
print("✅ Training completed")


# ATTENDANCE (SAME FILE)

already_printed = set()

def mark_attendance(label_name):
    roll, name = label_name.split("_")
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{roll}_{today}"

    file_path = os.path.join(os.getcwd(), "attendance.csv")

    # Already printed → do nothing
    if key in already_printed:
        return

    already_marked = False

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3 and row[0] == roll and row[2] == today:
                    already_marked = True
                    break

    if not already_marked:
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(file_path).st_size == 0:
                writer.writerow(["Roll", "Name", "Date", "Time"])

            now = datetime.now()
            writer.writerow([
                roll,
                name,
                today,
                now.strftime("%H:%M:%S")
            ])

        print(f"✅ Attendance marked: {roll} - {name} ({today})")
    else:
        print(f"ℹ️ {roll} already marked today")

    already_printed.add(key)



# WEBCAM

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("📸 Camera started | Press ENTER to exit")



while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x, y, w, h) in detected_faces:
        roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi)

        name = "Unknown"
        if confidence < 85:   # 👈 tuned value
            name = label_map[label]
            mark_attendance(name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({int(confidence)})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()


