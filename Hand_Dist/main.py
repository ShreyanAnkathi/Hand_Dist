import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Polynomial coefficients for distance conversion
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        continue

    hands, img = detector.findHands(img, draw=False)  # Ensure 'img' is modified

    if hands:
        hand = hands[0]  # First detected hand
        lmList = hand['lmList']  # List of landmarks
        x, y, w, h = hand['bbox']  # Bounding box
        x1, y1 = lmList[5][:2]  # Landmark 5
        x2, y2 = lmList[17][:2]  # Landmark 17

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
