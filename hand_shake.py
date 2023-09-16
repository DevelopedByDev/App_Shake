import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

width = int(cap.get(3))
height = int(cap.get(4))

size = (width, height)

patient_id = int(input("Enter patient ID: "))
result = cv2.VideoWriter(f'{patient_id}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

data_rows = []



while True:

    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    start_time = time.time()
    stop_time = 0.0
    time_diff = 0.0

    csv_row_list = [start_time, stop_time, time_diff]

    hand_results = hands.process(imgRGB)
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            # handLms.landmark is the Python list of all hand landmarks observed in one frame
            for lmk in handLms.landmark:
                current_landmark = [lmk.x, lmk.y, lmk.z]
                csv_row_list.append(current_landmark)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    stop_time = time.time()
    csv_row_list[1] = stop_time
    time_diff = stop_time - start_time
    csv_row_list[2] = time_diff

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()