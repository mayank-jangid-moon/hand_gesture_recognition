import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

data = []

frame_counter = 0

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
    max_num_hands=1) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    frame_counter += 1
    # image = cv2.imread("sample_rr.png")
    
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        
        
        # print(f"\n\n\n\nMultihandedness: {results.multi_handedness[idx]}")
        # print(f"\n\n\n\n Hand number {idx} : \n\n\n\n")

        frame_data = {"Frame": frame_counter}

        # break
        # Here is How to Get All the Coordinates

        for ids, landmark in enumerate(hand_landmarks.landmark):
          # print(ids, landmark)

          frame_data[f"Landmark_{ids}_X"] = landmark.x
          frame_data[f"Landmark_{ids}_Y"] = landmark.y
          frame_data[f"Landmark_{ids}_Z"] = landmark.z
          
        data.append(frame_data)
          
          # print (ids, landmark.x, landmark.y, landmark.z)
        mp_drawing.draw_landmarks(image, 
                                  hand_landmarks, 
                                  mp_hands.HAND_CONNECTIONS, 
                                  mp_drawing_styles.get_default_hand_landmarks_style(), 
                                  mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

df = pd.DataFrame(data)

df.to_csv("thumbs_up.csv", index=True)
print("CSV created sucessfully!!!")
cap.release()