import cv2
import numpy as np
import mediapipe as mp
import requests
from collections import deque

SEQ_LEN = 50
FEATURES = 126
API_URL = "http://localhost:8000/predict/"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def normalize_keypoints(keypoints):
    if keypoints.size == 0:
        return keypoints
    min_vals = np.min(keypoints, axis=0)
    max_vals = np.max(keypoints, axis=0)
    range_vals = max_vals - min_vals + 1e-5
    return ((keypoints - min_vals) / range_vals)

def extract_frame_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            if label == 'Left':
                left_hand = arr
            else:
                right_hand = arr

    combined = np.vstack((left_hand, right_hand))  
    return normalize_keypoints(combined).flatten()  

buffer = deque(maxlen=SEQ_LEN)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = extract_frame_keypoints(frame)

    if np.count_nonzero(keypoints) > 0:
        buffer.append(keypoints)
    
    if len(buffer) == SEQ_LEN:
        
        payload = {
            "sequence": [list(f) for f in buffer]
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                gloss = result['gloss']
                confidence = result['confidence']
                print(f"Prediction: {gloss} ({confidence:.2f})")
                cv2.putText(frame, f"{gloss} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("API Error:", response.text)
        except Exception as e:
            print("Request failed:", e)

        buffer.clear()  

    cv2.imshow("Live Sign Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
