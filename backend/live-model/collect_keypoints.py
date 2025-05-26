import cv2
import mediapipe as mp
import json
import os

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

label = input("Enter the label for this recording session (e.g., A, B, C): ").upper()
save_path = f"keypoints_{label}.json"

if os.path.exists(save_path):
    with open(save_path, "r") as f:
        data = json.load(f)
else:
    data = []

cap = cv2.VideoCapture(0)
print("Press 's' to save frame, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

    cv2.imshow("Keypoint Collector", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and results.multi_hand_landmarks:
        print(f"Saved frame for label {label}")
        data.append({
            "label": label,
            "keypoints": [coord for point in keypoints for coord in point]
        })

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

with open(save_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(data)} samples to {save_path}")