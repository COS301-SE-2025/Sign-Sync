import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import json
import numpy as np

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)
    index_to_label = {int(k): v for k, v in label_map.items()}

# Define the model architecture (must match training)
class KeypointClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=len(index_to_label)):
        super(KeypointClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.output(x)

# Load trained model
model = KeypointClassifier()
model.load_state_dict(torch.load("keypoint_model_upgraded.pth", map_location=torch.device('cpu')))
model.eval()

# Normalize keypoints: center on wrist (0) and scale using distance to middle finger MCP (9)
def normalize_keypoints(landmarks):
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = keypoints[0]
    centered = keypoints - wrist
    scale = np.linalg.norm(centered[9]) if np.linalg.norm(centered[9]) != 0 else 1
    normalized = centered / scale
    return normalized.flatten()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction_text = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                keypoints = normalize_keypoints(hand_landmarks.landmark)
                input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_index = output.argmax(dim=1).item()
                    prediction_text = index_to_label[pred_index]

                cv2.putText(image, f"Prediction: {prediction_text}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live ASL Prediction", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()