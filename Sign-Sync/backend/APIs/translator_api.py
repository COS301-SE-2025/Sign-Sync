from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import json
import numpy as np

app = FastAPI()

#testing merge

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)
    index_to_label = {int(k): v for k, v in label_map.items()}

# Define model
class KeypointClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(63, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(128, len(index_to_label))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.output(x)
    
model = KeypointClassifier()
model.load_state_dict(torch.load("../live-model/keypoint_model_upgraded.pth", map_location=torch.device("cpu")))
model.eval()

class Landmark(BaseModel):
    x: float
    y: float
    z: float

class KeypointsRequest(BaseModel):
    keypoints: list[Landmark]  # 21 landmarks, each with x/y/z

def normalize_keypoints(landmarks):
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = keypoints[0]
    centered = keypoints - wrist
    scale = np.linalg.norm(centered[9]) if np.linalg.norm(centered[9]) != 0 else 1
    normalized = centered / scale
    return normalized.flatten()

@app.post("/predict")
def predict(request: KeypointsRequest):
    norm = normalize_keypoints(request.keypoints)
    tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()
        label = index_to_label[pred]
    return {"prediction": label}