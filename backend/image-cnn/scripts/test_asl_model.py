import torch
from PIL import Image
import torchvision.transforms as transforms
import json
import sys

# Based off: https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data?select=asl_alphabet_train

MODEL_PATH = "../models/asl_model.pth"
LABEL_MAP_PATH = "../models/label_map.json"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

# Load label map
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

# Model
class ASLCNN(torch.nn.Module):
    def __init__(self, num_classes=24):
        super(ASLCNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 16 * 16, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLCNN(num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    print(f"Predicted Letter: {label_map[predicted.item()]}")