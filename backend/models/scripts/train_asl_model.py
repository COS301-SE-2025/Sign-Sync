import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os

# Paths
# DATA_DIR = "../datasets/asl_alphabet_train/asl_alphabet_train"
DATA_DIR = "../datasets/asl_small_test"
MODEL_SAVE_PATH = "../models/asl_model.pth"
LABEL_MAP_PATH = "../models/label_map.json"
BATCH_SIZE = 64
EPOCHS = 10

# Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Dataset and Loader
dataset = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = dataset.classes

# Save label map
with open(LABEL_MAP_PATH, "w") as f:
    json.dump({i: cls for i, cls in enumerate(class_names)}, f)

# Model
class ASLCNN(nn.Module):
    def __init__(self, num_classes=24):
        super(ASLCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)