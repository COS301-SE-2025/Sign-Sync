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