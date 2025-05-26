import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load all normalized keypoint data
all_data = []
all_labels = []

for file in os.listdir():
    if file.startswith("normalized_keypoints_") and file.endswith(".json"):
        with open(file, "r") as f:
            samples = json.load(f)
            for sample in samples:
                all_data.append(sample["keypoints"])
                all_labels.append(sample["label"])

# Encode labels to integers
le = LabelEncoder()
encoded_labels = le.fit_transform(all_labels)

# Convert to torch tensors
X = torch.tensor(all_data, dtype=torch.float32)
y = torch.tensor(encoded_labels, dtype=torch.long)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define an upgraded MLP with 2 hidden layers and dropout
class KeypointClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
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

model = KeypointClassifier(input_size=63, hidden_size=128, num_classes=len(le.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        accuracy = (test_outputs.argmax(1) == y_test).float().mean()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Test Acc: {accuracy:.4f}")

# Save model and label encoder
torch.save(model.state_dict(), "keypoint_model_upgraded.pth")
with open("label_map.json", "w") as f:
    json.dump({i: label for i, label in enumerate(le.classes_)}, f)

print("Upgraded model and label map saved.")