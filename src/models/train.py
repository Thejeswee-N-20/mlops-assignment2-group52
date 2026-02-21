"""
Baseline CNN training script for Cats vs Dogs classification.

This script:
1. Loads preprocessed dataset
2. Defines a simple CNN model
3. Trains the model
4. Saves trained model artifact

This forms the baseline model in our MLOps pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
import mlflow.pytorch

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "dvc_data/processed"
MODEL_PATH = "src/models/cnn_model.pt"

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Data Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ---------------------------
# CNN Model
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
mlflow.start_run()

mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("epochs", EPOCHS)
mlflow.log_param("learning_rate", LR)
# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {running_loss/len(train_loader):.4f}")
    mlflow.log_metric("train_loss", running_loss/len(train_loader), step=epoch)

# ---------------------------
# Save Model
# ---------------------------
Path("src/models").mkdir(exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)
print("Model saved successfully!")
mlflow.pytorch.log_model(model, "model")
mlflow.end_run()