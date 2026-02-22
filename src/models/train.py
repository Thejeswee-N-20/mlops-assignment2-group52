import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Data Paths
# -----------------------------
TRAIN_DIR = "dvc_data/processed/train"
VAL_DIR = "dvc_data/processed/val"

# -----------------------------
# Data Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Model Definition
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# -----------------------------
# Training Setup
# -----------------------------
model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []

# -----------------------------
# MLflow Tracking
# -----------------------------
mlflow.set_experiment("cats_dogs_classification")

with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LR)

    # -------------------------
    # Training Loop
    # -------------------------
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

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

        # Log metric
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)

    # -------------------------
    # Save Model Artifact
    # -------------------------
    os.makedirs("src/models", exist_ok=True)
    model_path = "src/models/cnn_model.pt"

    torch.save(model.state_dict(), model_path)

    # Log model to MLflow
    mlflow.pytorch.log_model(model, "model")

    # -------------------------
    # Log Loss Curve Artifact
    # -------------------------
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    loss_curve_path = "loss_curve.png"
    plt.savefig(loss_curve_path)

    mlflow.log_artifact(loss_curve_path)

print("Training complete and logged to MLflow.")