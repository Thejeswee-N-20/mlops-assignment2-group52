"""
FastAPI inference service for Cats vs Dogs classifier.

Provides:
1. Health check endpoint
2. Prediction endpoint

This service loads the trained CNN model
and serves predictions via REST API.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import io

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "src/models/cnn_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Cat", "Dog"]

# ---------------------------
# Model Definition (same as training)
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


# ---------------------------
# Load Model
# ---------------------------
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    return {
        "prediction": CLASS_NAMES[probs.argmax().item()],
        "confidence": float(probs.max().item())
    }