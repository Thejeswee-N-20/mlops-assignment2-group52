from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
import time

app = FastAPI()

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    filename="api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Monitoring metric
request_count = 0

# ---------------------------
# Model Definition
# ---------------------------
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


DEVICE = "cpu"
MODEL_PATH = "src/models/cnn_model.pt"

model = None

def load_model():
    global model
    if model is None:
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()


# ---------------------------
# Image Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ---------------------------
# Health Endpoint
# ---------------------------
@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {
        "status": "API is running",
        "requests_served": request_count
    }


# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count

    request_count += 1
    start_time = time.time()

    logger.info("Prediction request received")

    load_model()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)

    predicted_class = torch.argmax(probs).item()
    label = "Cat" if predicted_class == 0 else "Dog"

    latency = time.time() - start_time

    logger.info(f"Prediction completed in {latency:.3f} seconds")
    logger.info(f"Total requests served: {request_count}")

    return {
        "prediction": label,
        "confidence": float(probs[0][predicted_class])
    }