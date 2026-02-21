from PIL import Image
from pathlib import Path

def test_image_resize():
    img = Image.new("RGB", (500, 500))
    resized = img.resize((224, 224))
    assert resized.size == (224, 224)