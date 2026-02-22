"""
Post-deployment model performance evaluation script.
Simulates real prediction requests and compares with true labels.
"""

import requests

test_samples = [
    ("tests/sample_cat.jpg", "Cat"),
    ("tests/sample_dog.jpg", "Dog")
]

correct = 0

for image_path, true_label in test_samples:
    with open(image_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": f}
        )

    predicted = response.json()["prediction"]
    print(f"Predicted: {predicted}, True: {true_label}")

    if predicted == true_label:
        correct += 1

accuracy = correct / len(test_samples)
print("Post-deployment Accuracy:", accuracy)