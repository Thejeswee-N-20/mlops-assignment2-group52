import torch
from src.api.app import SimpleCNN

def test_prediction_probabilities():
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 224, 224)

    outputs = model(dummy_input)
    probs = torch.softmax(outputs, dim=1)

    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)