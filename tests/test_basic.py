"""
Basic unit tests for MLOps pipeline.
"""

import os
from pathlib import Path


def test_processed_data_exists():
    """Test if processed dataset folder exists."""
    assert Path("dvc_data/processed/train").exists()


def test_model_file_exists():
    """Test if trained model artifact exists."""
    assert Path("src/models/cnn_model.pt").exists()