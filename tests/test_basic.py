"""
Basic sanity tests for CI environment validation.
These tests ensure the runtime environment is working correctly.
"""

import sys


def test_python_version():
    """Ensure Python version is compatible."""
    assert sys.version_info.major >= 3


def test_environment_imports():
    """Ensure core libraries can be imported."""
    import torch
    import fastapi
    import mlflow

    assert True