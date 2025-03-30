import pytest
import torch
import inseq
import os
from pathlib import Path

# Model identifiers - using these instead of pre-loaded models
@pytest.fixture(scope="session")
def encoder_decoder_model_id():
    """Return a model identifier for a small encoder-decoder model."""
    return "Helsinki-NLP/opus-mt-en-fr"

@pytest.fixture(scope="session")
def decoder_only_model_id():
    """Return a model identifier for a small decoder-only model."""
    return "gpt2"

@pytest.fixture(scope="session")
def tiny_decoder_model_id():
    """Return a model identifier for a tiny decoder-only model."""
    return "sshleifer/tiny-gpt2"

# Text fixtures
@pytest.fixture
def sample_input_texts():
    """Return a list of sample input texts for testing."""
    return [
        "The developer argued with the designer because she did not like the design.",
        "A short example.",
        "This is a test with multiple tokens that should be analyzed."
    ]

@pytest.fixture
def sample_target_texts():
    """Return a list of sample target texts for testing."""
    return [
        "Le développeur s'est disputé avec le designer car elle n'aimait pas le design.",
        "Un exemple court.",
        "Ceci est un test avec plusieurs jetons qui devraient être analysés."
    ]

# Attribution methods fixtures
@pytest.fixture
def gradient_attribution_methods():
    """Return a list of gradient-based attribution methods."""
    return ["saliency", "integrated_gradients", "input_x_gradient", "deep_lift"]

@pytest.fixture
def perturbation_attribution_methods():
    """Return a list of perturbation-based attribution methods."""
    return ["occlusion", "lime"]

@pytest.fixture
def internals_attribution_methods():
    """Return a list of internals-based attribution methods."""
    return ["attention"]

@pytest.fixture
def output_dir():
    """Create and return a temporary output directory for test artifacts."""
    path = Path("test_outputs")
    path.mkdir(exist_ok=True)
    return path

@pytest.fixture
def run_large_model_tests():
    """Determine if large model tests should be run."""
    return os.environ.get("RUN_LARGE_MODEL_TESTS") == "1"

@pytest.fixture
def device():
    """Return the available device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"