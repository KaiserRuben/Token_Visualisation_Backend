import pytest
import inseq
from inseq.models import HuggingfaceEncoderDecoderModel, HuggingfaceDecoderOnlyModel


def test_load_encoder_decoder_model():
    """Test loading an encoder-decoder model."""
    model = inseq.load_model("Helsinki-NLP/opus-mt-en-fr", None)
    assert isinstance(model, HuggingfaceEncoderDecoderModel)
    assert model.is_encoder_decoder


def test_load_decoder_only_model():
    """Test loading a decoder-only model."""
    model = inseq.load_model("gpt2", None)
    assert isinstance(model, HuggingfaceDecoderOnlyModel)
    assert not model.is_encoder_decoder


def test_load_with_attribution_method():
    """Test loading a model with an attribution method."""
    model = inseq.load_model("Helsinki-NLP/opus-mt-en-fr", "integrated_gradients")
    assert model.attribution_method is not None
    assert model.attribution_method.__class__.__name__ == "IntegratedGradientsAttribution"


@pytest.mark.parametrize("method", [
    "saliency",
    "integrated_gradients",
    "input_x_gradient",
    "occlusion",
    "attention"
])
def test_load_with_different_attribution_methods(method):
    """Test loading a model with different attribution methods."""
    model = inseq.load_model("Helsinki-NLP/opus-mt-en-fr", method)
    assert model.attribution_method is not None
    assert model.attribution_method.__class__.__name__.lower().startswith(method.replace("_", ""))


def test_load_with_tokenizer_kwargs():
    """Test loading a multilingual model with tokenizer kwargs."""
    model = inseq.load_model(
        "facebook/m2m100_418M",
        "input_x_gradient",
        tokenizer_kwargs={"src_lang": "en", "tgt_lang": "fr"}
    )
    assert model is not None
    assert model.tokenizer is not None


def test_load_with_device(device):
    """Test loading a model on a specific device."""
    model = inseq.load_model("sshleifer/tiny-gpt2", None, device=device)
    assert model.device == device

    # Test that the model is actually on the specified device
    if device == "cuda":
        assert next(model.model.parameters()).is_cuda
    else:
        assert not next(model.model.parameters()).is_cuda


def test_list_feature_attribution_methods():
    """Test listing available attribution methods."""
    methods = inseq.list_feature_attribution_methods()
    assert isinstance(methods, list)
    assert len(methods) > 0
    # Check for some common methods
    assert "saliency" in methods
    assert "integrated_gradients" in methods
    assert "attention" in methods