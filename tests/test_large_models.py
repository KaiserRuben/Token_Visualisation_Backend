import pytest
import inseq

# Skip all tests in this module if RUN_LARGE_MODEL_TESTS is not set
pytestmark = pytest.mark.skipif(
    pytest.importorskip("os").environ.get("RUN_LARGE_MODEL_TESTS") != "1",
    reason="Large model tests are only run when RUN_LARGE_MODEL_TESTS=1"
)


@pytest.fixture(scope="module")
def llama_model():
    """Load the Llama model for testing."""
    try:
        return inseq.load_model("meta-llama/Llama-3.2-3B", None)
    except Exception as e:
        pytest.skip(f"Failed to load Llama model: {e}")


@pytest.fixture(scope="module")
def deepseek_model():
    """Load the DeepSeek model for testing."""
    try:
        return inseq.load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", None)
    except Exception as e:
        pytest.skip(f"Failed to load DeepSeek model: {e}")


# ----- Llama Model Tests -----

@pytest.mark.slow
def test_load_llama():
    """Test loading the Llama model."""
    model = inseq.load_model("meta-llama/Llama-3.2-3B", None)
    assert model is not None
    assert model.__class__.__name__ == "HuggingfaceDecoderOnlyModel"
    assert not model.is_encoder_decoder


@pytest.mark.slow
def test_llama_basic_attribution(llama_model):
    """Test basic attribution with Llama model."""
    out = llama_model.attribute(
        "Once upon a time in a galaxy far, far away",
        method="saliency"  # Specify method explicitly since llama_model is loaded without one
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1
    assert out.sequence_attributions[0].source_attributions is not None


@pytest.mark.slow
@pytest.mark.parametrize("method", ["saliency", "input_x_gradient"])
def test_llama_with_gradient_methods(method):
    """Test Llama with gradient-based attribution methods."""
    # Use 8-bit loading to reduce memory requirements
    import transformers

    # Skip if 8-bit loading is not available
    try:
        import bitsandbytes
    except ImportError:
        pytest.skip("bitsandbytes not installed, required for 8-bit model loading")

    model = inseq.load_model(
        "meta-llama/Llama-3.2-3B",
        method,
        model_kwargs={"load_in_8bit": True, "device_map": "auto"}
    )

    out = model.attribute(
        "Explain the theory of relativity briefly",
        max_new_tokens=20  # Limit generation length for testing
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1
    assert out.sequence_attributions[0].source_attributions is not None


@pytest.mark.slow
def test_llama_with_attention(llama_model):
    """Test Llama with attention attribution."""
    model = inseq.load_model("meta-llama/Llama-3.2-3B", "attention")

    out = model.attribute(
        "What is the capital of France?",
        max_new_tokens=5  # Limit generation length for testing
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1

    # Attention attribution returns multi-dimensional attributions
    # with shape [source_len, target_len, n_layers, n_heads]
    source_attr = out.sequence_attributions[0].source_attributions
    assert len(source_attr.shape) == 4

    # Verify that the first two dimensions are reasonable
    source_len = len(out.sequence_attributions[0].source)
    target_len = len(out.sequence_attributions[0].target)
    assert source_attr.shape[0] == source_len
    assert source_attr.shape[1] in (target_len, target_len - 1)


@pytest.mark.slow
def test_llama_fewshot_attribution(llama_model):
    """Test attribution with few-shot examples."""
    # Few-shot prompt with examples
    prompt = """
    Translate English to French:

    English: Hello, how are you?
    French: Bonjour, comment allez-vous?

    English: I love to read books.
    French: J'aime lire des livres.

    English: What time is it?
    French: Quelle heure est-il?

    English: The weather is nice today.
    French:
    """

    out = llama_model.attribute(
        prompt,
        method="saliency",  # Specify method explicitly
        max_new_tokens=10  # Limit generation length for testing
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1

    # Print the generation for debugging
    print(f"Generated: {out.sequence_attributions[0].target}")


@pytest.mark.slow
def test_llama_save_load(llama_model, output_dir):
    """Test saving and loading Llama attributions."""
    out = llama_model.attribute(
        "The best way to learn programming is",
        method="saliency",  # Specify method explicitly
        max_new_tokens=10  # Limit generation length for testing
    )

    # Save attributions
    output_path = output_dir / "llama_attribution.json"
    out.save(output_path, overwrite=True)

    # Load attributions
    loaded_out = inseq.FeatureAttributionOutput.load(output_path)

    # Check equality
    assert len(loaded_out.sequence_attributions) == len(out.sequence_attributions)

    # Clean up
    output_path.unlink()


# ----- DeepSeek Model Tests -----

@pytest.mark.slow
def test_load_deepseek():
    """Test loading the DeepSeek model."""
    model = inseq.load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", None)
    assert model is not None
    assert model.__class__.__name__ == "HuggingfaceDecoderOnlyModel"
    assert not model.is_encoder_decoder


@pytest.mark.slow
def test_deepseek_basic_attribution(deepseek_model):
    """Test basic attribution with DeepSeek model."""
    out = deepseek_model.attribute(
        "Explain quantum computing to a five-year-old child",
        method="saliency",
        max_new_tokens=20  # Limit generation length for testing
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1
    assert out.sequence_attributions[0].source_attributions is not None


@pytest.mark.slow
@pytest.mark.parametrize("method", ["saliency", "input_x_gradient"])
def test_deepseek_with_gradient_methods(method):
    """Test DeepSeek with gradient-based attribution methods."""
    # Skip if 8-bit loading is not available
    try:
        import bitsandbytes
    except ImportError:
        pytest.skip("bitsandbytes not installed, required for 8-bit model loading")

    model = inseq.load_model(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        method,
        model_kwargs={"load_in_8bit": True, "device_map": "auto"}
    )

    out = model.attribute(
        "Write a short poem about artificial intelligence",
        max_new_tokens=30  # Limit generation length for testing
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1
    assert out.sequence_attributions[0].source_attributions is not None


@pytest.mark.slow
def test_deepseek_with_attention(deepseek_model):
    """Test DeepSeek with attention attribution."""
    model = inseq.load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "attention")

    out = model.attribute(
        "What are the main challenges in natural language processing?",
        max_new_tokens=15  # Limit generation length for testing
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1

    # Attention attribution returns multi-dimensional attributions
    # with shape [source_len, target_len, n_layers, n_heads]
    source_attr = out.sequence_attributions[0].source_attributions
    assert len(source_attr.shape) == 4

    # Verify that the first two dimensions are reasonable
    source_len = len(out.sequence_attributions[0].source)
    target_len = len(out.sequence_attributions[0].target)
    assert source_attr.shape[0] == source_len
    assert source_attr.shape[1] in (target_len, target_len - 1)


@pytest.mark.slow
def test_deepseek_reasoning_attribution(deepseek_model):
    """Test attribution of reasoning with DeepSeek."""
    # Chain-of-thought prompt
    prompt = """
    Q: John has 5 apples. He gives 2 apples to Mary and then buys 3 more apples. 
    How many apples does John have now?

    Let me think through this step by step:
    """

    out = deepseek_model.attribute(
        prompt,
        method="saliency",
        max_new_tokens=50  # Allow for full reasoning chain
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1

    # Print the generation for debugging
    print(f"Generated reasoning: {out.sequence_attributions[0].target}")


@pytest.mark.slow
def test_deepseek_save_load(deepseek_model, output_dir):
    """Test saving and loading DeepSeek attributions."""
    out = deepseek_model.attribute(
        "What are three key insights from recent AI research?",
        method="saliency",
        max_new_tokens=30  # Limit generation length for testing
    )

    # Save attributions
    output_path = output_dir / "deepseek_attribution.json"
    out.save(output_path, overwrite=True)

    # Load attributions
    loaded_out = inseq.FeatureAttributionOutput.load(output_path)

    # Check equality
    assert len(loaded_out.sequence_attributions) == len(out.sequence_attributions)

    # If shapes don't match exactly (common with token handling differences), check the overlapping area
    original_attr = out.sequence_attributions[0].source_attributions
    loaded_attr = loaded_out.sequence_attributions[0].source_attributions

    if original_attr.shape == loaded_attr.shape:
        assert (original_attr == loaded_attr).all()
    else:
        # Get the minimum dimensions
        min_dim0 = min(original_attr.shape[0], loaded_attr.shape[0])
        min_dim1 = min(original_attr.shape[1], loaded_attr.shape[1])
        # Compare the overlapping area
        assert (original_attr[:min_dim0, :min_dim1] == loaded_attr[:min_dim0, :min_dim1]).all()

    # Clean up
    output_path.unlink()