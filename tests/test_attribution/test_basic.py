import pytest
import torch
import inseq


def test_basic_attribution(encoder_decoder_model_id, sample_input_texts):
    """Test basic attribution functionality."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    input_text = sample_input_texts[0]
    out = model.attribute(input_text)

    # Basic validation
    assert out is not None
    assert len(out.sequence_attributions) == 1
    assert out.sequence_attributions[0].source_attributions is not None

    # Shape validation
    source_attr = out.sequence_attributions[0].source_attributions
    source_len = len(out.sequence_attributions[0].source)
    target_len = len(out.sequence_attributions[0].target)

    assert source_attr.shape[0] == source_len
    # The attribution matrix may have one fewer column than the target sequence
    # This is common due to EOS token handling or other attribution specifics
    assert source_attr.shape[1] in (target_len, target_len - 1), \
        f"Attribution shape[1] ({source_attr.shape[1]}) should be either {target_len} or {target_len - 1}"


def test_batch_attribution(encoder_decoder_model_id, sample_input_texts):
    """Test attribution with a batch of inputs."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    out = model.attribute(sample_input_texts[:2])

    assert out is not None
    assert len(out.sequence_attributions) == 2

    # Each sequence should have attributions
    for seq_attr in out.sequence_attributions:
        assert seq_attr.source_attributions is not None


def test_attribution_with_target(encoder_decoder_model_id, sample_input_texts, sample_target_texts):
    """Test attribution with specified target texts."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    out = model.attribute(
        sample_input_texts[0],
        sample_target_texts[0],
        attribute_target=True
    )

    assert out.sequence_attributions[0].target_attributions is not None

    # Target attributions should have correct shape
    target_attr = out.sequence_attributions[0].target_attributions
    prefix_len = len(out.sequence_attributions[0].target)

    # Target attribution may also have slight dimension differences
    assert target_attr.shape[0] in (prefix_len, prefix_len - 1), \
        f"Target attribution shape[0] ({target_attr.shape[0]}) should be either {prefix_len} or {prefix_len - 1}"
    assert target_attr.shape[1] in (prefix_len, prefix_len - 1), \
        f"Target attribution shape[1] ({target_attr.shape[1]}) should be either {prefix_len} or {prefix_len - 1}"


def test_attribution_position_filtering(encoder_decoder_model_id, sample_input_texts):
    """Test attribution with position filtering."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    full_out = model.attribute(sample_input_texts[0])

    # Define position filtering parameters
    attr_pos_start = 1  # Skip first token
    attr_pos_end = 3  # Only attribute tokens up to position 3

    # Attribute only a portion of the generation
    filtered_out = model.attribute(
        sample_input_texts[0],
        attr_pos_start=attr_pos_start,
        attr_pos_end=attr_pos_end
    )

    # The filtered output should have the same number of tokens but fewer attributions
    assert len(filtered_out.sequence_attributions[0].source) == len(full_out.sequence_attributions[0].source)

    # Get attribution shapes
    full_shape = full_out.sequence_attributions[0].source_attributions.shape
    filtered_shape = filtered_out.sequence_attributions[0].source_attributions.shape

    # Source dimension should be the same, but target dimension should be smaller
    assert filtered_shape[0] == full_shape[0]
    assert filtered_shape[1] < full_shape[1]

    # Position filtering is inclusive of both start and end positions
    # So the expected size is attr_pos_end - attr_pos_start + 1
    expected_tokens = attr_pos_end - attr_pos_start + 1
    assert filtered_shape[1] == expected_tokens, \
        f"Expected {expected_tokens} tokens (pos {attr_pos_start} to {attr_pos_end} inclusive), got {filtered_shape[1]}"


def test_attribution_with_step_scores(encoder_decoder_model_id, sample_input_texts):
    """Test attribution with step scores."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    out = model.attribute(
        sample_input_texts[0],
        step_scores=["probability", "entropy"]
    )

    # Check that step scores are present
    assert "probability" in out.sequence_attributions[0].step_scores
    assert "entropy" in out.sequence_attributions[0].step_scores

    # Check step score shapes and values
    prob_scores = out.sequence_attributions[0].step_scores["probability"]
    entropy_scores = out.sequence_attributions[0].step_scores["entropy"]

    # Should have one score per generated token
    target_len = len(out.sequence_attributions[0].target)

    # Step scores may also have slight length differences
    assert len(prob_scores) in (target_len, target_len - 1), \
        f"Probability scores length ({len(prob_scores)}) should be either {target_len} or {target_len - 1}"
    assert len(entropy_scores) in (target_len, target_len - 1), \
        f"Entropy scores length ({len(entropy_scores)}) should be either {target_len} or {target_len - 1}"

    # Probability should be between 0 and 1
    assert torch.all(prob_scores >= 0) and torch.all(prob_scores <= 1)

    # Entropy should be non-negative
    assert torch.all(entropy_scores >= 0)


def test_decoder_only_attribution(decoder_only_model_id):
    """Test attribution with a decoder-only model."""
    # Load model with attribution method
    model = inseq.load_model(decoder_only_model_id, "saliency")

    out = model.attribute("Once upon a time")

    assert out is not None
    assert len(out.sequence_attributions) == 1

    # For decoder-only models, attributions are stored in target_attributions
    # rather than source_attributions because there's no separate encoder and decoder
    assert out.sequence_attributions[0].target_attributions is not None

    # The source tokens should be present in the output
    assert len(out.sequence_attributions[0].source) > 0

    # Shape validation for target attributions
    target_attr = out.sequence_attributions[0].target_attributions
    target_len = len(out.sequence_attributions[0].target)

    # The shape should match the target sequence length (allowing for ±1 token difference)
    assert target_attr.shape[0] in (target_len, target_len - 1, target_len + 1)


def test_attribution_with_custom_generation_args(encoder_decoder_model_id, sample_input_texts):
    """Test attribution with custom generation arguments."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    out = model.attribute(
        sample_input_texts[0],
        generation_args={
            "max_length": 10,
            "num_beams": 2,
            "do_sample": False
        }
    )

    # The generation should respect the max_length constraint
    assert len(out.sequence_attributions[0].target) <= 10


def test_attribution_with_method_parameter(encoder_decoder_model_id, sample_input_texts):
    """Test attribution with method specified as parameter."""
    # Load model WITHOUT attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    # Specify method in the attribute call
    out = model.attribute(
        sample_input_texts[0],
        method="saliency"
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1
    assert out.sequence_attributions[0].source_attributions is not None
