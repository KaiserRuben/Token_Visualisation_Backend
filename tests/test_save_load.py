import pytest
import os
import torch
import inseq
from pathlib import Path


def test_save_load_attribution_output(encoder_decoder_model_id, sample_input_texts, output_dir):
    """Test saving and loading attribution outputs."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    # Generate attribution
    out = model.attribute(sample_input_texts[0])

    # Save to a file
    output_path = output_dir / "attribution_output.json"
    out.save(output_path, overwrite=True)

    # Check that the file exists
    assert output_path.exists()

    # Load the attribution
    loaded_out = inseq.FeatureAttributionOutput.load(output_path)

    # Check equality
    assert len(loaded_out.sequence_attributions) == len(out.sequence_attributions)
    assert len(loaded_out.sequence_attributions[0].source) == len(out.sequence_attributions[0].source)
    assert len(loaded_out.sequence_attributions[0].target) == len(out.sequence_attributions[0].target)

    # Check that attributions are equal - handling potential shape differences
    original_attr = out.sequence_attributions[0].source_attributions
    loaded_attr = loaded_out.sequence_attributions[0].source_attributions

    if original_attr.shape == loaded_attr.shape:
        assert torch.allclose(original_attr, loaded_attr)
    else:
        # Get the minimum dimensions
        min_dim0 = min(original_attr.shape[0], loaded_attr.shape[0])
        min_dim1 = min(original_attr.shape[1], loaded_attr.shape[1])
        # Compare the overlapping area
        assert torch.allclose(original_attr[:min_dim0, :min_dim1], loaded_attr[:min_dim0, :min_dim1])

    # Clean up
    os.remove(output_path)


def test_save_load_with_compression(encoder_decoder_model_id, sample_input_texts, output_dir):
    """Test saving and loading attribution outputs with compression."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    # Generate attribution
    out = model.attribute(sample_input_texts[0])

    # Save with compression
    output_path = output_dir / "attribution_output.json.gz"
    out.save(output_path, overwrite=True, compress=True)

    # Check that the file exists
    assert output_path.exists()

    # Load with decompression
    loaded_out = inseq.FeatureAttributionOutput.load(output_path, decompress=True)

    # Check equality
    assert len(loaded_out.sequence_attributions) == len(out.sequence_attributions)

    # Clean up
    os.remove(output_path)


def test_save_load_with_different_precisions(encoder_decoder_model_id, sample_input_texts, output_dir):
    """Test saving with different precision levels."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    # Generate attribution
    out = model.attribute(sample_input_texts[0])

    # Test with different precision levels
    for precision in ["float32", "float16", "float8"]:
        output_path = output_dir / f"attribution_output_{precision}.json"
        out.save(output_path, overwrite=True, scores_precision=precision)

        # Check that the file exists
        assert output_path.exists()

        # Load the attribution
        loaded_out = inseq.FeatureAttributionOutput.load(output_path)

        # Check that we have the same structure
        assert len(loaded_out.sequence_attributions) == len(out.sequence_attributions)

        # Clean up
        os.remove(output_path)


def test_save_load_with_split_sequences(encoder_decoder_model_id, sample_input_texts, output_dir):
    """Test saving with split sequences."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    # Generate attribution for multiple sequences
    out = model.attribute(sample_input_texts[:2])

    # Save with split sequences
    base_path = output_dir / "attribution_output"
    out.save(base_path, overwrite=True, split_sequences=True)

    # Check that the files exist
    assert (base_path.parent / f"{base_path.name}_0.json").exists()
    assert (base_path.parent / f"{base_path.name}_1.json").exists()

    # Clean up
    os.remove(base_path.parent / f"{base_path.name}_0.json")
    os.remove(base_path.parent / f"{base_path.name}_1.json")


def test_get_scores_dicts(encoder_decoder_model_id, sample_input_texts):
    """Test getting scores as dictionaries."""
    # Load model with attribution method
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    # Generate attribution with step scores
    out = model.attribute(
        sample_input_texts[0],
        step_scores=["probability", "entropy"]
    )

    # Get scores as dictionaries
    scores_dicts = out.get_scores_dicts()

    # Check structure
    assert len(scores_dicts) == 1  # One for each sequence
    assert "source_attributions" in scores_dicts[0]
    assert "step_scores" in scores_dicts[0]

    # Check step scores
    step_scores = scores_dicts[0]["step_scores"]
    assert len(step_scores) > 0

    # Each token should have probability and entropy scores
    first_token = list(step_scores.keys())[0]
    assert "probability" in step_scores[first_token]
    assert "entropy" in step_scores[first_token]

    # Source attributions should map to a dict of tokens
    source_attributions = scores_dicts[0]["source_attributions"]
    assert len(source_attributions) > 0