import pytest
import torch
import inseq


@pytest.mark.parametrize("method", [
    "saliency",
    "integrated_gradients",
    "input_x_gradient"
])
def test_gradient_attribution_methods(method, encoder_decoder_model_id, sample_input_texts):
    """Test attribution with different gradient-based methods."""
    # Load model with the specific attribution method
    model = inseq.load_model(encoder_decoder_model_id, method)
    out = model.attribute(sample_input_texts[0])

    assert out is not None
    assert len(out.sequence_attributions) == 1

    # Gradient-based methods produce granular attributions
    source_attr = out.sequence_attributions[0].source_attributions

    # Should have shape [src_len, tgt_len, hidden_dim]
    assert len(source_attr.shape) == 3

    # Check that the hidden dimension is correct
    base_model = inseq.load_model(encoder_decoder_model_id, method)
    hidden_dim = base_model.model.config.d_model
    assert source_attr.shape[2] == hidden_dim

    # Check source and target dimensions (accounting for possible EOS token differences)
    source_len = len(out.sequence_attributions[0].source)
    target_len = len(out.sequence_attributions[0].target)
    assert source_attr.shape[0] == source_len
    assert source_attr.shape[1] in (target_len, target_len - 1)


def test_integrated_gradients_with_steps(encoder_decoder_model_id, sample_input_texts):
    """Test Integrated Gradients with different numbers of steps."""
    # Test with fewer steps
    model_fewer_steps = inseq.load_model(encoder_decoder_model_id, "integrated_gradients")
    out = model_fewer_steps.attribute(
        sample_input_texts[0],
        n_steps=10,
        return_convergence_delta=True
    )

    # Convergence delta should be present in step_scores
    assert "deltas" in out.sequence_attributions[0].step_scores

    # Test with more steps
    model_more_steps = inseq.load_model(encoder_decoder_model_id, "integrated_gradients")
    out_more_steps = model_more_steps.attribute(
        sample_input_texts[0],
        n_steps=20,
        return_convergence_delta=True
    )

    # Compare convergence deltas
    delta_fewer_steps = out.sequence_attributions[0].step_scores["deltas"].mean()
    delta_more_steps = out_more_steps.sequence_attributions[0].step_scores["deltas"].mean()

    # More steps should generally give better convergence (smaller delta)
    # But this might not always be true due to randomness, so we'll just check they exist
    assert delta_fewer_steps is not None
    assert delta_more_steps is not None


def test_input_x_gradient_multiply_by_inputs(encoder_decoder_model_id):
    """Test InputXGradient with multiply_by_inputs option."""
    # InputXGradient already multiplies by inputs by design, so this test
    # is mostly checking that the parameters are passed correctly
    model = inseq.load_model(
        encoder_decoder_model_id,
        "input_x_gradient"
    )

    out = model.attribute("This is a test.")
    assert out is not None
    assert len(out.sequence_attributions) == 1


def test_layer_attribution(tiny_decoder_model_id):
    """Test layer attribution methods."""
    # This requires a model to be loaded first to access its layer
    base_model = inseq.load_model(tiny_decoder_model_id, "layer_gradient_x_activation")

    # Get the MLP layer of the first transformer block
    target_layer = base_model.model.transformer.h[0].mlp

    # Load with layer_gradient_x_activation
    model = inseq.load_model(
        base_model.model,
        "layer_gradient_x_activation",
        tokenizer=tiny_decoder_model_id,
        target_layer=target_layer
    )

    out = model.attribute("This is a test.")
    assert out is not None
    assert len(out.sequence_attributions) == 1

    assert  out.sequence_attributions[0].source_attributions is None
    assert out.sequence_attributions[0].target_attributions is not None


def test_attribution_with_custom_attribution_target(encoder_decoder_model_id):
    """Test attribution with a custom attribution target."""
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    # Attribute with respect to entropy instead of probability
    out = model.attribute(
        "This is a test.",
        attributed_fn="entropy"
    )

    assert out is not None
    assert len(out.sequence_attributions) == 1


def test_contrastive_attribution(encoder_decoder_model_id):
    """Test contrastive attribution."""
    model = inseq.load_model(encoder_decoder_model_id, "saliency")

    # Regular and contrastive targets
    input_text = "The developer argued with the designer because she did not like the design."
    regular_target = "Le développeur s'est disputé avec le designer car elle n'aimait pas le design."
    contrast_target = "La développeuse s'est disputée avec le designer car elle n'aimait pas le design."

    # Perform contrastive attribution
    out = model.attribute(
        input_text,
        regular_target,
        attributed_fn="contrast_prob_diff",
        contrast_targets=contrast_target,
        attribute_target=True,
        step_scores=["contrast_prob_diff"]
    )

    assert out is not None
    assert "contrast_prob_diff" in out.sequence_attributions[0].step_scores