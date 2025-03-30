import pytest
import inseq


def test_default_aggregation(encoder_decoder_model_id, sample_input_texts):
    """Test default aggregation behavior."""
    # Use integrated_gradients to get granular attributions
    model = inseq.load_model(encoder_decoder_model_id, "integrated_gradients")
    out = model.attribute(sample_input_texts[0])

    # Get the original shape
    original_attr = out.sequence_attributions[0].source_attributions
    original_shape = original_attr.shape

    # Aggregate
    aggregated_out = out.aggregate()
    aggregated_attr = aggregated_out.sequence_attributions[0].source_attributions

    # For gradient attributions, aggregation should reduce the last dimension
    assert len(aggregated_attr.shape) < len(original_shape)
    assert aggregated_attr.shape[0] == original_shape[0]  # Source length preserved
    # Target length may be preserved or off by one due to token handling
    assert aggregated_attr.shape[1] in (original_shape[1], original_shape[1] - 1)


@pytest.mark.parametrize("aggregator", ["scores", "spans", "subwords"])
def test_different_aggregators(aggregator, encoder_decoder_model_id, sample_input_texts):
    """Test different aggregators."""
    model = inseq.load_model(encoder_decoder_model_id, "integrated_gradients")
    out = model.attribute(sample_input_texts[0])

    # Aggregate with the specified aggregator
    aggregated_out = out.aggregate(aggregator=aggregator)

    # Should get a valid result
    assert aggregated_out is not None
    assert len(aggregated_out.sequence_attributions) == 1


def test_aggregator_pipeline(encoder_decoder_model_id):
    """Test aggregator pipeline."""
    model = inseq.load_model(encoder_decoder_model_id, "integrated_gradients")

    # Use a text with subwords to test subword aggregation
    out = model.attribute("The unbelievable phenomenon happened yesterday.")

    # Create an aggregator pipeline
    pipeline = inseq.data.aggregator.AggregatorPipeline([
        inseq.data.aggregator.SubwordAggregator,
        inseq.data.aggregator.SequenceAttributionAggregator
    ])

    # Aggregate with the pipeline
    aggregated_out = out.aggregate(aggregator=pipeline)

    # Should get a valid result with reduced dimensions
    assert aggregated_out is not None
    assert len(aggregated_out.sequence_attributions) == 1

    # The number of tokens should be reduced
    original_tokens = out.sequence_attributions[0].source
    aggregated_tokens = aggregated_out.sequence_attributions[0].source

    # There should be fewer tokens after aggregation if subwords were merged
    assert len(aggregated_tokens) <= len(original_tokens)

    # And dimensions should be reduced
    original_shape = out.sequence_attributions[0].source_attributions.shape
    aggregated_shape = aggregated_out.sequence_attributions[0].source_attributions.shape
    assert len(aggregated_shape) < len(original_shape)


def test_subword_aggregation(encoder_decoder_model_id):
    """Test specifically the subword aggregator."""
    model = inseq.load_model(encoder_decoder_model_id, "integrated_gradients")

    # Use a text with tokenized subwords
    out = model.attribute("Tokenizing unprecedentedly complicated words.")

    # Aggregate with subword aggregator
    aggregated_out = out.aggregate(aggregator="subwords")

    # Original tokens
    original_tokens = [t.token for t in out.sequence_attributions[0].source]

    # Aggregated tokens
    aggregated_tokens = [t.token for t in aggregated_out.sequence_attributions[0].source]

    # We expect fewer tokens after aggregation
    assert len(aggregated_tokens) < len(original_tokens)

    # Print the tokens for debugging
    print(f"Original tokens: {original_tokens}")
    print(f"Aggregated tokens: {aggregated_tokens}")


def test_contiguous_span_aggregation(encoder_decoder_model_id):
    """Test the contiguous span aggregator."""
    model = inseq.load_model(encoder_decoder_model_id, "integrated_gradients")
    out = model.attribute("This is a test with spans.")

    # Define spans to aggregate
    source_spans = [(0, 2)]  # Aggregate "This is"

    # Aggregate with contiguous span aggregator
    aggregated_out = out.aggregate(
        aggregator="spans",
        source_spans=source_spans
    )

    # Original tokens
    original_tokens = [t.token for t in out.sequence_attributions[0].source]

    # Aggregated tokens
    aggregated_tokens = [t.token for t in aggregated_out.sequence_attributions[0].source]

    # We expect fewer tokens after aggregation
    assert len(aggregated_tokens) < len(original_tokens)

    # First token should be merged
    original_span = original_tokens[0:2]
    print(f"Original span: {original_span}")
    print(f"Aggregated tokens: {aggregated_tokens}")
    assert len(aggregated_tokens) == len(original_tokens) - len(source_spans[0]) + 1


def test_pair_aggregator(encoder_decoder_model_id):
    """Test the pair aggregator for comparing attributions."""
    model = inseq.load_model(encoder_decoder_model_id, "integrated_gradients")

    # Create two similar attributions
    out1 = model.attribute("The manager told the employee to finish the report.")
    out2 = model.attribute("The manager asked the employee to finish the report.")

    # Pre-aggregate them to get token-level attributions
    agg1 = out1.aggregate()
    agg2 = out2.aggregate()

    # Use pair aggregator to get the difference
    diff = agg1.sequence_attributions[0].aggregate(
        aggregator="pair",
        paired_attr=agg2.sequence_attributions[0]
    )

    # The result should have attributions (difference between the two)
    assert diff.source_attributions is not None

    # Shape should be the same as the original attributions
    assert diff.source_attributions.shape == agg1.sequence_attributions[0].source_attributions.shape