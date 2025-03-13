#!/usr/bin/env python3
"""
Basic attribution inference test script - monitors CPU utilization
during standard attribution without forcing parallelism.

Usage:
    python basic_attribution_test.py
"""

import os
import time
import torch
import inseq
import psutil
import threading
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def monitor_cpu_usage(stop_event, interval=0.5):
    """
    Monitor CPU usage across all cores.

    Args:
        stop_event: Threading event to signal when to stop monitoring
        interval: How often to sample CPU usage (in seconds)
    """
    process = psutil.Process()
    cpu_percentages = []
    per_cpu_usage = []

    logger.info(f"Starting CPU usage monitor (sampling every {interval}s)")

    while not stop_event.is_set():
        # Get CPU usage for this process
        try:
            # Get per-CPU usage for this process
            cpu_percent = process.cpu_percent(interval=interval)
            cpu_percentages.append(cpu_percent)

            # Get per-CPU system-wide usage
            per_cpu = psutil.cpu_percent(interval=0, percpu=True)
            per_cpu_usage.append(per_cpu)

            # Log current usage
            logger.info(f"Current CPU usage: {cpu_percent:.1f}% | Per CPU: {per_cpu}")

        except Exception as e:
            logger.error(f"Error monitoring CPU: {e}")
            break

    # Calculate statistics
    if cpu_percentages:
        avg_usage = sum(cpu_percentages) / len(cpu_percentages)
        max_usage = max(cpu_percentages)

        # Calculate the maximum usage for each CPU core
        per_cpu_max = [0] * len(per_cpu_usage[0])
        for sample in per_cpu_usage:
            for i, usage in enumerate(sample):
                per_cpu_max[i] = max(per_cpu_max[i], usage)

        logger.info(f"Average CPU usage: {avg_usage:.1f}%")
        logger.info(f"Maximum CPU usage: {max_usage:.1f}%")
        logger.info(f"Maximum per-CPU usage: {per_cpu_max}")
        logger.info(f"Number of CPU cores actively used: {sum(1 for x in per_cpu_max if x > 50)}")


def run_basic_attribution_test():
    """
    Run a basic attribution test without modifying any parallelism settings.
    """
    # Default parameters - using a smaller, well-tested model
    model_name = "distilgpt2"  # More reliable for testing
    method = "input_x_gradient"
    input_text = "The quick brown fox jumps over the lazy dog."

    logger.info("Starting basic attribution test with default settings")
    logger.info(f"Model: {model_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Input text: {input_text}")

    # Log default PyTorch threading setup (without modifying it)
    logger.info(f"Default PyTorch threads: {torch.get_num_threads()}")
    logger.info(f"CPU count: {psutil.cpu_count(logical=True)}")

    # Check environment variables (without setting them)
    env_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    for var in env_vars:
        logger.info(f"{var}: {os.environ.get(var, 'not set')}")

    try:
        # Start CPU monitoring in a separate thread
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_cpu_usage,
            args=(stop_monitoring,)
        )
        monitor_thread.start()

        # Load the model
        logger.info(f"Loading model {model_name}...")
        start_time = time.time()

        model_kwargs = {"attn_implementation": "eager"}
        tokenizer_kwargs = {
            "padding_side": "left",
            "pad_token": "<pad>"
        }

        try:
            # First attempt with standard loading
            model = inseq.load_model(
                model_name,
                method,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs
            )
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.warning(f"Error in standard model loading: {e}")

            # Alternative loading approach for more reliability
            logger.info("Trying alternative model loading approach...")
            import transformers

            # Load model and tokenizer separately
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_class = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            model = inseq.load_model(model_class, method, tokenizer=tokenizer)
            logger.info(f"Model loaded with alternative approach in {time.time() - start_time:.2f} seconds")

        # Ensure model is on CPU
        if hasattr(model, 'to'):
            model.to('cpu')
            logger.info("Model moved to CPU")

        # Basic tokenizer setup if needed
        if model.tokenizer.unk_token_id is None:
            model.tokenizer.unk_token = model.tokenizer.convert_ids_to_tokens(0)
            model.tokenizer.unk_token_id = 0

        if model.tokenizer.pad_token is None:
            if model.tokenizer.eos_token is not None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            else:
                model.tokenizer.pad_token = "<pad>"

        # Perform attribution with careful error handling
        logger.info("Starting attribution...")
        attribution_start = time.time()

        try:
            # Validate input text with tokenizer first
            tokens = model.tokenizer(input_text, return_tensors="pt")
            logger.info(f"Input tokenized to {len(tokens.input_ids[0])} tokens")

            # Check if tokens are in valid range
            max_id = model.tokenizer.vocab_size - 1
            if tokens.input_ids.max() > max_id:
                logger.warning(f"Token ID out of range! Max valid: {max_id}, Found: {tokens.input_ids.max()}")
                # Use a simpler input if original caused problems
                input_text = "Hello world"
                logger.info(f"Switching to simpler input text: '{input_text}'")

            # Run attribution
            attribution = model.attribute(
                input_text,
                step_scores=["probability"]
            )

            # Print attribution data for verification
            seq_attr = attribution.sequence_attributions[0]
            logger.info(
                f"Attribution created with {len(seq_attr.source)} source tokens and {len(seq_attr.target)} target tokens")

        except Exception as attr_error:
            logger.error(f"Attribution failed with error: {attr_error}")
            logger.info("Trying with simplified settings...")

            # Try a more basic attribution approach
            try:
                # Try with minimal configuration
                attribution = model.attribute(
                    "Hello",  # Minimal input
                    step_scores=["probability"],
                )
                logger.info("Simplified attribution succeeded")
            except Exception as e:
                logger.error(f"Even simplified attribution failed: {e}")
                # Let the error propagate
                raise

        attribution_time = time.time() - attribution_start
        logger.info(f"Attribution completed in {attribution_time:.2f} seconds")

        # Stop CPU monitoring
        stop_monitoring.set()
        monitor_thread.join()

        # Print summary
        logger.info("Attribution test completed successfully")
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.exception(f"Error during attribution test: {e}")
        if 'stop_monitoring' in locals():
            stop_monitoring.set()
        if 'monitor_thread' in locals() and monitor_thread.is_alive():
            monitor_thread.join()


if __name__ == "__main__":
    run_basic_attribution_test()