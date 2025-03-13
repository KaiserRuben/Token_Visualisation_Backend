#!/usr/bin/env python3
"""
This script tests PyTorch's CPU parallelism capabilities
and provides a diagnostic report on core utilization.

It's useful for verifying that your system can properly utilize multiple cores
with PyTorch operations.
"""

import os
import time
import torch
import multiprocessing as mp
import numpy as np
import argparse


def configure_env():
    """Set environment variables for optimal PyTorch parallelism."""
    cpu_count = mp.cpu_count()
    print(f"Setting environment for {cpu_count} CPU cores")

    # Set threading environment variables
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)

    # Configure OpenMP for better core utilization
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_PROC_BIND"] = "close"

    # Set PyTorch threads
    torch.set_num_threads(cpu_count)

    # Set inter-op parallelism if available
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(min(cpu_count, 4))
        print(f"Set inter-op threads to {min(cpu_count, 4)}")


def print_config():
    """Print current PyTorch and system configuration."""
    print("\n=== PyTorch Configuration ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of CPUs: {mp.cpu_count()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # Check PyTorch build info
    build_info = []
    for attr in dir(torch):
        if attr.startswith('has_') and callable(getattr(torch, attr)) and getattr(torch, attr)():
            build_info.append(attr[4:])  # Remove 'has_' prefix
    print(f"PyTorch build info: {', '.join(build_info)}")

    # Print environment variables
    print("\n=== Environment Variables ===")
    env_vars = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "OMP_PLACES", "OMP_PROC_BIND"
    ]
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'not set')}")


def test_matmul(size=5000):
    """Test parallelism with matrix multiplication."""
    print(f"\n=== Testing Matrix Multiplication ({size}x{size}) ===")

    # Create random matrices
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # Time the operation
    start = time.time()
    c = torch.matmul(a, b)
    elapsed = time.time() - start

    print(f"Matrix multiplication completed in {elapsed:.2f} seconds")
    return elapsed


def test_batch_mm(size=2000, batch_size=64):
    """Test parallelism with batched matrix operations."""
    print(f"\n=== Testing Batched Matrix Operations ({batch_size}x{size}x{size}) ===")

    # Create random batch of matrices
    a = torch.randn(batch_size, size, size)
    b = torch.randn(batch_size, size, size)

    # Time the operation
    start = time.time()
    c = torch.bmm(a, b)
    elapsed = time.time() - start

    print(f"Batched matrix multiplication completed in {elapsed:.2f} seconds")
    return elapsed


def test_conv(size=512, channels=32):
    """Test parallelism with convolution operations."""
    print(f"\n=== Testing 2D Convolution (Size: {size}, Channels: {channels}) ===")

    # Create random input and kernel
    input = torch.randn(4, channels, size, size)
    weight = torch.randn(channels * 2, channels, 3, 3)

    # Time the operation
    start = time.time()
    output = torch.nn.functional.conv2d(input, weight, padding=1)
    elapsed = time.time() - start

    print(f"2D Convolution completed in {elapsed:.2f} seconds")
    return elapsed


def test_attribution_like_workload(batch_size=8, seq_len=128, hidden_size=768):
    """Simulate an attribution-like workload with back-propagation."""
    print(f"\n=== Testing Attribution-like Workload (Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_size}) ===")

    # Create a simple transformer-like layer
    query = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    key = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    value = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

    # Time the forward and backward pass
    start = time.time()

    # Forward pass (similar to attention mechanism)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_size ** 0.5)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, value)

    # Compute a loss-like value
    loss = context.sum()

    # Backward pass
    loss.backward()

    elapsed = time.time() - start

    print(f"Attribution-like workload completed in {elapsed:.2f} seconds")
    return elapsed


def run_tests(args):
    """Run all performance tests."""
    results = {}

    # Basic matrix multiplication
    results['matmul'] = test_matmul(args.matrix_size)

    # Batched matrix operations
    results['batch_mm'] = test_batch_mm(args.batch_matrix_size, args.batch_size)

    # Convolution operations
    results['conv'] = test_conv(args.conv_size, args.channels)

    # Attribution-like workload
    results['attribution'] = test_attribution_like_workload(
        args.attr_batch_size, args.attr_seq_len, args.attr_hidden_size
    )

    return results


def main():
    parser = argparse.ArgumentParser(description='Test PyTorch CPU parallelism')

    # Test configuration
    parser.add_argument('--matrix-size', type=int, default=5000,
                        help='Size of matrices for matmul test')
    parser.add_argument('--batch-matrix-size', type=int, default=2000,
                        help='Size of matrices for batch matmul test')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for batch matmul test')
    parser.add_argument('--conv-size', type=int, default=512,
                        help='Input size for convolution test')
    parser.add_argument('--channels', type=int, default=32,
                        help='Number of channels for convolution test')
    parser.add_argument('--attr-batch-size', type=int, default=8,
                        help='Batch size for attribution test')
    parser.add_argument('--attr-seq-len', type=int, default=128,
                        help='Sequence length for attribution test')
    parser.add_argument('--attr-hidden-size', type=int, default=768,
                        help='Hidden size for attribution test')

    args = parser.parse_args()

    print("=" * 50)
    print("PyTorch CPU Parallelism Test")
    print("=" * 50)

    # Configure environment for optimal performance
    configure_env()

    # Print current configuration
    print_config()

    # Run tests
    results = run_tests(args)

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    for test, elapsed in results.items():
        print(f"{test}: {elapsed:.2f} seconds")

    print("\nNote: For ideal parallelism, you should observe CPU usage")
    print("across multiple cores during these tests. If only one core")
    print("is being utilized, there may be issues with your PyTorch")
    print("installation or system configuration.")


if __name__ == "__main__":
    main()