#!/usr/bin/env python3
"""
Dataloader benchmarking script for GPhyT datasets.

This script creates a dataloader with the specified configuration and measures
the time to load N batches, providing performance metrics for dataloader tuning.
"""

import argparse
import time
from pathlib import Path

from gphyt.data.dataset_utils import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch dataloader performance"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the datasets directory"
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=100,
        help="Number of batches to load for benchmarking (default: 100)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor for dataloader (default: 2)",
    )
    return parser.parse_args()


def benchmark_dataloader(dataloader, n_batches):
    """Benchmark dataloader by loading n_batches and measuring time."""
    print(f"Starting benchmark: loading {n_batches} batches...")

    # Warmup - load a few batches to initialize everything
    print("Warming up...")
    warmup_batches = min(5, n_batches)
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break

    # Actual benchmarking
    print("Running benchmark...")
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        # Simulate some basic operations that might be done with the batch
        x, y = batch
        _ = x.shape, y.shape  # Just access the shapes

        if i + 1 >= n_batches:
            break

        if (i + 1) % 10 == 0:
            print(f"Loaded {i + 1}/{n_batches} batches...")

    end_time = time.time()
    total_time = end_time - start_time

    return total_time


def main():
    args = parse_args()

    print("=" * 60)
    print("DATALOADER BENCHMARK")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Prefetch factor: {args.prefetch_factor}")
    print(f"Number of batches to benchmark: {args.n_batches}")
    print("-" * 60)

    # Create data configuration
    data_config = {
        "data_dir": args.data_dir,
        "datasets": [
            "cylinder_sym_flow_water",
            "cylinder_pipe_flow_water",
            "object_periodic_flow_water",
            "object_sym_flow_water",
            "object_sym_flow_air",
            "heated_object_pipe_flow_air",
            "cooled_object_pipe_flow_air",
            "rayleigh_benard_obstacle",
            "twophase_flow",
            "rayleigh_benard",
            "shear_flow",
            "euler_multi_quadrants_periodicBC",
            "acoustic_scattering_inclusions",
        ],
        "n_steps_input": 4,
        "n_steps_output": 1,
        "dt_stride": [1, 8],
        "use_normalization": True,
        "flip_x": 0.5,
        "flip_y": 0.5,
    }

    print("Creating dataloader...")
    try:
        num_workers = args.num_workers
        prefetch_factor = args.prefetch_factor
        if num_workers == 0:
            prefetch_factor = None

        dataloader = get_dataloader(
            data_config=data_config,
            seed=42,
            batch_size=args.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            data_fraction=1.0,
            split="train",
            is_distributed=False,
            shuffle=True,
        )
        print(
            f"Dataloader created successfully with {len(dataloader.dataset)} total samples"
        )
        print(f"Expected number of batches per epoch: {len(dataloader)}")

    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return

    # Run benchmark
    try:
        total_time = benchmark_dataloader(dataloader, args.n_batches)

        # Calculate statistics
        avg_batch_time = total_time / args.n_batches

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Average batch loading time: {avg_batch_time:.4f} seconds")
        print("-" * 60)

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return


if __name__ == "__main__":
    main()
