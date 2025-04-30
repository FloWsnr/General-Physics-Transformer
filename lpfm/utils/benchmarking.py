import timeit
import random
import psutil
import os
from typing import Tuple, Dict

from lpfm.data.phys_dataset import PhysicsDataset


def benchmark_dataset(dataset: PhysicsDataset, num_samples: int = 1000) -> Dict[str, float]:
    """
    Benchmark the dataset loading performance including RAM usage and timing.

    Parameters
    ----------
    dataset : PhysicsDataset
        The dataset to benchmark
    num_samples : int, optional
        Number of random samples to load, by default 100

    Returns
    -------
    Dict[str, float]
        Dictionary containing benchmark metrics:
        - 'avg_time': Average time per sample in seconds
        - 'max_ram': Maximum RAM usage in MB
        - 'total_time': Total time for all samples in seconds
    """
    process = psutil.Process(os.getpid())
    initial_ram = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Generate random indices
    indices = [random.randint(0, len(dataset) - 1) for _ in range(num_samples)]
    
    # Warmup
    for i in range(10):
        idx = random.randint(0, len(dataset) - 1)
        x, y = dataset[idx]
    
    # Benchmark
    start_time = timeit.default_timer()
    max_ram = initial_ram
    
    for idx in indices:
        x, y = dataset[idx]
        current_ram = process.memory_info().rss / 1024 / 1024
        max_ram = max(max_ram, current_ram)
    
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    
    return {
        'avg_time [s]': total_time / num_samples,
        'max_ram [MB]': max_ram - initial_ram,
        'total_time [s]': total_time
    }

if __name__ == "__main__":
    dataset = PhysicsDataset(
        "/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/data/datasets/euler_multi_quadrants_periodicBC/data/train",
        split="train",
        n_steps_input=4,
        n_steps_output=1,
    )
    print(benchmark_dataset(dataset))