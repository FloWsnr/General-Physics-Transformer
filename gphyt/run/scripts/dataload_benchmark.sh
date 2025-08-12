#!/usr/bin/bash

### Task name
#SBATCH --job-name=dataload_benchmark

### Output file
#SBATCH --output=/hpcwork/rwth1802/coding/General-Physics-Transformer/results/slrm_logs/dataload_benchmark_%j.out

### Start a single-node job for dataloader benchmarking
#SBATCH --nodes=1

### How many CPU cores to use (adjust based on num_workers to test)
#SBATCH --ntasks-per-node=32

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsa8rk@virginia.edu

### Maximum runtime per task (short job for benchmarking)
#SBATCH --time=24:00:00

#SBATCH --partition=standard

### No GPUs needed for dataloader benchmarking
### Uncomment next line if you want to test with GPU data transfers
##SBATCH --gres=gpu:1

#####################################################################################
############################# Setup #################################################
#####################################################################################

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate gphyt

######################################################################################
############################# Set paths and parameters ##############################
######################################################################################

# Set up paths
base_dir="/hpcwork/rwth1802/coding/General-Physics-Transformer"
python_exec="${base_dir}/gphyt/run/scripts/dataload_benchmark.py"
log_dir="${base_dir}/results"
data_dir="${base_dir}/data/datasets"

# Benchmark parameters - modify these as needed
n_batches=1000
batch_size=32

# Create results directory if it doesn't exist
mkdir -p "${log_dir}/slrm_logs"

#####################################################################################
############################# Dataloader Benchmarking ##############################
#####################################################################################

echo "=================================="
echo "Starting Dataloader Benchmarking"
echo "=================================="
echo "Data directory: $data_dir"
echo "Number of batches: $n_batches"
echo "Batch size: $batch_size"
echo "=================================="

# Test different worker configurations
worker_configs=(2 4 8 16)
prefetch_configs=(1 2 4 8)
omp_threads=(1 2 4 8 16) # number of OMP_NUM_THREADS

for num_workers in "${worker_configs[@]}"; do
    for prefetch_factor in "${prefetch_configs[@]}"; do
        for omp_thread in "${omp_threads[@]}"; do
            echo ""
            echo "----------------------------------------"
            echo "Testing: workers=$num_workers, prefetch=$prefetch_factor, omp_threads=$omp_thread"
            echo "----------------------------------------"

            # Set OMP_NUM_THREADS to avoid oversubscription
            export OMP_NUM_THREADS=$omp_thread

            # Run the benchmark
            /home/fw641779/miniforge3/envs/gphyt/bin/python $python_exec \
                --data_dir "$data_dir" \
                --n_batches $n_batches \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --prefetch_factor $prefetch_factor

            echo "----------------------------------------"
            echo "Completed: workers=$num_workers, prefetch=$prefetch_factor, omp_threads=$omp_thread"
            echo "----------------------------------------"
        done
    done
done

echo ""
echo "=================================="
echo "Dataloader Benchmarking Complete"
echo "=================================="

# Move the output file to results directory for easy access
benchmark_output="${log_dir}/slrm_logs/dataload_benchmark_${SLURM_JOB_ID}.out"
echo "Output saved to: $benchmark_output"