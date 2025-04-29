#!/usr/bin/bash

### Task name
#SBATCH --job-name=compress_data

### Output file
#SBATCH --output=/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs/slrm_logs/compress_data_%j.out


### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=8

### Mail notification configuration
#SBATCH --mail-type=NONE
#SBATCH --mail-user=zsa8rk@virginia.edu

### Maximum runtime per task
#SBATCH --time=24:00:00

module purge

path="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/data/datasets"

tar -cvf - "$path" | pigz -p 8 --verbose > compressed_data.tar.gz