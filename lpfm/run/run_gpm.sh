#!/usr/bin/bash

### Task name
#SBATCH --account=rwth1802
#SBATCH --job-name=train_gpm

### Output file
#SBATCH --output=/hpcwork/rwth1802/Coding/MetaPARC/logs/train_gpm_%j.out

### Default is 2540 MiB memory per (CPU) task = MPI rank
## Can be increased if larger partitions are used
##SBATCH --mem-per-cpu=2540

### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### Number of tasks (MPI ranks)
#SBATCH --ntasks=24

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsa8rk@uva.virginia.edu

### Maximum runtime per task
#SBATCH --time=1-00:00:00

### set number of GPUs per task
#SBATCH --gres=gpu:1

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
#SBATCH --array=1-10%1


#####################################################################################
############################# Setup #################################################
#####################################################################################
# Set up paths
python_exec="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/lpfm/run/train_slrm.py"
log_dir="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs/"
data_dir="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/data/datasets"
# sim_name (same as wandb id)
sim_name="gpm_run_1"
# sim directory
sim_dir="${log_dir}/${sim_name}"
# Get the actual SLURM output file path
SLURM_OUTPUT="${sim_dir}/logs/${SLURM_JOB_ID}.out"

# Try to find config file in sim_dir
config_file="${sim_dir}/config.yaml"
if [ -f "$config_file" ]; then
    echo "Config file found in $sim_dir, attempting restart..."
    restart=true
else
    echo "No config file found in $sim_dir, starting new training..."
    config_file="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/lpfm/run/config.yaml"
    restart=false
fi

# Load modules
module purge
module load CUDA/12.6.0

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate gpm


#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting GPM training..."
echo "config_file: $config_file"
echo "sim_dir: $sim_dir"
echo "restart: $restart"
echo "--------------------------------"

# Capture Python output and errors in a variable and run the script
python_output=$(python $python_exec \
    --config_file $config_file \
    --sim_name $sim_name \
    --sim_dir $sim_dir \
    --restart $restart \
    --data_dir $data_dir 2>&1)
# Write both the Python output/errors
echo "$python_output" >> $SLURM_OUTPUT