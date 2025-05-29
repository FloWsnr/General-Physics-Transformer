#!/bin/bash

### Task name
#SBATCH --account=sds_baek_energetic

### Job name
#SBATCH --job-name=eval_lpfm

### Output file
#SBATCH --output=/scratch/zsa8rk/logs/00_slrm_logs/eval_lpfm_%j.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=70

### How much memory in total (MB)
#SBATCH --mem=600G

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsa8rk@virginia.edu

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task (v100, a100, h200)
#SBATCH --gres=gpu:a100:4
##SBATCH --constraint=a100_80gb


### Partition
#SBATCH --partition=gpu

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
##SBATCH --array=1-10%1

### Set the time limit for the job, allows for graceful shutdown
### Should be lower than the time limit of the partition
### Format: HH:MM:SS
time_limit="24:00:00"

#####################################################################################
############################# Setup #################################################
#####################################################################################

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate lpfm

######################################################################################
############################# Set paths ##############################################
######################################################################################
# debug mode
# debug=true
# Set up paths
base_dir="/home/zsa8rk/Coding/Large-Physics-Foundation-Model"
python_exec="${base_dir}/lpfm/run/model_eval.py"
log_dir="/scratch/zsa8rk/logs"
base_config_file="${base_dir}/lpfm/run/scripts/config.yaml"
data_dir="/scratch/zsa8rk/datasets"
# sim_name (same as wandb id)
sim_name="m-main-4-1"
nnodes=1
ngpus_per_node=4
export OMP_NUM_THREADS=1 # (num cpu - num_workers) / num_gpus

# name of the checkpoint to use for evaluation. Can be "best_model" or a number of a epoch directory
checkpoint_name="best_model"

# sim directory
sim_dir="${log_dir}/${sim_name}"

#######################################################################################
############################# Setup sim dir and config file ###########################
#######################################################################################

# delete the sim_dir if it exists and debug is true
if [ "$debug" = true ]; then
    rm -rf $sim_dir
fi

# create the sim_dir if it doesn't exist
mkdir -p $sim_dir

# copy the slurm script to the sim_dir with .sh suffix
cp "$0" "${sim_dir}/slurm_eval_script.sh"

# Try to find config file in sim_dir
config_file="${sim_dir}/config.yaml"
if [ ! -f "$config_file" ]; then
    echo "No config file found in $sim_dir, copying base config..."
    cp $base_config_file $sim_dir
fi

#####################################################################################
############################# Evaluation ############################################
#####################################################################################
echo "--------------------------------"
echo "Starting LPFM evaluation..."
echo "config_file: $config_file"
echo "sim_dir: $sim_dir"
echo "using checkpoint: $checkpoint_name"
echo "--------------------------------"

exec_args="--config_file $config_file \
    --sim_name $sim_name \
    --log_dir $log_dir \
    --data_dir $data_dir \
    --checkpoint_name $checkpoint_name"

# Capture Python output and errors in a variable and run the script
torchrun --standalone --nproc_per_node=$ngpus_per_node $python_exec $exec_args

# move the output file to the sim_dir
mv ${log_dir}/slrm_logs/eval_${sim_name}_${SLURM_JOB_ID}.out $sim_dir 