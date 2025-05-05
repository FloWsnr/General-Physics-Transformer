#!/bin/bash

### Task name
#SBATCH --account=sds_baek_energetic

### Output file
#SBATCH --output=/home/zsa8rk/Coding/Large-Physics-Foundation-Model/logs/slrm_logs/train_lpfm_%j.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=70

### How much memory in total (MB)
#SBATCH --mem=300G

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task (v100, a100, h200)
#SBATCH --gres=gpu:4:a100
##SBATCH --constraint="a80|a40"
## SBATCH -C gpupod # use pod gpus...


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
python_exec="${base_dir}/lpfm/run/train.py"
log_dir="${base_dir}/logs"
base_config_file="${base_dir}/lpfm/run/config.yaml"
data_dir="/scratch/zsa8rk/datasets"
# sim_name (same as wandb id)
# sim_name="ti-main-run-all-0002"
sim_name="ti-test-run-no-grad-clip"
nnodes=1
ngpus_per_node=4
export OMP_NUM_THREADS=1 # (num cpu - num_workers) / num_gpus

# use a checkpoint to continue training with a new config file (learning rate, etc.)
new_training=false
# use the best model for potential restart
best_model=false

# NOTE: set cuda visible devices, MUST be consecutive numbers
# USE ONLY FOR DEBUGGING, non-slurm jobs
# export CUDA_VISIBLE_DEVICES=0,1

######### Multi-Node Setup #########
# rdzv_id=$SLURM_JOB_ID


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
cp "$0" "${sim_dir}/slurm_script.sh"

if [ "$new_training" = true ]; then
    # copy a new config file to the sim_dir and use it as the config file
    config_file="${sim_dir}/$(date +%Y%m%d)_config.yaml"
    cp $base_config_file $config_file
    restart=false
    echo "Using checkpoint to continue training with new config file..."
else
    # Try to find config file in sim_dir
    restart_config_file="${sim_dir}/config.yaml"
    if [ -f "$restart_config_file" ]; then
    echo "Config file found in $sim_dir, attempting restart..."
        # if the config file is found, use it as the config file
        restart=true
        config_file=$restart_config_file
    else
        echo "No config file found in $sim_dir, starting new training..."
        # copy the base config file to sim_dir and use it as the config file
        cp $base_config_file $sim_dir
        config_file="${sim_dir}/config.yaml"
        restart=false
    fi
fi

#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting LPFM training..."
echo "config_file: $config_file"
echo "sim_dir: $sim_dir"
echo "restart: $restart"
echo "new_training: $new_training"
echo "using best model for restart: $best_model"
echo "--------------------------------"

exec_args="--config_file $config_file \
    --sim_name $sim_name \
    --log_dir $log_dir \
    --data_dir $data_dir \
    --time_limit $time_limit"

# Add --restart if the restart flag is true
if [ "$restart" = true ]; then
    exec_args="$exec_args --restart"
fi
if [ "$new_training" = true ]; then
    exec_args="$exec_args --new_training"
fi
if [ "$best_model" = true ]; then
    exec_args="$exec_args --best_model"
fi

# Capture Python output and errors in a variable and run the script
torchrun --standalone --nproc_per_node=$ngpus_per_node $python_exec $exec_args

# move the output file to the sim_dir
mv ${log_dir}/slrm_logs/train_lpfm_${SLURM_JOB_ID}.out $sim_dir