#!/usr/bin/bash

### Task name
#SBATCH --account=rwth1802
#SBATCH --job-name=train_lpfm

### Output file
#SBATCH --output=/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs/slrm_logs/train_lpfm_%j.out


### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive

### How much memory per core
#SBATCH --mem-per-cpu=5200

### Mail notification configuration
#SBATCH --mail-type=NONE
#SBATCH --mail-user=zsa8rk@virginia.edu

### Maximum runtime per task
#SBATCH --time=24:00:00
##SBATCH --time=00:30:00

### set number of GPUs per task
#SBATCH --gres=gpu:4

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
##SBATCH --array=1-10%1

### Set the time limit for the job, allows for graceful shutdown
### Should be lower than the time limit of the partition
### Format: HH:MM:SS
time_limit="24:00:00"

#####################################################################################
############################# Setup #################################################
#####################################################################################

# Load modules
module purge
module load CUDA/12.6.0

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
python_exec="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/lpfm/run/train.py"
log_dir="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs"
data_dir="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/data/datasets"
config_file="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/lpfm/run/config.yaml"
# sim_name (same as wandb id)
# sim_name="ti-main-run-all-0002"
sim_name="ti-main-run-all-0002"
nnodes=1
ngpus_per_node=4
export OMP_NUM_THREADS=1 # (num cpu - num_workers) / num_gpus

# use a checkpoint to continue training with a new config file (learning rate, etc.)
new_training_from_checkpoint=true

# NOTE: set cuda visible devices, MUST be consecutive numbers
# USE ONLY FOR DEBUGGING, non-slurm jobs
# export CUDA_VISIBLE_DEVICES=1

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

# copy the slurm script to the sim_dir
cp /hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/lpfm/run/run_lpfm.sh $sim_dir

if [ "$new_training_from_checkpoint" = true ]; then
    # overwrite the config file in the sim_dir
    cp $config_file $sim_dir
    restart=false
    echo "Using checkpoint to continue training with new config file..."
else
    # Try to find config file in sim_dir
    config_file="${sim_dir}/config.yaml"
    if [ -f "$config_file" ]; then
    echo "Config file found in $sim_dir, attempting restart..."
        restart=true
    else
        echo "No config file found in $sim_dir, starting new training..."
        # copy config file to sim_dir
        cp $config_file $sim_dir
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
echo "new_training_from_checkpoint: $new_training_from_checkpoint"
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
if [ "$new_training_from_checkpoint" = true ]; then
    exec_args="$exec_args --new_training_from_checkpoint"
fi

# Capture Python output and errors in a variable and run the script
torchrun --standalone --nproc_per_node=$ngpus_per_node $python_exec $exec_args

# move the output file to the sim_dir
mv /hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs/slrm_logs/train_lpfm_${SLURM_JOB_ID}.out $sim_dir
