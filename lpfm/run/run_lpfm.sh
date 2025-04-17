#!/usr/bin/bash

### Task name
#SBATCH --account=rwth1802
#SBATCH --job-name=train_lpfm

### Output file
#SBATCH --output=/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs/slrm_logs/train_lpfm_%j.out


### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
##SBATCH --cpus-per-task=24

### How much memory per core
#SBATCH --mem-per-cpu=5200

### Mail notification configuration
#SBATCH --mail-type=END
#SBATCH --mail-user=zsa8rk@virginia.edu

### Maximum runtime per task
##SBATCH --time=1-00:00:00
#SBATCH --time=00:15:00

### set number of GPUs per task
#SBATCH --gres=gpu:1

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
##SBATCH --array=1-10%1

### Set the time limit for the job, allows for graceful shutdown
### Should be lower than the time limit of the partition
### Format: HH:MM:SS
time_limit="00:15:00"

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
debug=true


# Set up paths
python_exec="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/lpfm/run/train.py"
log_dir="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs"
data_dir="/hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/data/datasets"
# sim_name (same as wandb id)
sim_name="test-run_distributed-02"
# sim directory
sim_dir="${log_dir}/${sim_name}"

# delete the sim_dir if it exists and debug is true
if [ "$debug" = true ]; then
    rm -rf $sim_dir
fi

# create the sim_dir if it doesn't exist
mkdir -p $sim_dir

# copy the slurm script to the sim_dir
cp /hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/lpfm/run/run_lpfm.sh $sim_dir

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

#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting LPFM training..."
echo "config_file: $config_file"
echo "sim_dir: $sim_dir"
echo "restart: $restart"
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

# Capture Python output and errors in a variable and run the script
torchrun --standalone --nproc_per_node=2 $python_exec $exec_args

# move the output file to the sim_dir
mv /hpcwork/rwth1802/coding/Large-Physics-Foundation-Model/logs/slrm_logs/train_lpfm_${SLURM_JOB_ID}.out $sim_dir
