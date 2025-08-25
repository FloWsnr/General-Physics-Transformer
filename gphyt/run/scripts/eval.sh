#!/usr/bin/bash

### Task name
#SBATCH --account=rwth1802
#SBATCH --job-name=eval_gphyt

### Output file
#SBATCH --output=/hpcwork/rwth1802/coding/GPhyT/results/slrm_logs/eval_gphyt_%j.out

### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=23
#SBATCH --exclusive

### Mail notification configuration
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zsa8rk@virginia.edu

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task
#SBATCH --gres=gpu:1

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
conda activate gphyt

######################################################################################
############################# Set paths ##############################################
######################################################################################
# debug mode
# debug=true
# Set up paths
base_dir="/hpcwork/rwth1802/coding/General-Physics-Transformer"
python_exec="${base_dir}/gphyt/run/model_eval.py"
log_dir="${base_dir}/results"
data_dir="${base_dir}/data/datasets"
base_config_file="${base_dir}/gphyt/run/scripts/config.yaml"
# sim_name (same as wandb id)
sim_name="ti-main-run-all-0007"
nnodes=1
ngpus_per_node=1
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
echo "Starting GPhyT evaluation..."
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