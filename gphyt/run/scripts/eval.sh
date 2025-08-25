#!/usr/bin/bash

### Task name
#SBATCH --account=rwth1802
#SBATCH --job-name=eval_gphyt

### Output file
#SBATCH --output=/hpcwork/rwth1802/coding/General-Physics-Transformer/results/slrm_logs/eval_gphyt_%j.out

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


#####################################################################################
############################# Setup #################################################
#####################################################################################

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

python_bin="/home/fw641779/miniforge3/envs/gphyt/bin/python"
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
# Try to find config file in sim_dir
config_file="${sim_dir}/config_eval.yaml"
if [ ! -f "$config_file" ]; then
    echo "No config_eval.yaml file found in $sim_dir, aborting..."
    exit 1
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
$python_bin $python_exec $exec_args

# move the output file to the sim_dir
mv ${log_dir}/slrm_logs/eval_${sim_name}_${SLURM_JOB_ID}.out $sim_dir