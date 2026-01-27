#!/usr/bin/bash

### Task name
#SBATCH --account=sds_baek_energetic
#SBATCH --job-name=eval-m-main-ar4

### Output file
#SBATCH --output=results/slrm_logs/eval-m-main-ar4_%j.out

### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=18

### How much memory in total (MB)
#SBATCH --mem=300G


### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email@example.com

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task (v100, a100, h200)
##SBATCH --gres=gpu:a6000:2
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb

### Partition
#SBATCH --partition=gpu


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
base_dir="/path/to/General-Physics-Transformer"

python_bin="/path/to/miniforge3/envs/gphyt/bin/python"
python_exec="${base_dir}/gphyt/run/model_eval.py"
log_dir="${base_dir}/results"
data_dir="/path/to/datasets"
base_config_file="${base_dir}/gphyt/run/scripts/config.yaml"
sim_name="m-main-ar4-01"
# forcasts
forecast="1 4 8 12 16 20 24"
# subdir name
sub_dir="eval/all_horizons"
debug=false


nnodes=1
ngpus_per_node=1
export OMP_NUM_THREADS=1

# name of the checkpoint to use for evaluation. Can be "best_model" or a number of a epoch directory
checkpoint_name="best_model"

# sim directory
sim_dir="${log_dir}/${sim_name}"



#######################################################################################
############################# Setup sim dir and config file ###########################
#######################################################################################

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
    --checkpoint_name $checkpoint_name \
    --forecast_horizons $forecast \
    --subdir_name $sub_dir"

if [ "$debug" = true ]; then
    echo "Running in debug mode."
    exec_args ="$exec_args --debug"
fi

# Capture Python output and errors in a variable and run the script
$python_bin $python_exec $exec_args

# move the output file to the sim_dir
mv ${log_dir}/slrm_logs/eval_${sim_name}_${SLURM_JOB_ID}.out $sim_dir