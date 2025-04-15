#!/usr/bin/zsh

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

### Partition
#SBATCH --partition=c23ms

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsh8rk@uva.virginia.edu

### Maximum runtime per task
#SBATCH --time=1-00:00:00

### set number of GPUs per task
#SBATCH --gres=gpu:1

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
#SBATCH --array=1-10%1


# Get the actual SLURM output file path
SLURM_OUTPUT="/hpcwork/fw641779/lbm/output_logs/two-phase_lbm_${SLURM_JOB_ID}.out"

# Load modules
module purge
module load CUDA/12.7.0
# Set up paths
palabos_exec="/home/fw641779/Coding/lattice-boltzmann-wetting/mplbm-ut-mirror/src/2-phase_LBM/ShanChen"
config_file="/home/fw641779/Coding/lattice-boltzmann-wetting/lbm_wetting/twophase.yml"
# sim_dir="/hpcwork/fw641779/lbm/Test_Cones"
sim_dir="/hpcwork/fw641779/lbm/Toray-120C/100cov/structure0"
sim_name="sim_run_1"

export CONDA_ROOT=$HOME/miniforge3
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/fw641779/miniforge3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/fw641779/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/fw641779/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/fw641779/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/home/fw641779/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "/home/fw641779/miniforge3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

#####################################################################################
############################# 2-phase prep ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting 2-phase prep..."
echo "config_file: $config_file"
echo "sim_dir: $sim_dir"
echo "sim_name: $sim_name"
echo "--------------------------------"

# Activate environment (will work whether conda was just initialized or was already available)
mamba activate lbm
# Capture Python output and errors in a variable and run the script
python_output=$(python /home/fw641779/Coding/lattice-boltzmann-wetting/lbm_wetting/2_phase_prep.py $config_file $sim_dir $sim_name 2>&1)
# Write both the Python output/errors
echo "$python_output" >> $SLURM_OUTPUT

# Get the return code of the python script and abort if it failed
return_code=$?
if [ $return_code -ne 0 ]; then
    echo "Python script failed with code $return_code. Error messages above." >> $SLURM_OUTPUT
    exit $return_code
else
    # Else continue with running the simulation
    input_file=$sim_dir/$sim_name/input/2_phase_sim_input.xml

    # Start time measurement
    start_time=$(date +%s)

    $MPIEXEC $FLAGS_MPI_BATCH $palabos_exec $input_file

    # Calculate and output elapsed time
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "--------------------------------"
    echo "Simulation completed in $(($elapsed_time / 60)) minutes"
    echo "--------------------------------"
fi

# # Process VTI files
python /home/fw641779/Coding/lattice-boltzmann-wetting/lbm_wetting/utils/postprocessing/vti_processing.py $sim_dir $sim_name

# Calculate saturation
python /home/fw641779/Coding/lattice-boltzmann-wetting/lbm_wetting/utils/postprocessing/post_calc.py -s $sim_dir -n $sim_name

## Append the output file to the simulation directory log
cat /hpcwork/fw641779/lbm/output_logs/two-phase_lbm_$SLURM_JOB_ID.out >> $sim_dir/$sim_name/output/log.out
