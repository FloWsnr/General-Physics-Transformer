#####################################################################################
############################# Setup #################################################
#####################################################################################

# Load modules (Note: This is typically handled differently in Windows)
# module purge
# module load CUDA/12.6.0

# Activate conda environment
$CONDA_ROOT = "C:\ProgramData\miniforge3"
# Initialize conda for PowerShell
& "$CONDA_ROOT\shell\condabin\conda-hook.ps1"
conda activate gphyt

######################################################################################
############################# Set paths ##############################################
######################################################################################
# debug mode
# $debug = $true

# Set up paths
# Set time limit
$time_limit = "24:00:00"
$base_dir = "C:\Users\zsa8rk\Coding\GPhyT"
$python_exec = "$base_dir\gphyt\run\train.py"
$log_dir = "$base_dir\logs"
$data_dir = "$base_dir\data\datasets"
$base_config_file = "$base_dir\gphyt\run\scripts\config.yaml"
$sim_name = "ti-cyl-sym-flow-0001"

# use a checkpoint to continue training with a new config file (learning rate, etc.)
$new_training = $false
# config to use for new training, located in the log dir
$new_config_name = "config_cooldown.yaml"
# name of the checkpoint to use for training. Can be "best_model" or a number of a epoch directory
# if last_checkpoint, the last checkpoint is used
$checkpoint_name = "last_checkpoint"

# sim directory
$sim_dir = Join-Path $log_dir $sim_name

#######################################################################################
############################# Setup sim dir and config file ###########################
#######################################################################################

# delete the sim_dir if it exists and debug is true
if ($debug) {
    if (Test-Path $sim_dir) {
        Remove-Item -Path $sim_dir -Recurse -Force
    }
}

# create the sim_dir if it doesn't exist
if (-not (Test-Path $sim_dir)) {
    New-Item -ItemType Directory -Path $sim_dir | Out-Null
}

# copy the script to the sim_dir
Copy-Item -Path $PSCommandPath -Destination $sim_dir

if ($new_training) {
    $config_file = Join-Path $sim_dir $new_config_name
    $restart = $false
    Write-Host "Using checkpoint to continue training with new config file..."
}
else {
    # Try to find config file in sim_dir
    $restart_config_file = Join-Path $sim_dir "config.yaml"
    if (Test-Path $restart_config_file) {
        Write-Host "Config file found in $sim_dir, attempting restart..."
        $restart = $true
        $config_file = $restart_config_file
    }
    else {
        Write-Host "No config file found in $sim_dir, starting new training..."
        # copy config file to sim_dir
        Copy-Item -Path $base_config_file -Destination $sim_dir
        $config_file = Join-Path $sim_dir "config.yaml"
        $restart = $false
    }
}

#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
Write-Host "--------------------------------"
Write-Host "Starting GPhyT training..."
Write-Host "config_file: $config_file"
Write-Host "sim_dir: $sim_dir"
Write-Host "restart: $restart"
Write-Host "new_training: $new_training"
Write-Host "using checkpoint: $checkpoint_name"
Write-Host "--------------------------------"

# Build the command with proper argument formatting
$exec_args = "--config_file `"$config_file`""
$exec_args += " --sim_name `"$sim_name`""
$exec_args += " --log_dir `"$log_dir`""
$exec_args += " --data_dir `"$data_dir`""
$exec_args += " --checkpoint_name `"$checkpoint_name`""

if ($time_limit) {
    $exec_args += " --time_limit `"$time_limit`""
}

if ($restart) {
    $exec_args += " --restart"
}
if ($new_training) {
    $exec_args += " --new_training"
}

# Run the training script with torchrun for distributed training
$cmd = "python $python_exec $exec_args"
Invoke-Expression $cmd

# Move the output file to the sim_dir (if it exists)
$output_file = Join-Path $log_dir "slrm_logs" "train_gphyt_$PID.out"
if (Test-Path $output_file) {
    Move-Item -Path $output_file -Destination $sim_dir
} 