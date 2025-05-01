# Set time limit
$time_limit = "24:00:00"

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
conda activate lpfm

######################################################################################
############################# Set paths ##############################################
######################################################################################
# debug mode
# $debug = $true

# Set up paths
$python_exec = "C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\lpfm\run\train.py"
$log_dir = "C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\logs"
$data_dir = "C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\data\datasets"
$config_file = "C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\lpfm\run\config.yaml"
# $config_file = "C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\logs\ti-main-run-single-0004\config_cooldown.yaml"
$sim_name = "ti-cyl-sym-flow-0001"
$new_training_from_checkpoint = $false

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

if ($new_training_from_checkpoint) {
    # overwrite the config file in the sim_dir
    Copy-Item -Path $config_file -Destination $sim_dir
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
        Copy-Item -Path $config_file -Destination $sim_dir
        $restart = $false
    }
}

#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
Write-Host "--------------------------------"
Write-Host "Starting LPFM training..."
Write-Host "config_file: $config_file"
Write-Host "sim_dir: $sim_dir"
Write-Host "restart: $restart"
Write-Host "new_training_from_checkpoint: $new_training_from_checkpoint"
Write-Host "--------------------------------"

# Build the command with proper argument formatting
$cmd = "python $python_exec"
$cmd += " --config_file `"$config_file`""
$cmd += " --sim_name `"$sim_name`""
$cmd += " --log_dir `"$log_dir`""
$cmd += " --data_dir `"$data_dir`""
$cmd += " --time_limit `"$time_limit`""

if ($new_training_from_checkpoint) {
    $cmd += " --new_training_from_checkpoint"
}

# Add --restart if the restart flag is true
if ($restart) {
    $cmd += " --restart"
}

# Run the training script
Invoke-Expression $cmd

# Move the output file to the sim_dir (if it exists)
$output_file = Join-Path $log_dir "slrm_logs" "train_lpfm_$PID.out"
if (Test-Path $output_file) {
    Move-Item -Path $output_file -Destination $sim_dir
} 