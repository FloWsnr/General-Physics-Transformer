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
$sim_name = "ti-main-run-single-0004"
$nnodes = 1
$ngpus_per_node = 1
$env:OMP_NUM_THREADS = 1 # (num cpu - num_workers) / num_gpus

# NOTE: set cuda visible devices, MUST be consecutive numbers
# USE ONLY FOR DEBUGGING, non-slurm jobs
# $env:CUDA_VISIBLE_DEVICES = "1"

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

# Try to find config file in sim_dir
$config_file = Join-Path $sim_dir "config.yaml"
if (Test-Path $config_file) {
    Write-Host "Config file found in $sim_dir, attempting restart..."
    $restart = $true
} else {
    Write-Host "No config file found in $sim_dir, starting new training..."
    # copy config file to sim_dir
    Copy-Item -Path "C:\Users\zsa8rk\Coding\Large-Physics-Foundation-Model\lpfm\run\config.yaml" -Destination $sim_dir
    $restart = $false
}

#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
Write-Host "--------------------------------"
Write-Host "Starting LPFM training..."
Write-Host "config_file: $config_file"
Write-Host "sim_dir: $sim_dir"
Write-Host "restart: $restart"
Write-Host "--------------------------------"

# Build the command with proper argument formatting
$cmd = "python $python_exec"
$cmd += " --config_file `"$config_file`""
$cmd += " --sim_name `"$sim_name`""
$cmd += " --log_dir `"$log_dir`""
$cmd += " --data_dir `"$data_dir`""
$cmd += " --time_limit `"$time_limit`""

# Add --restart if the restart flag is true
if ($restart) {
    $cmd += " --restart"
}

# Run the training script
Invoke-Expression $cmd 