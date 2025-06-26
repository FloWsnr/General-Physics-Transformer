# PowerShell script for running GPhyT evaluation

#####################################################################################
############################# Setup #################################################
#####################################################################################

# Activate conda environment
$CONDA_ROOT = "C:\ProgramData\miniforge3"
# Initialize conda for PowerShell
& "$CONDA_ROOT\shell\condabin\conda-hook.ps1"
conda activate gphyt

######################################################################################
############################# Set paths ##############################################
######################################################################################
# Set up paths
$base_dir = "C:\Users\zsa8rk\Coding\General-Physics-Transformer"
$python_exec = Join-Path $base_dir "gphyt\run\model_eval.py"
$log_dir = Join-Path $base_dir "logs"
$data_dir = Join-Path $base_dir "data\datasets"
$base_config_file = Join-Path $base_dir "gphyt\run\scripts\config.yaml"

# sim_name (same as wandb id)
$sim_name = ""
# name of the checkpoint to use for evaluation. Can be "best_model" or a number of a epoch directory
$checkpoint_name = "best_model"
 
# sim directory
$sim_dir = Join-Path $log_dir $sim_name

#######################################################################################
############################# Setup sim dir and config file ###########################
#######################################################################################

# create the sim_dir if it doesn't exist
if (-not (Test-Path $sim_dir)) {
    New-Item -ItemType Directory -Path $sim_dir | Out-Null
}

# copy the script to the sim_dir with .ps1 suffix
Copy-Item $MyInvocation.MyCommand.Path (Join-Path $sim_dir "eval_script.ps1")

# Try to find config file in sim_dir
$config_file = Join-Path $sim_dir "config.yaml"
if (-not (Test-Path $config_file)) {
    Write-Host "No config file found in $sim_dir, copying base config..."
    Copy-Item $base_config_file $sim_dir
}

#####################################################################################
############################# Evaluation ############################################
#####################################################################################
Write-Host "--------------------------------"
Write-Host "Starting GPhyT evaluation..."
Write-Host "config_file: $config_file"
Write-Host "sim_dir: $sim_dir"
Write-Host "using checkpoint: $checkpoint_name"
Write-Host "--------------------------------"

# Build the arguments for the Python script
$exec_args = @(
    "--config_file", $config_file,
    "--sim_name", $sim_name,
    "--log_dir", $log_dir,
    "--data_dir", $data_dir,
    "--checkpoint_name", $checkpoint_name
)

$cmd = "python $python_exec $exec_args"
Invoke-Expression $cmd