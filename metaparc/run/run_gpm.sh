#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --job-name=gpm  # Sets the job name
#SBATCH --time=00:15:00         # Run time of 15 minutes
#SBATCH --ntasks=8         
#SBATCH --output=stdout.txt
#SBATCH --account=rwth1802
#SBATCH --gres=gpu:1 # number of GPUs

### Program Code
srun hostname
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate myenv

python train.py --config config.yaml