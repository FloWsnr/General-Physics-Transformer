# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPhyT (General Physics Transformer) is a physics-aware transformer model trained to predict over 11 different physics systems using in-context learning. The model uses a differentiator-integrator approach similar to numerical solvers for accurate long-term predictions.

## Development Environment Setup

```bash
conda create -n gphyt python=3.12
conda activate gphyt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install einops h5py imageio ipykernel matplotlib neuraloperator pandas the-well wandb dotenv torchtnt pytest
pip install -e .
```

## Key Commands

### Training
- **Local training**: `python gphyt/run/train.py --config_file gphyt/run/scripts/config.yaml --sim_name <name> --log_dir logs --data_dir data/datasets`
- **SLURM training**: `sbatch gphyt/run/scripts/train.sh` (for HPC clusters)
- **Distributed training**: Uses `torchrun --standalone --nproc_per_node=<gpus> gphyt/run/train.py`

### Evaluation
- **Model evaluation**: `python gphyt/run/model_eval.py --config_file <config> --sim_name <name> --log_dir logs --data_dir data/datasets --checkpoint_name best_model`
- **SLURM evaluation**: `sbatch gphyt/run/scripts/eval.sh`
- **Rollout evaluation**: `python gphyt/run/rollout_eval.py`

### Testing
- **Run all tests**: `pytest`
- **Run specific test module**: `pytest tests/test_model/test_transformer/`
- **Run with coverage**: `pytest --cov=gphyt`

### Data Management
- **Download Well datasets**: `the-well-download --base-path /path/to/data --dataset <dataset_name>`
- **Preprocess data**: `python gphyt/data/preprocess.py`
- **Analyze datasets**: `python gphyt/data/analyze_data.py`

## Architecture Overview

### Core Components
- **`gphyt/model/transformer/`**: Main transformer architecture with physics-aware attention mechanisms
  - `model.py`: Main GPT-style transformer model
  - `attention.py`: Multi-head attention with physics constraints
  - `ax_attention.py`: Axial attention for spatial dimensions
  - `derivatives.py`: Differentiable derivative operations
  - `num_integration.py`: Numerical integration methods (Euler, RK4, Heun)
  - `pos_encodings.py`: Positional encodings (absolute, RoPE)

- **`gphyt/model/tokenizer/`**: Converts between spatial fields and transformer tokens
  - `tokenizer.py`: Main tokenization logic (conv_net, linear, vqvae modes)
  - `tokenizer_utils.py`: Utility functions for tokenization

- **`gphyt/data/`**: Data loading and preprocessing
  - `phys_dataset.py`: Main physics dataset class
  - `well_dataset.py`: Interface for The Well datasets
  - `dataset_utils.py`: Data loading utilities and batch creation

- **`gphyt/run/`**: Training and evaluation scripts
  - `train.py`: Main training loop with DDP support
  - `model_eval.py`: Model evaluation and metrics
  - `lr_scheduler.py`: Learning rate scheduling (linear warmup + cosine annealing)

### Data Format
- Uses HDF5 files following The Well format specification
- Fields organized as: `t0_fields` (scalars), `t1_fields` (vectors), `t2_fields` (tensors)
- Data shape: `(n_trajectories, n_steps, x, y)` or `(n_trajectories, n_steps, x, y, 2)` for vectors
- Physics systems include: Navier-Stokes, heat transfer, two-phase flow, acoustic scattering, natural convection

### Model Configuration
- Model sizes: GPT_S, GPT_M, GPT_L, GPT_XL (defined in `model_specs.py`)
- Key config parameters in `gphyt/run/scripts/config.yaml`:
  - `model.transformer.model_size`: Model architecture size
  - `model.transformer.use_derivatives`: Enable physics-aware derivatives
  - `model.transformer.integrator`: Numerical integration method
  - `training.batch_size`: Per-GPU batch size
  - `data.n_steps_input`: Number of input timesteps for in-context learning

### Physics-Aware Features
- **Derivatives**: Automatic differentiation for computing spatial/temporal derivatives
- **Integration**: Multiple numerical integration schemes for time-stepping
- **In-context learning**: Model learns physics from initial timesteps without retraining
- **Multi-physics**: Trained on 11+ different physics systems simultaneously

## Development Patterns

### Adding New Physics Systems
1. Create dataset in The Well format (HDF5)
2. Add dataset name to `config.yaml` datasets list
3. Ensure proper field naming (pressure, density, temperature, velocity components)
4. Test with existing data loading pipeline

### Model Development
- New attention mechanisms go in `gphyt/model/transformer/`
- Follow existing patterns for registration in `model.py`
- Add corresponding tests in `tests/test_model/test_transformer/`
- Use `torch.compile` for optimization (enabled by default)

### Distributed Training
- Uses PyTorch DDP for multi-GPU training
- SLURM scripts configured for HPC clusters (RWTH Aachen)
- Automatic checkpointing and resuming supported
- WandB integration for experiment tracking

## Important File Locations
- Main config: `gphyt/run/scripts/config.yaml`
- SLURM scripts: `gphyt/run/scripts/*.sh`
- Test fixtures: `tests/conftest.py`
- Model specifications: `gphyt/model/model_specs.py`
- Training logs and checkpoints: `results/<sim_name>/`

## Best Practices and Reminders
- **Environment Management**:
  - Use the correct conda env when running python or pytest (/home/fw641779/minifor
ge3/envs/gphyt/bin/python)
  - Activate environment with `conda activate gphyt`