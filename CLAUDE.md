# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

General Physics Transformer (GPhyT) — a foundation model for physics simulation that combines a transformer-based neural differentiator with numerical integration. Trains on HDF5 datasets (The Well format) spanning diverse physical systems (Navier-Stokes, heat transfer, two-phase flow, etc.).

Paper: https://arxiv.org/abs/2509.13805

## Build & Install

```bash
# Uses uv with hatchling build backend. PyTorch from cu129 index.
pip install -e ".[dev]"
```

Requires Python >= 3.13.

## Running Tests

```bash
# All tests
pytest tests/

# Single test file
pytest tests/test_models/test_transformer/test_model.py

# Single test
pytest tests/test_models/test_transformer/test_model.py::TestClassName::test_method -v
```

Known: ~21 data tests fail because the dummy HDF5 fixtures in `tests/conftest.py` use generic field names (e.g. "variable_field1") instead of physics field names like "pressure". These are pre-existing failures.

## Training & Evaluation

```bash
# Single-GPU training
python gphyt/train/run_training.py --config_path <path_to_config.yaml>

# Multi-GPU with torchrun
torchrun --standalone --nproc_per_node=N gphyt/train/run_training.py --config_path <config>

# Evaluation
python gphyt/train/model_eval.py --config_file <config> --sim_name <name> --log_dir <dir> --data_dir <dir> --checkpoint_name <best_model|epoch_num> --forecast_horizons 1 4 8
```

SLURM scripts in `gphyt/train/scripts/` (train_riv.sh, eval.sh).

## Architecture

### 3-Layer Structure (ML-Training-Suite pattern)

```
gphyt/
├── data/          # Data loading & preprocessing
├── models/        # Model definitions
└── train/         # Training loop, evaluation, utilities
    ├── scripts/   # Shell scripts (SLURM launchers)
    └── utils/     # Checkpointing, logging, LR scheduling, visualization
```

### Key Entry Points & Factory Functions

- **`gphyt/models/model_utils.py`** — `get_model(model_config)`: dispatches to `gphyt`, `unet`, or `fno` based on `architecture` key
- **`gphyt/data/dataset.py`** — `get_dataset(config, split)`: builds `SuperDataset` of `PhysicsDataset`s
- **`gphyt/train/run_training.py`** — CLI entry point; parses YAML config, builds model/data/optimizer, creates `Trainer`

### Model Architectures

- **GPhyT** (`gphyt/models/transformer/`): Tokenizer → Transformer (attention blocks with RoPE or absolute pos encoding) → Finite-difference derivative estimation → Numerical integration (Euler/RK4/Heun) → Detokenizer. Size variants: GPT_S, GPT_M, GPT_L, GPT_XL (in `model_specs.py`).
- **FNO** (`gphyt/models/fno.py`): Fourier Neural Operator wrapper using `neuraloperator` library's `FNO` class with `n_modes=(t,h,w)` tuple.
- **UNet** (`gphyt/models/unet.py`): Convolutional U-Net. Variants: UNet_S, UNet_M.

### Data Pipeline

- **`PhysicsDataset`** (`gphyt/data/phys_dataset.py`): wraps `WellDataset` (HDF5 loader), handles input/output windowing (`n_steps_input`, `n_steps_output`), z-score normalization, spatial resizing, dt_stride, and data augmentation (flips).
- **`SuperDataset`**: concatenates multiple `PhysicsDataset`s, supports `max_samples` per dataset.
- Data format: HDF5 per The Well spec — fields stored in `t0_fields/`, `t1_fields/`, `t2_fields/` groups with shape `(n_trajectories, n_steps, x, y [, n_dim])`.

### Training

- **`Trainer`** (`gphyt/train/train_base.py`): full training loop with DDP support, AMP (bfloat16/float16), gradient checkpointing via `mem_budget`, autoregressive training (`n_ar_steps`), W&B logging.
- **`Evaluator`** (`gphyt/train/eval.py`): validation loop supporting AR rollout evaluation.
- **`model_eval.py`**: standalone detailed evaluation with per-dataset, per-horizon metrics and visualization.

### Config Format

Nested YAML with top-level keys: `wandb`, `logging`, `model`, `training`, `data`. See `gphyt/train/train.yml` for the reference config. Also supports a flat format for backward compatibility.

### Loss Functions

- Generic: MSE, MAE, RMSE, NRMSE, VRMSE in `gphyt/models/loss_fns.py`
- GPhyT-specific: NMSELoss, VMSELoss, RNMSELoss, RVMSELoss in `gphyt/models/transformer/loss_fns.py` (support `return_scalar` and dimension selection)
