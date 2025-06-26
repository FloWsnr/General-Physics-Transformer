# Towards a General Physics Transformer
## Introduction

This repository contains the code for the paper "Towards a General Physics Transformer". The GPhyT is a physics-aware transformer model trained to predict over
11 different physics systems. Specifically, the model leverages "in-context learning", i.e. learning from a set of initial time steps to deduce the physics of the system.
The makes the model capable of learning new physics systems without any additional pretraining and without any user input.


## Installation

We are using conda and pip to manage the dependencies.

```bash
conda create -n gphyt python=3.12
conda activate gphyt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install einops h5py imageio ipykernel matplotlib neuraloperator pandas the-well tqdm dadaptation wandb dotenv prodigyopt torchtnt
pip install -e .
```


## Datasets

This study uses both self-made datasets and datasets from [The Well](https://polymathic-ai.org/the_well/).
The well-datasets can be downloaded like this:

```bash
the-well-download --base-path /home/gphyt --dataset turbulent_radiative_layer_2D
```

All datasets are formatted according to the-well [format](https://polymathic-ai.org/the_well/data_format/).
In general, the data are hdf5 files, one for each parameter set.
Inside the hdf5 files, the features are stored in the t0 (scalar), t1 (vector), and t2 (tensor) groups.
The arrays are shaped as (n_trajectories, n_steps, x, y) or for vector features as (n_trajectories, n_steps, x, y, 2).

### Physics

The datasets cover the following physics:

- Incompressible Navier-Stokes
- Compressible Navier-Stokes
- Flow with heat transfer
- Obstacles and wall interactions
- Two-phase flow
- Acoustic scattering
- Natural convection