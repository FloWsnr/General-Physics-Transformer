# Large Physics Foundation Model for Computational Fluid Dynamics

This repository contains the code for the Large Physics Foundation Model (LPFM) for Computational Fluid Dynamics.
The LPFM is a transformer-based foundation model capable of learning many different systems of CFD equations at once.


## Introduction



## Installation

We are using conda and pip to manage the dependencies.

```bash
conda create -n lpfm python=3.12
conda activate lpfm
pip install einops h5py imageio ipykernel matplotlib neuraloperator pandas the-well torch torchvision tqdm dadaptation wandb dotenv prodigyopt
pip install -e .
```


## Datasets

This study uses both self-made datasets and datasets from [The Well](https://polymathic-ai.org/the_well/).
The well-datasets can be downloaded like this:

```bash
the-well-download --base-path /home/lpfm --dataset turbulent_radiative_layer_2D
```

All datasets are formatted according to the-well [format](https://polymathic-ai.org/the_well/data_format/).
In general, the data are hdf5 files, one for each parameter set.
Inside the hdf5 files, the features are stored in the t0 (scalar), t1 (vector), and t2 (tensor) groups.
The arrays are shaped as (n_trajectories, n_steps, x, y) or for vector features as (n_trajectories, n_steps, x, y, 2).