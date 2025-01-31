# MetaPARC
Meta-Learning Model for Physics-based Neural Nets

## Introduction



## Installation

```bash
mamba create -n parc python=3.12
mamba activate parc
mamba install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
mamba install numpy matplotlib tqdm ipykernel imageio pandas h5py
pip install the_well
pip install -e .
```


## Datasets

This study uses both self-made datasets and datasets from [The Well](https://polymathic-ai.org/the_well/).
The well-datasets can be downloaded like this:

```bash
the-well-download --base-path /home/flwi01/Coding/MetaPARC/data/tasks --dataset turbulent_radiative_layer_2D
```
