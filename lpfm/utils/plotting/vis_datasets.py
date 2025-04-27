import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lpfm.data.phys_dataset import PhysicsDataset

# Dictionary mapping field names to colormap names
FIELD_COLORS = {
    "pressure": "plasma",
    "density": "cividis",
    "temperature": "hot",
    "velocity_x": "viridis",
    "velocity_y": "viridis",
}

datasets = [
    "cooled_object_pipe_flow_air",
    "heated_object_pipe_flow_air",
    "object_sym_flow_air",
    "object_sym_flow_water",
    "object_periodic_flow_water",
    "cylinder_sym_flow_water",
    "cylinder_pipe_flow_water",
    "euler_multi_quadrants_periodicBC",
    "turbulent_radiative_layer_2D",
    "rayleigh_benard",
    "shear_flow",
    "twophase_flow",
]


def get_dataset_sample(dataset: PhysicsDataset, index: int) -> np.ndarray:
    x, _ = dataset[index]
    return x.numpy()


def sample_to_image(
    sample: np.ndarray,
    save_path: Path,
) -> np.ndarray:
    # sample is a tensor of shape (T, H, W, C)
    # transpose W and H
    sample = sample.transpose(0, 2, 1, 3)

    for i, (field, cmap) in enumerate(FIELD_COLORS.items()):
        vmin = sample[..., i].min()
        vmax = sample[..., i].max()
        for time_step in range(sample.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(sample[time_step, ..., i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")
            fig_path = save_path / f"{field}_{time_step}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()

    return sample


def main():
    base_path = Path("data/datasets")
    split = "train"
    for dataset in datasets:
        print(f"Processing {dataset}")
        dataset_path = base_path / f"{dataset}/data/{split}"
        dataset = PhysicsDataset(
            dataset_path,
            split=split,
            n_steps_input=4,
            dt_stride=10,
            nan_to_zero=False,
        )

        save_path = dataset_path.parents[1] / "images"
        save_path.mkdir(parents=True, exist_ok=True)
        sample = get_dataset_sample(dataset, 0)
        sample_to_image(sample, save_path)


if __name__ == "__main__":
    main()
