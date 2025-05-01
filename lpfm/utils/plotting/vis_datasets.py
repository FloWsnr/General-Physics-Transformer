import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from lpfm.data.phys_dataset import PhysicsDataset

# Dictionary mapping field names to colormap names
FIELD_COLORS = OrderedDict(
    [
        ("pressure", "plasma"),
        ("density", "cividis"),
        ("temperature", "hot"),
        ("velocity_x", "viridis"),
        ("velocity_y", "viridis"),
        ("velocity_mag", "viridis"),
    ]
)

datasets = [
    # "cooled_object_pipe_flow_air",
    "heated_object_pipe_flow_air",
    # "object_sym_flow_air",
    # "object_sym_flow_water",
    # "object_periodic_flow_water",
    # "cylinder_sym_flow_water",
    # "cylinder_pipe_flow_water",
    # "euler_multi_quadrants_periodicBC",
    # "turbulent_radiative_layer_2D",
    # "rayleigh_benard",
    # "shear_flow",
    # "twophase_flow",
    # "rayleigh_benard_obstacle",
    # "acoustic_scattering_inclusions",
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
    vel_mag = np.linalg.norm(sample[..., -2:], axis=-1)
    sample = np.concatenate([sample, vel_mag[..., None]], axis=-1)

    for i, (field, cmap) in enumerate(FIELD_COLORS.items()):
        vmin = np.nanmin(sample[..., i])
        vmax = np.nanmax(sample[..., i])
        print(f"Field: {field}, vmin: {vmin}, vmax: {vmax}")
        for time_step in range(sample.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(sample[time_step, ..., i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")
            # Remove all padding and margins
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            fig.tight_layout(pad=0)
            fig_path = save_path / f"{field}_{time_step}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close()

    return sample


include_field_names = {
        "t0_fields": ["pressure", "density", "temperature"],
        "t1_fields": ["velocity"],
    }

def main():
    base_path = Path("data/datasets")
    split = "train"
    for dataset in datasets:
        print(f"Processing {dataset}")
        dataset_path = base_path / f"{dataset}/data/{split}"
        if not dataset_path.exists():
            print(f"Skipping {dataset} because it does not exist")
            continue
        dataset = PhysicsDataset(
            dataset_path,
            split=split,
            n_steps_input=4,
            dt_stride=10,
            nan_to_zero=False,
            include_field_names=include_field_names,
        )

        save_path = dataset_path.parents[1] / "images"
        save_path.mkdir(parents=True, exist_ok=True)
        sample = get_dataset_sample(dataset, 10050)
        sample_to_image(sample, save_path)


if __name__ == "__main__":
    main()
