import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict, namedtuple
from gphyt.data.phys_dataset import PhysicsDataset

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

DatasetInfo = namedtuple("DatasetInfo", ["name", "traj_idx"])

DATASETS = [
    DatasetInfo("cooled_object_pipe_flow_air", 10),
    DatasetInfo("heated_object_pipe_flow_air", 10),
    DatasetInfo("object_sym_flow_air", 10),
    DatasetInfo("object_sym_flow_water", 10),
    DatasetInfo("object_periodic_flow_water", 10),
    DatasetInfo("cylinder_sym_flow_water", 10),
    DatasetInfo("cylinder_pipe_flow_water", 10),
    DatasetInfo("euler_multi_quadrants_periodicBC", 10),
    DatasetInfo("turbulent_radiative_layer_2D", 10),
    DatasetInfo("rayleigh_benard", 10),
    DatasetInfo("shear_flow", 10),
    DatasetInfo("twophase_flow", 10),
    DatasetInfo("rayleigh_benard_obstacle", 10),
    DatasetInfo("acoustic_scattering_inclusions", 10),
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
    for dataset, traj_idx in DATASETS:
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
        sample = get_dataset_sample(dataset, traj_idx)
        sample_to_image(sample, save_path)


if __name__ == "__main__":
    main()
