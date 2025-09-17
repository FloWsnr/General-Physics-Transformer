from pathlib import Path
import numpy as np

from gphyt.utils.rollout_video import generate_channel_gif
from gphyt.data.phys_dataset import PhysicsDataset


COLORMAPS = [
    "plasma",
    "cividis",
    "magma",
    "viridis",
]
DATASET_IDX = {
    "cooled_object_pipe_flow_air": (10, 50),
    "cylinder_pipe_flow_water": (43, 50),
    "cylinder_sym_flow_water": (30, 50),
    "euler_multi_quadrants_openBC": (50, 10),
    "euler_multi_quadrants_periodicBC": (50, 10),
    "heated_object_pipe_flow_air": (10, 50),
    "object_periodic_flow_water": (5, 50),
    "object_sym_flow_air": (15, 50),
    "object_sym_flow_water": (5, 50),
    "open_obj_water": (1, 50),
    "rayleigh_benard": (5, 10),
    "rayleigh_benard_obstacle": (5, 50),
    "supersonic_flow": (2, 10),
    "turbulent_radiative_layer_2D": (5, 10),
    "shear_flow": (13, 10),
    "twophase_flow": (23, 10),
}

if __name__ == "__main__":
    data_dir = Path("data/datasets")
    video_dir = Path("results/videos")

    for sub_dir, idx in DATASET_IDX.items():
        test_dir = data_dir / sub_dir / "data/test"
        dataset = PhysicsDataset(
            test_dir, use_normalization=True, full_trajectory_mode=True
        )
        x, y = dataset[idx[0]]
        fps = idx[1]
        data = np.concatenate([x.numpy(), y.numpy()], axis=0)  # (T, H, W, C)

        # combine u,v to magnitude
        u = data[..., -2]
        v = data[..., -1]
        magnitude = np.sqrt(u**2 + v**2)[..., np.newaxis]
        data = np.concatenate([data[..., :-2], magnitude], axis=-1)

        T, H, W, C = data.shape
        for channel, cmap in zip(range(C), COLORMAPS):
            output_path = video_dir / f"{sub_dir}" / f"channel_{channel}.gif"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            c_data = data[..., channel]  # (T, H, W)
            if np.all(c_data == 0):
                print(f"  Skipping channel {channel} for {sub_dir} (all zeros)")
                continue
            generate_channel_gif(
                c_data,
                output_path=output_path,
                fps=fps,
                cmap=cmap,
            )
