import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from lpfm.data.dataset_utils import get_datasets


base_path = Path("/scratch/zsa8rk/datasets")
plots_dir = base_path / "velocity_plots"
plots_dir.mkdir(exist_ok=True)  # Create plots directory if it doesn't exist

datasets = [
    # name, dt_stride, background channel
    ("cooled_object_pipe_flow_air", 8, 2),
    ("heated_object_pipe_flow_air", 8, 2),
    ("object_sym_flow_air", 8, 0),
    ("object_sym_flow_water", 8, 0),
    ("object_periodic_flow_water", 8, 0),
    ("cylinder_sym_flow_water", 8, 0),
    ("cylinder_pipe_flow_water", 8, 0),
    ("euler_multi_quadrants_periodicBC", 1, 0),
    ("turbulent_radiative_layer_2D", 1, 2),
    ("rayleigh_benard", 1, 2),
    ("shear_flow", 1, 0),
    ("twophase_flow", 2, 1),
    ("acoustic_scattering_inclusions", 1, 0),
    ("rayleigh_benard_obstacle", 8, 2),
]

for dataset_name, dt_stride, background_channel in datasets:
    print(f"Plotting {dataset_name}")
    data_config = {
        "datasets": [dataset_name],
        "data_dir": base_path,
        "n_steps_input": 1,
        "n_steps_output": 1,
        "dt_stride": dt_stride,
        "out_shape": [256, 128],
        "max_samples_per_ds": None,
        "max_rollout_steps": 1000,
        "full_trajectory_mode": True,
        "zero_field_value": 0.0,
        "nan_to_zero": False,
    }

    loaded_datasets = get_datasets(data_config, split="train")
    dataset = loaded_datasets[dataset_name]

    # Get a sample from the dataset (t, y, x, c)
    x, full_trajectory = dataset[5]

    # Create coordinate grids
    y, x = np.meshgrid(
        np.arange(full_trajectory.shape[2]), np.arange(full_trajectory.shape[1])
    )
    skip = 4  # Adjust this value to show more or fewer arrows

    # Create the figure and initial plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Initial background plot
    background = np.swapaxes(full_trajectory[0, ..., background_channel], 0, 1)
    im = ax.imshow(background, cmap="viridis", origin="lower", alpha=0.7)
    plt.colorbar(im, label="Background Field")

    # Initial quiver plot
    vel_x = full_trajectory[0, ..., 3]
    vel_y = full_trajectory[0, ..., 4]
    quiver = ax.quiver(
        x[::skip, ::skip],
        y[::skip, ::skip],
        vel_x[::skip, ::skip],
        vel_y[::skip, ::skip],
        scale=50,
        color="white",
        alpha=0.8,
    )

    ax.set_title(f"Velocity Field with Background - {dataset_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")

    def update(frame):
        # Update background
        background = np.swapaxes(full_trajectory[frame, ..., background_channel], 0, 1)
        im.set_array(background)

        # Update velocity vectors
        vel_x = full_trajectory[frame, ..., 3]
        vel_y = full_trajectory[frame, ..., 4]
        quiver.set_UVC(vel_x[::skip, ::skip], vel_y[::skip, ::skip])

        return im, quiver

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=full_trajectory.shape[0],
        interval=100,  # 100ms between frames
        blit=True,
    )

    # Save as GIF
    anim.save(
        plots_dir / f"{dataset_name}_velocity_field.gif",
        writer="pillow",
        fps=5,  # 10 frames per second
        dpi=300,
    )
    plt.close()
