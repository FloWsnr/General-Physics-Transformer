from pathlib import Path
import numpy as np
import torch

from gphyt.model.transformer.model import get_model
from gphyt.model.fno import get_model as get_fno_model
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

model_config = {
    "img_size": (4, 256, 128),
    "tokenizer": {
        "detokenizer_mode": "linear",
        "detokenizer_overlap": 0,
        "tokenizer_mode": "linear",
        "tokenizer_overlap": 0,
    },
    "transformer": {
        "att_mode": "full",
        "dropout": 0.0,
        "input_channels": 5,
        "integrator": "Euler",
        "model_size": "GPT_XL",
        "patch_size": [1, 16, 16],
        "pos_enc_mode": "absolute",
        "stochastic_depth_rate": 0.0,
        "use_derivatives": True,
    },
}

# fno_config = {
#     "model_size": "FNO_M",
# }


@torch.inference_mode()
def _rollout(
    model: torch.nn.Module,
    device: torch.device,
    x: torch.Tensor,
    y: torch.Tensor,
    rollout: bool = False,
    amp: bool = True,
) -> torch.Tensor:
    """Rollout the model on a trajectory.

    Parameters
    ----------
    model : torch.nn.Module
        The model to rollout
    device : torch.device
        The device to use
    x : torch.Tensor
        Input tensor of shape (T_in, H, W, C)
    y : torch.Tensor
        Ground truth tensor of shape (T_out, H, W, C)

    rollout : bool, optional
        Whether to rollout the full trajectory, by default False

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
        Tuple containing the predicted outputs,
        the ground truth, and the losses dict for each criterion at each timestep
    """

    input = x.to(device)
    full_traj = y.to(device)

    # add batch dimension
    input = input.unsqueeze(0)
    full_traj = full_traj.unsqueeze(0)

    B, T, H, W, C = full_traj.shape
    num_timesteps = T

    outputs = []
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp,
    ):
        for i in range(num_timesteps):
            # Predict next timestep
            output = model(input)  # (B, 1T, H, W, C)
            # if the output is nan, stop the rollout
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"Output is NaN or Inf at timestep {i}, stopping rollout.")
                break

            outputs.append(output.clone())
            # Update input
            if rollout:
                input = torch.cat([input[:, 1:, ...], output], dim=1)
            else:
                input = torch.cat(
                    [input[:, 1:, ...], full_traj[:, i, ...].unsqueeze(1)], dim=1
                )

    # remove batch dimension
    outputs = torch.cat(outputs, dim=1)
    outputs = outputs.squeeze(0)
    # Return predictions and ground truth (excluding first timestep)
    return outputs


def clean_cp(cp: dict) -> dict:
    clean_state_dict = {}
    for key, value in cp["model_state_dict"].items():
        # Check if the key starts with 'module._orig_mod.'
        if "module." in key:
            key = key.replace("module.", "")
        if "_orig_mod." in key:
            key = key.replace("_orig_mod.", "")
        # Keep the key as is
        clean_state_dict[key] = value

    clean_state_dict.pop("_metadata", None)
    cp["model_state_dict"] = clean_state_dict
    return cp


if __name__ == "__main__":
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    data_dir = Path("data/datasets")
    video_dir = Path("results/videos/xl-main-ft")
    checkpoint_dir = Path("results") / "xl-main-ft-02"

    model = get_model(model_config)
    # model = get_fno_model(fno_config)
    cp = torch.load(
        checkpoint_dir / "best_model.pth", map_location=device, weights_only=False
    )
    cp = clean_cp(cp)
    model.load_state_dict(cp["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    for sub_dir, idx in DATASET_IDX.items():
        test_dir = data_dir / sub_dir / "data/test"
        dataset = PhysicsDataset(
            test_dir,
            n_steps_input=4,
            n_steps_output=1,
            use_normalization=True,
            full_trajectory_mode=True,
        )
        x, y = dataset[idx[0]]

        print(f"Rolling out for dataset {sub_dir}")
        predictions = (
            _rollout(model=model, device=device, x=x, y=y, rollout=True, amp=True)
            .cpu()
            .numpy()
        )

        gt = np.concatenate([x.numpy(), y.numpy()], axis=0)  # (T, H, W, C)
        pred = np.concatenate([x.numpy(), predictions], axis=0)  # (T, H, W, C)

        # combine u,v to magnitude
        u = gt[..., -2]
        v = gt[..., -1]
        magnitude = np.sqrt(u**2 + v**2)[..., np.newaxis]
        gt = np.concatenate([gt[..., :-2], magnitude], axis=-1)

        # same for prediction
        u = pred[..., -2]
        v = pred[..., -1]
        magnitude = np.sqrt(u**2 + v**2)[..., np.newaxis]
        pred = np.concatenate([pred[..., :-2], magnitude], axis=-1)

        if sub_dir == "twophase_flow":
            # cut the timesteps
            gt = gt[:100]
            pred = pred[:100]

        T, H, W, C = gt.shape
        fps = idx[1]
        print(f"Generating videos for {sub_dir})")
        for channel, cmap in zip(range(C), COLORMAPS):
            output_path = video_dir / f"{sub_dir}"
            output_path.mkdir(parents=True, exist_ok=True)

            gt_c_data = gt[..., channel]  # (T, H, W)
            pred_c_data = pred[..., channel]  # (T, H, W)
            if np.all(gt_c_data == 0):
                print(f"  Skipping channel {channel} for {sub_dir} (all zeros)")
                continue
            generate_channel_gif(
                gt_c_data,
                output_path=output_path / f"gt_channel_{channel}.gif",
                fps=fps,
                cmap=cmap,
            )
            generate_channel_gif(
                pred_c_data,
                output_path=output_path / f"pred_channel_{channel}.gif",
                fps=fps,
                cmap=cmap,
            )
