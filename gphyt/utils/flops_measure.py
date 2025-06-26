import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torchtnt.utils.flops import FlopTensorDispatchMode
from torch.profiler import profile, record_function, ProfilerActivity

from lpfm.model.transformer.model import get_model


def load_model(model_config: dict) -> nn.Module:
    model = get_model(model_config)
    model = torch.compile(model, mode="max-autotune")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def measure_flops_2(model, input_shape: tuple[int, ...]) -> tuple[dict, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_scaler = torch.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # warmup
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        input = torch.randn(input_shape).to(device)
        res = model(input).mean()

    with profile(
        activities=[ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        with record_function("foward"):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                input = torch.randn(input_shape).to(device)
                res = model(input).mean()

        with record_function("backward"):
            grad_scaler.scale(res).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

    return prof.key_averages().table(row_limit=-1)


def measure_flops(model, input_shape: tuple[int, ...]) -> tuple[dict, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_scaler = torch.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # warmup
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        input = torch.randn(input_shape).to(device)
        res = model(input).mean()

    with FlopTensorDispatchMode(model) as ftdm:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            input = torch.randn(input_shape).to(device)
            res = model(input).mean()
        flops_forward = copy.deepcopy(ftdm.flop_counts)

        # reset count before counting backward flops
        ftdm.reset()

        grad_scaler.scale(res).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        flops_backward = copy.deepcopy(ftdm.flop_counts)
    return flops_forward, flops_backward


def sum_flops(flops_dict: dict) -> int:
    flops = 0
    for _, value in flops_dict.items():
        if isinstance(value, dict) or isinstance(value, defaultdict):
            flops += sum_flops(value)
        else:
            flops += value
    return flops


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_config = {
        "transformer": {
            "model_size": "LPFM_XL",
            "input_channels": 5,
            "att_mode": "full",
            "dropout": 0.0,
            "pos_enc_mode": "absolute",
            "patch_size": [1, 16, 16],
            "stochastic_depth_rate": 0.0,
            "use_derivatives": True,
            "parc_mode": True,
        },
        "tokenizer": {
            "tokenizer_mode": "linear",
            "detokenizer_mode": "linear",
            "tokenizer_overlap": 0,
            "detokenizer_overlap": 0,
            "tokenizer_net_channels": None,
            "detokenizer_net_channels": None,
        },
        "img_size": [4, 256, 128],
    }
    model = load_model(model_config)
    input_shape = (16, 4, 256, 128, 5)
    flops_forward, flops_backward = measure_flops(model, input_shape)

    # count flops for each layer
    flops_forward = sum_flops(flops_forward)
    flops_backward = sum_flops(flops_backward)

    # print flops in human readable format
    print(f"Forward flops: {flops_forward / 1e12:.2f} TFlops")
    print(f"Backward flops: {flops_backward / 1e12:.2f} TFlops")

    # sum forward and backward flops
    total_flops = flops_forward + flops_backward
    print(f"Total flops: {total_flops / 1e12:.2f} TFlops")
