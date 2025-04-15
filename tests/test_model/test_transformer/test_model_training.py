import pytest
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from einops import rearrange
from lpfm.model.transformer.model import get_model as get_transformer_model
from lpfm.model.base_models.resnet import get_model as get_resnet_model
from lpfm.data.mock_data import (
    MockMovingCircleData,
    MockCircleData,
    MockShrinkingCircleData,
)
from lpfm.model.ax_vit.avit import build_avit, AViTParams

from dadaptation import DAdaptAdam

from torchvision.datasets import MNIST
from torchvision import transforms


@pytest.mark.skip(reason="Test is not ready yet")
def test_avit_training_moving_circle():
    channels = 1
    height = 64
    width = 64
    time_steps = 4
    num_samples = 1000
    batch_size = 10

    params = AViTParams(
        patch_size=(4, 4),
        embed_dim=100,
        processor_blocks=1,
        n_states=1,
        block_type="axial",
        space_type="axial_attention",
        time_type="attention",
        num_heads=1,
        bias_type="rel",
        gradient_checkpointing=False,
    )

    model = build_avit(params)
    # model.expand_projections(2)
    for n, p in model.debed.named_parameters():
        print(n, p.shape)
    model.expand_projections(2)
    for n, p in model.debed.named_parameters():
        print(n, p.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = MockShrinkingCircleData(channels, time_steps, height, width, num_samples)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = DAdaptAdam(model.parameters(), lr=1)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    try:
        loss_history = []
        for epoch in range(16):
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                x = rearrange(x, "b t h w c -> t b c h w")
                y = rearrange(y, "b t h w c -> t b c h w")

                labels = [[0] for _ in range(x.shape[1])]
                labels = torch.tensor(labels).to(device)
                bcs = torch.tensor([[0, 0]]).to(device)

                optimizer.zero_grad()
                output = model(x, labels, bcs)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Loss: {loss.item()}")
                loss_history.append(loss.item())
            scheduler.step()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    # show multiple samples
    indices = [0, 10, 20, 30]

    for idx in indices:
        x, y = data[idx]
        x = rearrange(x, "t h w c -> t c h w")
        y = rearrange(y, "t h w c -> t c h w")
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        x = x.to(device)
        y = y.to(device)

        labels = [[0] for _ in range(x.shape[1])]
        labels = torch.tensor(labels).to(device)
        bcs = torch.tensor([[0, 0]]).to(device)

        model.eval()
        with torch.no_grad():
            output = model(x, labels, bcs)
        # Convert tensors to numpy for plotting
        output = output.unsqueeze(0)
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()

        # reshape back
        x_np = rearrange(x_np, "t b c h w -> b t h w c")
        y_np = rearrange(y_np, "t b c h w -> b t h w c")
        output_np = rearrange(output_np, "t b c h w -> b t h w c")

        # Create figure with 3 columns: input, ground truth, and prediction
        fig, axs = plt.subplots(time_steps, 3, figsize=(15, 12))

        # Create a common normalization for all plots
        vmin = min(
            x_np[0, ..., 0].min(), y_np[0, ..., 0].min(), output_np[0, ..., 0].min()
        )
        vmax = max(
            x_np[0, ..., 0].max(), y_np[0, ..., 0].max(), output_np[0, ..., 0].max()
        )

        images = []
        if time_steps == 1:
            # Plot all subplots
            im0 = axs[0].imshow(x_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im1 = axs[1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im2 = axs[2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            images.append(im0)
            images.append(im1)
            images.append(im2)
            # Add column titles
            axs[0].set_title("Input (x)")
            axs[1].set_title("Ground Truth (y)")
            axs[2].set_title("Prediction (output)")
        else:
            for i in range(time_steps):
                im0 = axs[i, 0].imshow(x_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
                im1 = axs[i, 1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                im2 = axs[i, 2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                images.append(im0)
                images.append(im1)
                images.append(im2)
            # Add column titles
            axs[0, 0].set_title("Input (x)")
            axs[0, 1].set_title("Ground Truth (y)")
            axs[0, 2].set_title("Prediction (output)")

        # Add a single colorbar for all subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(images[0], cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    plt.plot(loss_history)
    plt.show()


@pytest.mark.skip(reason="Test is not ready yet")
def test_model_training_moving_circle():
    channels = 1
    height = 16
    width = 16
    time_steps = 2
    num_samples = 1000
    batch_size = 10

    config = {
        "transformer": {
            "input_channels": channels,
            "hidden_channels": 120,  # must be divisible by 6
            "mlp_dim": 128,
            "num_heads": 1,
            "num_layers": 1,
            "pos_enc_mode": "absolute",
            "patch_size": (1, 4, 4),
            "dropout": 0.0,
            "stochastic_depth_rate": 0.0,
        },
        "img_size": (time_steps, height, width),
        "tokenizer": {
            "tokenizer_mode": "linear",
            "detokenizer_mode": "linear",
            "tokenizer_net_channels": [16, 32, 64],
            "detokenizer_net_channels": [64, 32, 16],
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_transformer_model(config)
    model.to(device)

    data = MockMovingCircleData(
        channels, time_steps, height, width, num_samples, binary=False
    )
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    try:
        loss_history = []
        for epoch in range(30):
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Loss: {loss.item()}")
                loss_history.append(loss.item())
            scheduler.step()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    # show multiple samples
    indices = [10]

    for idx in indices:
        x, y = data[idx]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = x.to(device)
        y = y.to(device)

        model.eval()
        with torch.no_grad():
            output = model(x)
        # Convert tensors to numpy for plotting
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()

        # Create figure with 3 columns: input, ground truth, and prediction
        fig, axs = plt.subplots(time_steps, 3, figsize=(15, 12))

        # Create a common normalization for all plots
        vmin = min(
            x_np[0, ..., 0].min(), y_np[0, ..., 0].min(), output_np[0, ..., 0].min()
        )
        vmax = max(
            x_np[0, ..., 0].max(), y_np[0, ..., 0].max(), output_np[0, ..., 0].max()
        )

        images = []
        if time_steps == 1:
            # Plot all subplots
            im0 = axs[0].imshow(x_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im1 = axs[1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im2 = axs[2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            images.append(im0)
            images.append(im1)
            images.append(im2)
            # Add column titles
            axs[0].set_title("Input (x)")
            axs[1].set_title("Ground Truth (y)")
            axs[2].set_title("Prediction (output)")
        else:
            for i in range(time_steps):
                im0 = axs[i, 0].imshow(x_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
                im1 = axs[i, 1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                im2 = axs[i, 2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                images.append(im0)
                images.append(im1)
                images.append(im2)
            # Add column titles
            axs[0, 0].set_title("Input (x)")
            axs[0, 1].set_title("Ground Truth (y)")
            axs[0, 2].set_title("Prediction (output)")

        # Add a single colorbar for all subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(images[0], cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    plt.plot(loss_history)
    plt.show()


@pytest.mark.skip(reason="Test is not ready yet")
def test_model_training_shrinking_circle():
    channels = 1
    height = 64
    width = 64
    time_steps = 2
    num_samples = 1000
    batch_size = 20

    config = {
        "transformer": {
            "input_channels": channels,
            "hidden_channels": 120,  # must be divisible by 6
            "mlp_dim": 128,
            "num_heads": 1,
            "num_layers": 2,
            "pos_enc_mode": "absolute",
            "patch_size": (1, 16,16),
            "dropout": 0.0,
            "stochastic_depth_rate": 0.0,
        },
        "img_size": (time_steps, height, width),
        "tokenizer": {
            "tokenizer_mode": "linear",
            "detokenizer_mode": "linear",
            "tokenizer_net_channels": [16, 32, 64],
            "detokenizer_net_channels": [64, 32, 16],
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_transformer_model(config)
    model.to(device)

    data = MockShrinkingCircleData(
        channels, time_steps, height, width, num_samples, binary=True
    )
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    try:
        loss_history = []
        for epoch in range(30):
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Loss: {loss.item()}")
                loss_history.append(loss.item())
            # scheduler.step()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    # show multiple samples
    indices = [10]

    for idx in indices:
        x, y = data[idx]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = x.to(device)
        y = y.to(device)

        model.eval()
        with torch.no_grad():
            output = model(x)
        # Convert tensors to numpy for plotting
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()

        # Create figure with 3 columns: input, ground truth, and prediction
        fig, axs = plt.subplots(time_steps, 3, figsize=(15, 12))

        # Create a common normalization for all plots
        vmin = min(
            x_np[0, ..., 0].min(), y_np[0, ..., 0].min(), output_np[0, ..., 0].min()
        )
        vmax = max(
            x_np[0, ..., 0].max(), y_np[0, ..., 0].max(), output_np[0, ..., 0].max()
        )

        images = []
        if time_steps == 1:
            # Plot all subplots
            im0 = axs[0].imshow(x_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im1 = axs[1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im2 = axs[2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            images.append(im0)
            images.append(im1)
            images.append(im2)
            # Add column titles
            axs[0].set_title("Input (x)")
            axs[1].set_title("Ground Truth (y)")
            axs[2].set_title("Prediction (output)")
        else:
            for i in range(time_steps):
                im0 = axs[i, 0].imshow(x_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
                im1 = axs[i, 1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                im2 = axs[i, 2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                images.append(im0)
                images.append(im1)
                images.append(im2)
            # Add column titles
            axs[0, 0].set_title("Input (x)")
            axs[0, 1].set_title("Ground Truth (y)")
            axs[0, 2].set_title("Prediction (output)")

        # Add a single colorbar for all subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(images[0], cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    plt.plot(loss_history)
    plt.show()


@pytest.mark.skip(reason="Test is not ready yet")
def test_resnet_training_shrinking_circle():
    channels = 1
    height = 16
    width = 16
    time_steps = 1
    num_samples = 1000
    batch_size = 10

    in_channels = channels * time_steps

    config = {
        "in_channels": in_channels,
        "block_dimensions": [4, 16, 16, 4, in_channels],
        "kernel_size": 3,
        "pooling": False,
        "padding": "same",
        "padding_mode": "zeros",
        "stride": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet_model(config)
    model.to(device)

    # data = MockShrinkingCircleData(channels, time_steps, height, width, num_samples)
    data = MNIST(
        root=r"C:\Users\zsa8rk\Coding\MetaPARC\logs",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    try:
        loss_history = []
        for epoch in range(16):
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                x = rearrange(x, "b t h w c -> b (c t) h w")
                y = rearrange(y, "b t h w c -> b (c t) h w")

                optimizer.zero_grad()
                output = model(x)
                output = rearrange(output, "b (c t) h w -> b t h w c", c=channels)
                y = rearrange(y, "b (c t) h w -> b t h w c", c=channels)
                x = rearrange(x, "b (c t) h w -> b t h w c", c=channels)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Loss: {loss.item()}")
                loss_history.append(loss.item())
            scheduler.step()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    # show multiple samples
    indices = [0, 10, 20, 30]

    for idx in indices:
        x, y = data[idx]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = x.to(device)
        y = y.to(device)

        x = rearrange(x, "b t h w c -> b (c t) h w")
        y = rearrange(y, "b t h w c -> b (c t) h w")

        model.eval()
        with torch.no_grad():
            output = model(x)
        output = rearrange(output, "b (c t) h w -> b t h w c", c=channels)
        y = rearrange(y, "b (c t) h w -> b t h w c", c=channels)
        x = rearrange(x, "b (c t) h w -> b t h w c", c=channels)
        # Convert tensors to numpy for plotting
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()

        # Create figure with 3 columns: input, ground truth, and prediction
        fig, axs = plt.subplots(time_steps, 3, figsize=(15, 12))

        # Create a common normalization for all plots
        vmin = min(
            x_np[0, ..., 0].min(), y_np[0, ..., 0].min(), output_np[0, ..., 0].min()
        )
        vmax = max(
            x_np[0, ..., 0].max(), y_np[0, ..., 0].max(), output_np[0, ..., 0].max()
        )

        images = []
        if time_steps == 1:
            # Plot all subplots
            im0 = axs[0].imshow(x_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im1 = axs[1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im2 = axs[2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            images.append(im0)
            images.append(im1)
            images.append(im2)
            # Add column titles
            axs[0].set_title("Input (x)")
            axs[1].set_title("Ground Truth (y)")
            axs[2].set_title("Prediction (output)")
        else:
            for i in range(time_steps):
                im0 = axs[i, 0].imshow(x_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
                im1 = axs[i, 1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                im2 = axs[i, 2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                images.append(im0)
                images.append(im1)
                images.append(im2)
            # Add column titles
            axs[0, 0].set_title("Input (x)")
            axs[0, 1].set_title("Ground Truth (y)")
            axs[0, 2].set_title("Prediction (output)")

        # Add a single colorbar for all subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(images[0], cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    plt.plot(loss_history)
    plt.show()


@pytest.mark.skip(reason="Test is not ready yet")
def test_resnet_training_moving_circle():
    channels = 1
    height = 16
    width = 16
    time_steps = 1
    num_samples = 1000
    batch_size = 100

    in_channels = channels * time_steps

    config = {
        "in_channels": in_channels,
        "block_dimensions": [32, 64, 64, 32, in_channels],
        "kernel_size": 3,
        "pooling": False,
        "padding": "same",
        "padding_mode": "zeros",
        "stride": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet_model(config)
    model.to(device)

    data = MockMovingCircleData(
        channels, time_steps, height, width, num_samples, binary=True
    )
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    try:
        loss_history = []
        for epoch in range(16):
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                x = rearrange(x, "b t h w c -> b (c t) h w")
                y = rearrange(y, "b t h w c -> b (c t) h w")

                optimizer.zero_grad()
                output = model(x)
                output = rearrange(output, "b (c t) h w -> b t h w c", c=channels)
                y = rearrange(y, "b (c t) h w -> b t h w c", c=channels)
                x = rearrange(x, "b (c t) h w -> b t h w c", c=channels)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Loss: {loss.item()}")
                loss_history.append(loss.item())
            # scheduler.step()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    # show multiple samples
    indices = [0]

    for idx in indices:
        x, y = data[idx]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = x.to(device)
        y = y.to(device)

        x = rearrange(x, "b t h w c -> b (c t) h w")
        y = rearrange(y, "b t h w c -> b (c t) h w")

        model.eval()
        with torch.no_grad():
            output = model(x)
        output = rearrange(output, "b (c t) h w -> b t h w c", c=channels)
        y = rearrange(y, "b (c t) h w -> b t h w c", c=channels)
        x = rearrange(x, "b (c t) h w -> b t h w c", c=channels)
        # Convert tensors to numpy for plotting
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()

        # Create figure with 3 columns: input, ground truth, and prediction
        fig, axs = plt.subplots(time_steps, 3, figsize=(15, 12))

        # Create a common normalization for all plots
        vmin = min(
            x_np[0, ..., 0].min(), y_np[0, ..., 0].min(), output_np[0, ..., 0].min()
        )
        vmax = max(
            x_np[0, ..., 0].max(), y_np[0, ..., 0].max(), output_np[0, ..., 0].max()
        )

        images = []
        if time_steps == 1:
            # Plot all subplots
            im0 = axs[0].imshow(x_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im1 = axs[1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            im2 = axs[2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
            images.append(im0)
            images.append(im1)
            images.append(im2)
            # Add column titles
            axs[0].set_title("Input (x)")
            axs[1].set_title("Ground Truth (y)")
            axs[2].set_title("Prediction (output)")
        else:
            for i in range(time_steps):
                im0 = axs[i, 0].imshow(x_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
                im1 = axs[i, 1].imshow(y_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                im2 = axs[i, 2].imshow(output_np[0, 0, ..., 0], vmin=vmin, vmax=vmax)
                images.append(im0)
                images.append(im1)
                images.append(im2)
            # Add column titles
            axs[0, 0].set_title("Input (x)")
            axs[0, 1].set_title("Ground Truth (y)")
            axs[0, 2].set_title("Prediction (output)")

        # Add a single colorbar for all subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(images[0], cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    plt.plot(loss_history)
    plt.show()


@pytest.mark.skip(reason="Test is not ready yet")
def test_resnet_training_mnist():
    channels = 1
    height = 16
    width = 16
    time_steps = 1
    num_samples = 1000
    batch_size = 10

    in_channels = channels * time_steps

    config = {
        "in_channels": in_channels,
        "block_dimensions": [32, 64, 64, 32, in_channels],
        "kernel_size": 3,
        "pooling": False,
        "padding": "same",
        "padding_mode": "zeros",
        "stride": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet_model(config)
    model.to(device)

    data = MNIST(
        root=r"C:\Users\zsa8rk\Coding\MetaPARC\logs",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    try:
        loss_history = []
        for epoch in range(16):
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                output = model(x)

                loss = criterion(output, x)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Loss: {loss.item()}")
                loss_history.append(loss.item())
            scheduler.step()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)

    # Convert tensors to numpy for plotting
    # batch, channels, height, width
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()

    # Create figure with 2 rows: input and prediction and three batch samples (col)
    fig, axs = plt.subplots(2, 3, figsize=(15, 12))

    # Create a common normalization for all plots
    vmin = min(x_np.min(), output_np.min())
    vmax = max(x_np.max(), output_np.max())

    images = []
    # Plot three MNIST images and their predictions
    for i in range(3):
        # Input image
        im0 = axs[0, i].imshow(x_np[i, 0, ...], vmin=vmin, vmax=vmax, cmap="gray")
        # Prediction
        im1 = axs[1, i].imshow(output_np[i, 0, ...], vmin=vmin, vmax=vmax, cmap="gray")
        images.append(im0)
        images.append(im1)

        # Add row labels
        axs[0, i].set_ylabel(f"Sample {i}")

    # Add column titles
    axs[0, 0].set_title("Input")
    axs[1, 0].set_title("Prediction")

    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(images[0], cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    plt.plot(loss_history)
    plt.show()


if __name__ == "__main__":
    test_model_training_shrinking_circle()
