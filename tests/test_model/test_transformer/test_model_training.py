import pytest
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from metaparc.model.transformer.model import get_model as get_transformer_model
from metaparc.model.base_models.resnet import get_model as get_resnet_model
from metaparc.data.mock_data import (
    MockMovingCircleData,
    MockCircleData,
    MockShrinkingCircleData,
)


@pytest.mark.skip(reason="Test is not ready yet")
def test_model_training_moving_circle():
    channels = 1
    height = 16
    width = 16
    time_steps = 1
    num_samples = 1000
    batch_size = 10

    config = {
        "transformer": {
            "input_channels": channels,
            "hidden_channels": 600,  # must be divisible by 6
            "mlp_dim": 48,
            "num_heads": 1,
            "num_layers": 1,
            "pos_enc_mode": "rope",
            "patch_size": (1, 2, 2),
            "dropout": 0.0,
            "stochastic_depth_rate": 0.0,
        },
        "img_size": (time_steps, height, width),
        "tokenizer": {
            "tokenizer_mode": "conv_net",
            "detokenizer_mode": "conv_net",
            "tokenizer_net_channels": [16, 32, 64, 128, 256],
            "detokenizer_net_channels": [256, 128, 64, 32, 16],
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_transformer_model(config)
    model.to(device)

    data = MockShrinkingCircleData(channels, time_steps, height, width, num_samples)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    try:
        loss_history = []
        for epoch in range(10):
            for batch in dataloader:
                x = batch
                x = x.to(device)

                y = x[:, 1:, ...]
                x = x[:, :-1, ...]

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

    # get one sample
    x = data[0]
    x = x.unsqueeze(0)
    x = x.to(device)
    y = y.to(device)
    y = x[:, 1:, ...]
    x = x[:, :-1, ...]

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
    vmin = min(x_np[0, ..., 0].min(), y_np[0, ..., 0].min(), output_np[0, ..., 0].min())
    vmax = max(x_np[0, ..., 0].max(), y_np[0, ..., 0].max(), output_np[0, ..., 0].max())

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
            im1 = axs[i, 1].imshow(y_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
            im2 = axs[i, 2].imshow(output_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
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
def test_model_training_circle():
    channels = 1
    height = 16
    width = 16
    time_steps = 1
    num_samples = 1000
    batch_size = 10

    config = {
        "input_channels": channels,
        "hidden_channels": 6 * 5,  # must be divisible by 6
        "mlp_dim": 64,
        "num_heads": 1,
        "num_layers": 1,
        "patch_size": (1, 4, 4),
        "tokenizer_mode": "linear",
        "pos_enc_mode": "rope",
        "stochastic_depth_rate": 0.0,
        "dropout": 0.0,
        "img_size": (time_steps, height, width),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_transformer_model(config)
    model.to(device)

    data = MockCircleData(channels, time_steps, height, width, num_samples)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    try:
        loss_history = []
        for epoch in range(15):
            for batch in dataloader:
                x = batch
                x = x.to(device)
                y = x[:, 0, ...].unsqueeze(1)
                x = x[:, 0, ...].unsqueeze(1)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

                print(f"Epoch {epoch}, Loss: {loss.item()}")
            scheduler.step()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    # get four samples
    indices = [0, 10, 20, 30]
    samples = []
    outputs = []

    for idx in indices:
        x = data[idx]
        x = x.unsqueeze(0)
        x = x.to(device)
        output = model(x)
        samples.append(x)
        outputs.append(output)

    # Convert tensors to numpy for plotting
    x_np = [s.detach().cpu().numpy() for s in samples]
    output_np = [o.detach().cpu().numpy() for o in outputs]

    # Create figure with 2 columns (input and prediction) and 4 rows (samples)
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))

    # Create a common normalization for all plots
    vmin = min(
        [x[0, ..., 0].min() for x in x_np] + [o[0, ..., 0].min() for o in output_np]
    )
    vmax = max(
        [x[0, ..., 0].max() for x in x_np] + [o[0, ..., 0].max() for o in output_np]
    )

    # Plot all subplots
    images = []
    for i in range(4):
        im0 = axs[i, 0].imshow(x_np[i][0, 0, ..., 0], vmin=vmin, vmax=vmax)
        im1 = axs[i, 1].imshow(output_np[i][0, 0, ..., 0], vmin=vmin, vmax=vmax)
        if i == 0:  # Only add the first image to the list for colorbar
            images.append(im0)

    # Add column titles
    axs[0, 0].set_title("Input (x)")
    axs[0, 1].set_title("Prediction (output)")

    # Add row titles
    for i in range(4):
        axs[i, 0].set_ylabel(f"Sample {indices[i]}")

    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(images[0], cax=cbar_ax)

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    plt.plot(loss_history)
    plt.show()


@pytest.mark.skip(reason="Test is not ready yet")
def test_model_training_moving_circle_resnet():
    channels = 1
    height = 16
    width = 16
    time_steps = 2
    num_samples = 1000
    batch_size = 10

    config = {
        "in_channels": channels * time_steps,
        "block_dimensions": [4, 8, 4, channels * time_steps],
        "kernel_size": 3,
        "pooling": False,
        "padding": "same",
        "padding_mode": "zeros",
        "stride": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet_model(config)
    model.to(device)

    data = MockMovingCircleData(channels, time_steps, height, width, num_samples)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    try:
        loss_history = []
        for epoch in range(20):
            for batch in dataloader:
                x = batch
                x = x.to(device)

                y = x[:, 1:, ...]
                x = x[:, :-1, ...]

                x = rearrange(x, "b t h w c-> b (t c) h w")
                y = rearrange(y, "b t h w c-> b (t c) h w")

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

    # get one sample
    x = data[0]
    x = x.unsqueeze(0)
    y = x[:, 1:, ...]
    x = x[:, :-1, ...]
    x = rearrange(x, "b t h w c-> b (t c) h w")
    y = rearrange(y, "b t h w c-> b (t c) h w")
    x = x.to(device)
    y = y.to(device)
    output = model(x)

    output = rearrange(output, "b (t c) h w-> b t h w c", t=time_steps)
    y = rearrange(y, "b (t c) h w-> b t h w c", t=time_steps)
    x = rearrange(x, "b (t c) h w-> b t h w c", t=time_steps)

    # Convert tensors to numpy for plotting
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()

    # Create figure with 3 columns: input, ground truth, and prediction
    fig, axs = plt.subplots(time_steps, 3, figsize=(15, 12))

    # Create a common normalization for all plots
    vmin = min(x_np[0, ..., 0].min(), y_np[0, ..., 0].min(), output_np[0, ..., 0].min())
    vmax = max(x_np[0, ..., 0].max(), y_np[0, ..., 0].max(), output_np[0, ..., 0].max())

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
            im1 = axs[i, 1].imshow(y_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
            im2 = axs[i, 2].imshow(output_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
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


if __name__ == "__main__":
    # test_model_training_moving_circle_resnet()
    test_model_training_moving_circle()
    # test_model_training_circle()
