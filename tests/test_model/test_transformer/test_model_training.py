import pytest
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from metaparc.model.transformer.model import get_model
from metaparc.data.mock_data import MockData


@pytest.mark.skip(reason="Test is not ready yet")
def test_model_training():

    channels = 5
    height = 32
    width = 16
    time_steps = 4
    num_lines = 2

    config = {
        "input_channels": channels,
        "hidden_channels": 6*5, # must be divisible by 6
        "mlp_dim": 128,
        "num_heads": 1,
        "num_layers": 2,
        "patch_size": (2, 4, 4),
        "tokenizer_mode": "linear",
        "stochastic_depth": 0.0,
        "dropout": 0.0,
        "img_size": (time_steps, height, width),
    }

    model = get_model(config)

    data = MockData(channels, time_steps, height, width, num_lines)
    dataloader = DataLoader(data, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(40):
        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")
        scheduler.step()

    # get one sample
    x, y = data[height//2]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
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
    
    # Plot all subplots
    images = []
    for i in range(time_steps):
        im0 = axs[i, 0].imshow(x_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
        im1 = axs[i, 1].imshow(y_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
        im2 = axs[i, 2].imshow(output_np[0, i, ..., 0], vmin=vmin, vmax=vmax)
        images.append(im0)
        images.append(im1)
        images.append(im2)
    
    # Add column titles
    axs[0, 0].set_title('Input (x)')
    axs[0, 1].set_title('Ground Truth (y)')
    axs[0, 2].set_title('Prediction (output)')
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(images[0], cax=cbar_ax)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    assert loss < 0.1

if __name__ == "__main__":
    test_model_training()
