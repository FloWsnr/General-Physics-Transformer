import torch
import torch.nn as nn


class Euler(nn.Module):
    """Euler integration. Fixed step, 1st order.

    Parameters
    ----------
    dt : torch.Tensor
        The derivative of the input state with respect to time
    input : torch.Tensor
        The input state (x from dataset)
    step_size : float
        The step size, by default 1.0
    """

    def __init__(self):
        super(Euler, self).__init__()

    def forward(self, dt: torch.Tensor, input: torch.Tensor, step_size: float = 1.0):
        final_state = input + step_size * dt
        return final_state
