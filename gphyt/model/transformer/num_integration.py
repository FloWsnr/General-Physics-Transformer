"""
Numerical integration methods
By: Florian Wiesner
Date: 2025-05-05
"""

from typing import Callable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Integrator(nn.Module, ABC):
    def __init__(self):
        super(Integrator, self).__init__()

    @abstractmethod
    def forward(self, diff: Callable, input: torch.Tensor, step_size: float = 1.0):
        pass


class Euler(Integrator):
    """Euler integration. Fixed step, 1st order.

    Parameters
    ----------
    diff : Callable
        The derivative function
    input : torch.Tensor
        The input state (x from dataset)
    step_size : float
        The step size, by default 1.0
    """

    def __init__(self):
        super(Euler, self).__init__()

    def forward(self, diff: Callable, input: torch.Tensor, step_size: float = 1.0):
        final_state = input + step_size * diff(input)
        return final_state


class RK4(Integrator):
    """Runge-Kutta 4th order integration. Fixed step, 4th order.

    Parameters
    ----------
    diff : Callable
        The derivative function
    input : torch.Tensor
        The input state (x from dataset)
    step_size : float
        The step size, by default 1.0
    """

    def __init__(self):
        super(RK4, self).__init__()

    def forward(self, diff: Callable, input: torch.Tensor, step_size: float = 1.0):
        k1 = diff(input)
        k2 = diff(input + 0.5 * step_size * k1)
        k3 = diff(input + 0.5 * step_size * k2)
        k4 = diff(input + step_size * k3)
        update = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        final_state = input + step_size * update
        return final_state


class Heun(Integrator):
    """Heun's method. Fixed step, 2nd order.
    Parameters
    ----------
    diff : Callable
        The derivative function
    input : torch.Tensor
        The input state (x from dataset)
    step_size : float
        The step size, by default 1.0
    """

    def __init__(self):
        super(Heun, self).__init__()

    def forward(self, diff: Callable, input: torch.Tensor, step_size: float = 1.0):
        k1 = diff(input)
        k2 = diff(input + step_size * k1)
        update = 0.5 * (k1 + k2)
        final_state = input + step_size * update
        return final_state
