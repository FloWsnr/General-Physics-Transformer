from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn


class NumericalIntegrator(nn.Module, ABC):
    def __init__(self, **kwarg):
        super(NumericalIntegrator, self).__init__(**kwarg)

    @abstractmethod
    def forward(self, f: Callable, t: float, current: torch.Tensor, step_size: float):
        pass


class Euler(NumericalIntegrator):
    """
    Euler integration. Fixed step, 1st order.

    Parameters
    ----------
    f : callable
        The function that returns time derivative
    t : float
        Current time
    current : torch.Tensor
        The current state and velocity variables
    step_size : float
        Integration step size

    Returns
    -------
    final_state : torch.Tensor
        The next state and velocity variables, same shape as current
    update : torch.Tensor
        The updates at this step, same shape as current
    """

    def __init__(self, **kwarg):
        super(Euler, self).__init__(**kwarg)

    def forward(self, f: Callable, t: float, current: torch.Tensor, step_size: float):
        update = f(t, current)
        final_state = current + step_size * update
        return final_state, update


class RK4(NumericalIntegrator):
    def __init__(self, **kwarg):
        super(RK4, self).__init__(**kwarg)

    def forward(self, f: Callable, t: float, current: torch.Tensor, step_size: float):
        """
        RK4 integration. Fixed step, 4th order.

        Parameters
        ----------
        f: function
            The function that returns time derivative
        t: float
            Current time
        current: torch.Tensor
            The current state and velocity variables
        step_size: float
            Integration step size

        Returns
        -------
        final_state: torch.Tensor
            The next state and velocity variables, same shape as current
        update: torch.Tensor
            The updates at this step, same shape as current
        """
        # Compute k1
        k1 = f(t, current)
        # Compute k2
        inp_k2 = current + 0.5 * step_size * k1
        k2 = f(t + 0.5 * step_size, inp_k2)
        # Compute k3
        inp_k3 = current + 0.5 * step_size * k2
        k3 = f(t + 0.5 * step_size, inp_k3)
        # Compute k4
        inp_k4 = current + step_size * k3
        k4 = f(t + step_size, inp_k4)
        # Final
        update = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        final_state = current + step_size * update
        return final_state, update


class Heun(NumericalIntegrator):
    def __init__(self, **kwarg):
        super(Heun, self).__init__(**kwarg)

    def forward(self, f: Callable, t: float, current: torch.Tensor, step_size: float):
        """
        Heun integration. Fixed step, explicit, 2nd order.

        Parameters
        ----------
        f : function
            The function that returns time deriviative
        t : float
            Current time
        current : torch.Tensor
            The current state and velocity variables
        step_size : float
            Integration step size

        Returns
        -------
        final_state : tensor
            with the same shape of ```current```, the next state and velocity variables
        update : tensor
            with the same shape of ```current```, the update in this step
        """
        # Compute k1
        k1 = f(t, current)
        # Compute k2
        inp_k2 = current + step_size * k1
        k2 = f(t + step_size, inp_k2)
        # Final
        update = 1 / 2 * (k1 + k2)
        final_state = current + step_size * update
        return final_state, update
