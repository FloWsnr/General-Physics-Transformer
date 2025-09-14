"""
Custom loss functions for the transformer model.
By: Florian Wiesner
Date: 2025-04-25
"""

from typing import Optional

import torch
import torch.nn as nn


class NMSELoss(nn.Module):
    """Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
        which is time, height, width

    return_scalar: bool, optional
        Whether to return a scalar loss, by default True

    clip_max: float, optional
        Maximum value for the loss, by default None
    """

    def __init__(
        self,
        dims: tuple = (1, 2, 3),
        return_scalar: bool = True,
        clip_max: Optional[float] = None,
    ):
        super().__init__()
        self.dims = dims
        self.return_scalar = return_scalar
        self.clip_max = clip_max

    def forward(self, pred, target) -> torch.Tensor:
        """Calculate the normalized mean square error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Normalized MSE loss
        """
        # Calculate residuals
        residuals = pred - target
        # Normalize by mean squared target values (with small epsilon)
        target_norm = target.pow(2).mean(self.dims, keepdim=True) + 1e-6
        # Calculate normalized MSE
        nmse = residuals.pow(2).mean(self.dims, keepdim=True) / target_norm
        if self.return_scalar:
            return nmse.mean()
        if self.clip_max is not None:
            nmse = torch.clamp(nmse, max=self.clip_max)
        return nmse


class VMSELoss(nn.Module):
    """Variance-Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
        which is time, height, width

    return_scalar: bool, optional
        Whether to return a scalar loss, by default True

    clip_max: float, optional
        Maximum value for the loss, by default None
    """

    def __init__(
        self,
        dims: tuple = (1, 2, 3),
        return_scalar: bool = True,
        clip_max: Optional[float] = None,
    ):
        """Initialize Variance-Normalized MSE loss."""
        super().__init__()
        self.dims = dims
        self.return_scalar = return_scalar
        self.clip_max = clip_max

    def forward(self, pred, target) -> torch.Tensor:
        """Calculate the variance-normalized mean square error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Variance-Normalized MSE loss
        """
        # Calculate residuals
        residuals = pred - target
        # Calculate variance
        norm = torch.std(target, dim=self.dims, keepdim=True) ** 2 + 1e-6
        # Calculate normalized MSE
        nmse = residuals.pow(2).mean(self.dims, keepdim=True) / norm
        if self.return_scalar:
            return nmse.mean()
        if self.clip_max is not None:
            nmse = torch.clamp(nmse, max=self.clip_max)
        return nmse


class RNMSELoss(NMSELoss):
    """Root Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
        which is time, height, width

    return_scalar: bool, optional
        Whether to return a scalar loss, by default True

    clip_max: float, optional
        Maximum value for the loss, by default None
    """

    def __init__(
        self,
        dims: tuple = (1, 2, 3),
        return_scalar: bool = True,
        clip_max: Optional[float] = None,
    ):
        """Initialize Root NMSE loss."""
        super().__init__(dims, return_scalar, clip_max)

    def forward(self, pred, target) -> torch.Tensor:
        """Calculate the root normalized mean square error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Root Normalized MSE loss
        """
        nmse = super().forward(pred, target)
        return torch.sqrt(nmse)


class RVMSELoss(VMSELoss):
    """Root Variance-Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)

    return_scalar: bool, optional
        Whether to return a scalar loss, by default True

    clip_max: float, optional
        Maximum value for the loss, by default None
    """

    def __init__(
        self,
        dims: tuple = (1, 2, 3),
        return_scalar: bool = True,
        clip_max: Optional[float] = None,
    ):
        """Initialize Root Variance-Normalized MSE loss."""
        super().__init__(dims, return_scalar, clip_max)

    def forward(self, pred, target) -> torch.Tensor:
        """Calculate the root variance-normalized mean square error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Root Variance-Normalized MSE loss
        """
        nmse = super().forward(pred, target)
        return torch.sqrt(nmse)


class RMSE(nn.Module):
    """Root Mean Squared Error loss function."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target))
