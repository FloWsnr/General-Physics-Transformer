"""
Custom loss functions.
By: Florian Wiesner
Date: 2025-04-25
"""

from typing import Optional

import torch
import torch.nn as nn


class MSE(nn.Module):
    """Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default None
    """

    def __init__(
        self,
        dims: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        self.dims = dims

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the mean squared error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            MSE loss
        """
        mse = (pred - target).pow(2).mean(dim=self.dims)
        return mse


class MAE(nn.Module):
    """Mean Absolute Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default None
    """

    def __init__(
        self,
        dims: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        self.dims = dims

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the mean absolute error.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            MAE loss
        """
        mae = (pred - target).abs().mean(dim=self.dims)
        return mae


class NMSE(MSE):
    """Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default None
    """

    def __init__(
        self,
        dims: Optional[tuple[int, ...]] = None,
    ):
        super().__init__(dims)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        mse = super().forward(pred, target)
        return mse / (target.pow(2).mean(self.dims) + 1e-6)


class VMSE(MSE):
    """Variance-Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default None
    """

    def __init__(
        self,
        dims: Optional[tuple[int, ...]] = None,
    ):
        """Initialize Variance-Normalized MSE loss."""
        super().__init__(dims)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        mse = super().forward(pred, target)
        # Calculate variance
        norm = torch.std(target, dim=self.dims) ** 2 + 1e-6
        return mse / norm


class NRMSE(NMSE):
    """Normalized Root Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default None
    """

    def __init__(
        self,
        dims: Optional[tuple[int, ...]] = None,
    ):
        """Initialize Root NMSE loss."""
        super().__init__(dims)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


class VRMSE(VMSE):
    """Variance-Normalized Root Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
    """

    def __init__(
        self,
        dims: Optional[tuple[int, ...]] = None,
    ):
        super().__init__(dims)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
            Variance-Normalized RMSE loss
        """
        nmse = super().forward(pred, target)
        return torch.sqrt(nmse)


class RMSE(MSE):
    """Root Mean Squared Error loss function."""

    def __init__(self, dims: Optional[tuple[int, ...]] = None):
        super().__init__(dims)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(super().forward(pred, target))
