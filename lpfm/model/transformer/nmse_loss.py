import torch
import torch.nn as nn


class NMSELoss(nn.Module):
    """Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
        which is time, height, width

    reduce: bool, optional
        Whether to reduce the loss over the dimensions, by default True

    Attributes
    ----------
    dims : tuple
        Dimensions to reduce over
    """

    def __init__(self, dims=(1, 2, 3), reduce=True):
        """Initialize NMSE loss.

        Parameters
        ----------
        dims : tuple, optional
            Dimensions to reduce over, by default (1, 2, 3)
        """
        super().__init__()
        self.dims = dims
        self.reduce = reduce
    def forward(self, pred, target):
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
        target_norm = target.pow(2).mean(self.dims, keepdim=True) + 1e-8
        # Calculate normalized MSE
        nmse = residuals.pow(2).mean(self.dims, keepdim=True) / target_norm
        if self.reduce:
            return nmse.mean()
        return nmse


class VMSELoss(nn.Module):
    """Variance-Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
        which is time, height, width
    reduce: bool, optional
        Whether to reduce the loss over the dimensions, by default True

    Attributes
    ----------
    dims : tuple
        Dimensions to reduce over
    """

    def __init__(self, dims=(1, 2, 3), reduce=True):
        """Initialize Variance-Normalized MSE loss.

        Parameters
        ----------
        dims : tuple, optional
            Dimensions to reduce over, by default (1, 2, 3)
        """
        super().__init__()
        self.dims = dims
        self.reduce = reduce
    def forward(self, pred, target):
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
        norm = torch.std(target, dim=self.dims, keepdim=True) ** 2 + 1e-8
        # Calculate normalized MSE
        nmse = residuals.pow(2).mean(self.dims, keepdim=True) / norm
        if self.reduce:
            return nmse.mean()
        return nmse


class RNMSELoss(NMSELoss):
    """Root Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
        which is time, height, width

    reduce: bool, optional
        Whether to reduce the loss over the dimensions, by default True

    Attributes
    ----------
    dims : tuple
        Dimensions to reduce over
    """

    def __init__(self, dims=(1, 2, 3), reduce=True):
        """Initialize Root NMSE loss.

        Parameters
        ----------
        dims : tuple, optional
            Dimensions to reduce over, by default (1, 2, 3)
        """
        super().__init__(dims, reduce)

    def forward(self, pred, target):
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


class VRNMSELoss(VMSELoss):
    """Root Variance-Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)

    reduce: bool, optional
        Whether to reduce the loss over the dimensions, by default True

    Attributes
    ----------
    dims : tuple
        Dimensions to reduce over
    """

    def __init__(self, dims=(1, 2, 3), reduce=True):
        """Initialize Root Variance-Normalized MSE loss.

        Parameters
        ----------
        dims : tuple, optional
            Dimensions to reduce over, by default (1, 2, 3)
        """
        super().__init__(dims, reduce)

    def forward(self, pred, target):
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
