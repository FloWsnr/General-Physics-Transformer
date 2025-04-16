import torch.nn as nn


class NMSELoss(nn.Module):
    """Normalized Mean Squared Error loss function.

    Parameters
    ----------
    dims : tuple, optional
        Dimensions to reduce over, by default (1, 2, 3)
        which is time, height, width

    Attributes
    ----------
    dims : tuple
        Dimensions to reduce over
    """

    def __init__(self, dims=(1, 2, 3)):
        """Initialize NMSE loss.

        Parameters
        ----------
        dims : tuple, optional
            Dimensions to reduce over, by default (1, 2, 3)
        """
        super().__init__()
        self.dims = dims

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
        # Return mean over batch dimensions
        return nmse.mean()
