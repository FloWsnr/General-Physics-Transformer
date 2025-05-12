from typing import Optional, Tuple

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from einops import rearrange

from lpfm.model.transformer.model import FiniteDifference


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and a skip connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block(x)
        return x + residual


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module that maps continuous inputs to discrete codes.

    Parameters
    ----------
    codebook_size : int
        Number of embeddings in the codebook
    codebook_dim : int
        Dimension of each embedding vector
    commitment_cost : float
        Commitment cost for the codebook loss
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the vector quantizer.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, time, height, width, embedding_dim)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Quantized tensor
            - Codebook loss
            - Encoding indices
        """
        # Reshape input to (batch_size * time * height * width, embedding_dim)
        z_flat = z.reshape(-1, self.codebook_dim)

        # Calculate distances to codebook vectors
        d = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Get closest codebook indices
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.commitment_cost * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices


class VQVAETokenizer(nn.Module):
    """
    VQ-VAE Tokenizer that encodes input into latent representation.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    hidden_dim : int
        Hidden dimension for the encoder conv layers
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # First strided conv
            nn.Conv3d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Second strided conv
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            # Two residual blocks
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VQ-VAE tokenizer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, time, height, width)

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, channels, time, height, width)
        """
        return self.encoder(x)


class VQVAEDetokenizer(nn.Module):
    """
    VQ-VAE Detokenizer that decodes discrete latent codes back to the original space.

    Parameters
    ----------
    out_channels : int
        Number of output channels
    hidden_dim : int
        Hidden dimension for the decoder
    """

    def __init__(
        self,
        out_channels: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Decoder
        self.decoder = nn.Sequential(
            # Two residual blocks
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            # First transposed conv
            nn.ReLU(),
            nn.ConvTranspose3d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            # Second transposed conv
            nn.ConvTranspose3d(
                hidden_dim, out_channels, kernel_size=4, stride=2, padding=1
            ),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VQ-VAE detokenizer.

        Parameters
        ----------
        z_q : torch.Tensor
            Quantized tensor of shape (batch_size, channels, time, height, width)

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape (batch_size, channels, time, height, width)
        """
        return self.decoder(z_q)


class VQVAE(nn.Module):
    """
    Complete VQ-VAE model that combines tokenizer and detokenizer.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    hidden_dim : int
        Hidden dimension for encoder/decoder
    codebook_size : int
        Number of embeddings in the codebook
    codebook_dim : int
        Dimension of each embedding vector
    commitment_cost : float
        Commitment cost for the codebook loss
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        codebook_size: int = 512,
        codebook_dim: int = 64,
        commitment_cost: float = 0.25,
        derivatives: bool = False,
    ):
        super().__init__()
        num_fields = in_channels
        input_channels = in_channels
        if derivatives:
            self.derivatives = FiniteDifference(
                num_channels=num_fields, filter_1d="2nd"
            )
            input_channels = (
                num_fields * 4
            )  # more input channels due to derivatives dt, dx ,dy
        else:
            self.derivatives = None

        # Rearrangement layers
        self.to_conv = Rearrange("b t h w c -> b c t h w")
        self.from_conv = Rearrange("b c t h w -> b t h w c")

        self.tokenizer = VQVAETokenizer(
            in_channels=input_channels,
            hidden_dim=hidden_dim,
        )

        # Projection layers for codebook
        self.to_codebook = nn.Conv3d(hidden_dim, codebook_dim, kernel_size=1)
        self.from_codebook = nn.Conv3d(codebook_dim, hidden_dim, kernel_size=1)

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
        )

        self.detokenizer = VQVAEDetokenizer(
            out_channels=num_fields,
            hidden_dim=hidden_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor.
        """
        x = self.to_conv(x)
        z = self.tokenizer(x)
        z = self.to_codebook(z)
        z_q, loss, indices = self.quantizer(z)
        return z_q, loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode the quantized tensor.
        """
        z_q = self.from_codebook(z_q)
        x_recon = self.detokenizer(z_q)
        x_recon = self.from_conv(x_recon)
        return x_recon

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time, height, width, channels)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Reconstructed tensor
            - Codebook loss
            - Encoding indices
        """
        if self.derivatives:
            dt, dh, dw = self.derivatives(x)
            x = torch.cat([x, dt, dh, dw], dim=-1)
        # Encode
        z_q, loss, indices = self.encode(x)
        # Decode
        x_recon = self.decode(z_q)

        return x_recon, loss, indices

    def get_3d_indices(
        self,
        indices: torch.Tensor,
        shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Reshape the flattened indices back to 3D format.

        Parameters
        ----------
        indices : torch.Tensor
            Flattened indices from the quantizer
        shape : Tuple[int, int, int, int]
            Shape of the original tensor

        Returns
        -------
        torch.Tensor
            Reshaped indices with shape (batch_size, time//4, height//4, width//4)
        """
        B, T, H, W = shape

        # Use einops to rearrange the flattened indices back to 3D format
        return rearrange(indices, "(b t h w) -> b t h w", b=B, t=T, h=H, w=W)
