import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module that maps continuous inputs to discrete codes.

    Parameters
    ----------
    num_embeddings : int
        Number of embeddings in the codebook
    embedding_dim : int
        Dimension of each embedding vector
    commitment_cost : float
        Commitment cost for the codebook loss
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

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
        z_flat = z.reshape(-1, self.embedding_dim)

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
    VQ-VAE Tokenizer that encodes input into discrete latent codes.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    hidden_dim : int
        Hidden dimension for the encoder
    num_embeddings : int
        Number of embeddings in the codebook
    embedding_dim : int
        Dimension of each embedding vector
    commitment_cost : float
        Commitment cost for the codebook loss
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, embedding_dim, 1),
        )

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE tokenizer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time, height, width, channels)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Quantized tensor
            - Codebook loss
            - Encoding indices
        """
        # Rearrange for Conv3d
        x = x.permute(0, 4, 1, 2, 3)

        # Encode
        z = self.encoder(x)

        # Rearrange back to original format
        z = z.permute(0, 2, 3, 4, 1)

        # Quantize
        z_q, loss, indices = self.quantizer(z)

        return z_q, loss, indices


class VQVAEDetokenizer(nn.Module):
    """
    VQ-VAE Detokenizer that decodes discrete latent codes back to the original space.

    Parameters
    ----------
    out_channels : int
        Number of output channels
    hidden_dim : int
        Hidden dimension for the decoder
    embedding_dim : int
        Dimension of each embedding vector
    """

    def __init__(
        self,
        out_channels: int,
        hidden_dim: int,
        embedding_dim: int = 64,
    ):
        super().__init__()

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dim, out_channels, 1),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VQ-VAE detokenizer.

        Parameters
        ----------
        z_q : torch.Tensor
            Quantized tensor of shape (batch_size, time, height, width, embedding_dim)

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape (batch_size, time, height, width, channels)
        """
        # Rearrange for ConvTranspose3d
        z_q = z_q.permute(0, 4, 1, 2, 3)

        # Decode
        x_recon = self.decoder(z_q)

        # Rearrange back to original format
        x_recon = x_recon.permute(0, 2, 3, 4, 1)

        return x_recon


class VQVAE(nn.Module):
    """
    Complete VQ-VAE model that combines tokenizer and detokenizer.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    hidden_dim : int
        Hidden dimension for encoder/decoder
    num_embeddings : int
        Number of embeddings in the codebook
    embedding_dim : int
        Dimension of each embedding vector
    commitment_cost : float
        Commitment cost for the codebook loss
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.tokenizer = VQVAETokenizer(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.detokenizer = VQVAEDetokenizer(
            out_channels=in_channels, hidden_dim=hidden_dim, embedding_dim=embedding_dim
        )

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
        z_q, loss, indices = self.tokenizer(x)
        x_recon = self.detokenizer(z_q)

        return x_recon, loss, indices
