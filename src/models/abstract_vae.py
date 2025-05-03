import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractVAE(nn.Module, ABC):
    def __init__(self, in_channels, latent_dim):
        super(AbstractVAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x):
        """Encode input into latent mean and log variance"""
        pass

    def reparameterize(self, mu, logvar):
        """Sample latent vector from mean and log variance using the reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @abstractmethod
    def decode(self, z):
        """Decode latent vector back to input space"""
        pass

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def reconstruct(self, x):
        """Reconstruct input by encoding and decoding"""
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            return recon