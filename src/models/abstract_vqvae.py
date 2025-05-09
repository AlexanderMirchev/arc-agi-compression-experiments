import torch
from abc import abstractmethod
from models.abstract_vae import AbstractVAE

class AbstractVQVAE(AbstractVAE):
    def __init__(self):
        super(AbstractVQVAE, self).__init__()
    
    @abstractmethod
    def encode(self, x):
        """Encode input into discrete latent representation"""
        pass
    
    # Override the reparameterize method since VQ-VAEs don't use it
    def reparameterize(self, *args):
        """VQ-VAEs don't use the reparameterization trick"""
        raise NotImplementedError("VQ-VAEs don't use the reparameterization trick")
    
    @abstractmethod
    def decode(self, z_q):
        """Decode quantized latent vector back to input space"""
        pass
    
    def forward(self, x):
        z_e = self.encode(x) 
        z_q, vq_loss, perplexity = self.quantize(z_e) 
        recon_x = self.decode(z_q)
        return recon_x, vq_loss, z_e, z_q
    
    @abstractmethod
    def quantize(self, z_e):
        """Quantize the encoder output to the nearest codebook vectors"""
        pass
    
    def reconstruct(self, x):
        """Reconstruct input by encoding, quantizing and decoding"""
        with torch.no_grad():
            recon, _, _, _ = self.forward(x)
            return recon