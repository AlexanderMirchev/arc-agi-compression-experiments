from typing import Callable, Optional
from torch import Tensor
from torch.utils.data import DataLoader

from data.pair_dataset import PairDataset
from models.abstract_vae import AbstractVAE

class Pipeline:
    def __init__(
        self,
        model: AbstractVAE,
        preprocess_fn: Callable[[Tensor], Tensor],
        postprocess_fn: Callable[[Tensor, Tensor], Tensor], # accepts expected dimensions
        compress_fn: Optional[Callable[[Tensor], Tensor]] = None,
        decompress_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        if not isinstance(model, AbstractVAE):
            raise TypeError("model must be an instance of AbstractVAE")
        
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.compress_fn = compress_fn
        self.decompress_fn = decompress_fn

    def create_data_loader(self, pair_data, batch_size=32, shuffle=True):
        dataset = PairDataset(pair_data, self.preprocess_fn, self.compress_fn)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def encode(self, x):
        return self.model.encode(x)
    
    def reparameterize(self, mu, logvar):
        return self.model.reparameterize(mu, logvar)
    
    def decode(self, z):        
        x = self.model.decode(z)
        
        if self.decompress_fn:
            z = self.decompress_fn(z)

        return x
    
    def forward(self, x):
        return self.model.forward(x)
    
    def reconstruct(self, x):
        return self.model.reconstruct(x)

    

    