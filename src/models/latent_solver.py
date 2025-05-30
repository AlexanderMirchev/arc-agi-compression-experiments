import torch.nn as nn

class LatentSolver(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super(LatentSolver, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_in):
        return self.model(z_in)
