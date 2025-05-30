import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from data.augmentations import augment_grid_pairs
from data.scale_processing import scaling, reverse_scaling
from models.abstract_vae import AbstractVAE
from models.pipeline import Pipeline

from utils.load_data import get_grids
from utils.train_vae import vae_loss, train, validate
from utils.view import plot_losses

class ConvolutionalVAEV3(AbstractVAE):
    def __init__(self, in_channels=10, starting_filters=64, feature_dim=[4, 4], latent_dim=128):
        super(ConvolutionalVAEV3, self).__init__()

        self.feature_dim = feature_dim
        self.starting_filters = starting_filters

        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, starting_filters, kernel_size=3, stride=2, padding=1),  # -> 30->15
            nn.BatchNorm2d(starting_filters),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(starting_filters, starting_filters*2, kernel_size=3, stride=2, padding=1),  # -> 15->8
            nn.BatchNorm2d(starting_filters*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(starting_filters*2, starting_filters*4, kernel_size=3, stride=1, padding=1),  # -> 8x8
            nn.BatchNorm2d(starting_filters*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(starting_filters*4, starting_filters*8, kernel_size=3, stride=2, padding=1),  # -> 8->4
            nn.BatchNorm2d(starting_filters*8),
            nn.LeakyReLU(inplace=True),
        )

        self.flatten_dim = starting_filters * 8 * np.prod(feature_dim)

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(starting_filters*8, starting_filters*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(starting_filters*4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(starting_filters*4, starting_filters*2, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(starting_filters*2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(starting_filters*2, starting_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(starting_filters),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(starting_filters, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(z.size(0), self.starting_filters*8, *self.feature_dim)
        x = self.decoder(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x

def preprocess_grid(grid):
    grid = scaling(grid, height=30, width=30, direction='norm')

    num_classes = 10
    grid_tensor = torch.tensor(grid, dtype=torch.long)
    grid_tensor = torch.clamp(grid_tensor, 0, num_classes - 1)
    one_hot = F.one_hot(grid_tensor, num_classes=10).permute(2, 0, 1).float()
    return one_hot

def postprocess_grid(grid, grid_original):
    _, grid = torch.max(grid, dim=0)
    grid = grid.detach().cpu().numpy()
    return reverse_scaling(grid_original, grid)

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    training_data, validation_data = get_grids(filepath="data/training")

    training_grid_pairs = [pair for task in training_data.values() for pairs in task.values() for pair in pairs]
    validation_grid_pairs = [pair for task in validation_data.values() for pairs in task.values() for pair in pairs]

    model = ConvolutionalVAEV3(
        in_channels=10, 
        starting_filters=64, 
        latent_dim=256,
        feature_dim=[4, 4]
    ).to(device)
    
    print(f"Model architecture: {model}")

    training_grid_pairs = augment_grid_pairs(training_grid_pairs, target_count=15000)
    print(f"Loaded {len(training_grid_pairs)} (after augmentation) training grid pairs and {len(validation_grid_pairs)} validation grid pairs.")

    pipeline = Pipeline(
        model=model,
        preprocess_fn=preprocess_grid,
        postprocess_fn=postprocess_grid,
    )

    batch_size = 16
    train_loader = pipeline.create_data_loader(training_grid_pairs, batch_size=batch_size, shuffle=True)
    val_loader = pipeline.create_data_loader(validation_grid_pairs, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    max_epochs = 100
    patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    for epoch in range(1, max_epochs + 1):
        try:
            beta = 5.0
            train_loss = train(model, 
                                train_loader, 
                                loss_fn=vae_loss, 
                                optimizer=optimizer, 
                                device=device, 
                                beta=beta, 
                                epoch=epoch)
            
            val_loss = validate(model, 
                                val_loader, 
                                loss_fn=vae_loss,
                                device=device, 
                                beta=beta, 
                                epoch=epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'checkpoints/conv_vae_batchnorm_epoch.pt')

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation loss. Best: {best_val_loss}")
                break
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            continue
    
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()