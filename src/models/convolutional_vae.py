import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from data.augmentations import augment_grid_pairs
from data.scale_processing import scaling, reverse_scaling
from models.abstract_vae import AbstractVAE
from models.pipeline import Pipeline

from utils.load_data import get_grids
from utils.train_vae import train, validate

class ConvolutionalVAE(AbstractVAE):
    def __init__(self, in_channels=10, num_filters=128, feature_dim=[2, 2], latent_dim=128):
        super(ConvolutionalVAE, self).__init__(in_channels, latent_dim)

        self.num_filters = num_filters
        self.feature_dim = feature_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2),  # -> 128x14x14
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2),  # -> 128x7x7
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2),  # -> 128x2x2
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
        )

        self.flatten_dim = num_filters * np.prod(feature_dim)

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(num_filters, in_channels, kernel_size=4, stride=2),
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
        x = x.view(z.size(0), self.num_filters, *self.feature_dim)
        x = self.decoder(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def preprocess_grid(grid):
    grid = scaling(grid, height=30, width=30, direction='norm')

    num_classes = 10
    grid_tensor = torch.tensor(grid, dtype=torch.long)
    grid_tensor = torch.clamp(grid_tensor, 0, num_classes - 1)
    one_hot = F.one_hot(grid_tensor, num_classes=10).permute(2, 0, 1).float()
    return one_hot

def postprocess_grid(grid, grid_original):
    grid = torch.argmax(F.softmax(grid, dim=1), dim=1).squeeze(0).numpy()
    return reverse_scaling(grid_original, grid)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # Load the data
    training_data, validation_data = get_grids(filepath="data/training")

    # print([pair for task in training_data.values() for pairs in task.values() for pair in pairs][0])
    training_grid_pairs = [pair for task in training_data.values() for pairs in task.values() for pair in pairs]
    validation_grid_pairs = [pair for task in validation_data.values() for pairs in task.values() for pair in pairs]

    model = ConvolutionalVAE(
        in_channels=10, 
        num_filters=128, 
        latent_dim=128,
        feature_dim=[2, 2]
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
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    max_epochs = 100
    patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    for epoch in range(1, max_epochs + 1):
        try:
            beta = 0.1

            train_loss = train(model, train_loader, optimizer, device, beta=beta, epoch=epoch)
            val_loss = validate(model, val_loader, device, beta=beta, epoch=epoch)
            
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