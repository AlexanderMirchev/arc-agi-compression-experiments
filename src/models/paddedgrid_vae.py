import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.arc_dataset import GridDataset
from data.scale_processing import scaling
from utils.load_data import get_grids
from utils.train_vae import train, validate

class VAE(nn.Module):
    def __init__(self, in_channels=10, num_filters=128, latent_dim=32, l2_penalty=0.2):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.l2_penalty = l2_penalty

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=0),  # 11x30x30 -> 128x14x14
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1),  # 128x14x14 -> 128x7x7
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=3, padding=0),  # 128x7x7 -> 128x2x2
            nn.ReLU(True)
        )

        self.flatten_dim = num_filters * 2 * 2
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4, stride=3, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_filters, in_channels, kernel_size=4, stride=2, padding=0),
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(z.size(0), -1, 2, 2)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def reconstruct(self, x):
        """Reconstruct an input by passing it through the encoder and decoder"""
        with torch.no_grad():
            # Get reconstruction
            recon_logits, _, _ = self.forward(x)
            
            # Apply softmax to get probabilities
            probs = F.softmax(recon_logits, dim=1)
            
            # Get the most likely class for each position
            reconstruction = torch.argmax(probs, dim=1)
            
            return reconstruction

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
    
# Visualization function
def visualize_grid(grid, title="Grid"):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the data
    training_grids, validation_grids = get_grids(filepath="data/training")
    print(f"Loaded {len(training_grids)} training grids and {len(validation_grids)} validation grids.")
    
    batch_size = 4

    def normalize_fn(x, hw):
        h, w = hw
        # Normalize the grid to the target size
        return scaling(x, height=h, width=w, direction='norm')
    
    training_dataset = GridDataset(training_grids, normalize_fn=normalize_fn, augment=True)
    
    train_loader = DataLoader(
        training_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )

    validation_dataset = GridDataset(validation_grids, normalize_fn=normalize_fn, augment=False)
    val_loader = DataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    channels = 10
    
    # Initialize the model
    model = VAE(
        in_channels=channels, 
        num_filters=128, 
        latent_dim=64,
        l2_penalty=1e-5
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=model.l2_penalty)
    
    # Train the model
    epochs = 100
    
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        try:
            beta = 0.1

            train_loss = train(model, train_loader, optimizer, device, beta=beta, epoch=epoch)
            val_loss = validate(model, val_loader, device, beta=beta, epoch=epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save checkpoint
            if epoch % 10 == 0:
                plot_losses(train_losses, val_losses)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'checkpoints/vae_checkpoint_epoch.pt')
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            continue

if __name__ == "__main__":
    main()