import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import json
from models.grid_augmentations import GridAugmenter

class VAE(nn.Module):
    def __init__(self, in_channels=11, latent_dim=16, base_filters=32, num_classes=11, min_spatial_size=4):
        super(VAE, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.min_spatial_size = min_spatial_size

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=3, stride=1, padding=1)
        self.enc_conv3 = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, stride=1, padding=1)
        
        # Latent space projection
        self.mu_conv = nn.Conv2d(base_filters*4, latent_dim, kernel_size=1)
        self.logvar_conv = nn.Conv2d(base_filters*4, latent_dim, kernel_size=1)
        
        # Decoder - more capacity to reconstruct details
        self.dec_conv1 = nn.Conv2d(latent_dim, base_filters*4, kernel_size=1)
        self.dec_conv2 = nn.Conv2d(base_filters*4, base_filters*2, kernel_size=3, stride=1, padding=1)
        self.dec_conv3 = nn.Conv2d(base_filters*2, base_filters, kernel_size=3, stride=1, padding=1)
        
        # Final output layer - using 1x1 convolution for class prediction
        self.output_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def encode(self, x):
        # Get input shape for later use
        input_h, input_w = x.size(2), x.size(3)
        
        # First encoding layer
        x = F.leaky_relu(self.enc_conv1(x), 0.2)
        # Use adaptive pooling for flexibility with grid sizes
        x = F.adaptive_avg_pool2d(x, (max(2, input_h//2), max(2, input_w//2)))
        
        # Second encoding layer
        x = F.leaky_relu(self.enc_conv2(x), 0.2)
        x = self.dropout(x)  # Apply dropout for regularization
        x = F.adaptive_avg_pool2d(x, (max(1, input_h//4), max(1, input_w//4)))
        
        # Third encoding layer
        x = F.leaky_relu(self.enc_conv3(x), 0.2)
        
        # Get latent parameters with slightly reduced variance
        mu = self.mu_conv(x)
        logvar = torch.clamp(self.logvar_conv(x), min=-4.0, max=4.0)  # Prevent extreme variances
        
        return mu, logvar, (input_h, input_w)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, original_size):
        # Start decoding
        x = F.leaky_relu(self.dec_conv1(z), 0.2)
        
        # Calculate intermediate sizes for upscaling
        h_mid = max(2, original_size[0]//2)
        w_mid = max(2, original_size[1]//2)
        
        # First upscale - using bilinear for smoother results on small grids
        x = F.interpolate(x, size=(h_mid, w_mid), mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.dec_conv2(x), 0.2)
        x = self.dropout(x)  # Apply dropout
        
        # Final upscale to original size
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.dec_conv3(x), 0.2)
        
        # Get output logits
        logits = self.output_conv(x)
        
        return logits
    
    def forward(self, x):
        # Encode and get latent representation
        mu, logvar, original_size = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z, original_size)
        
        # Ensure output has same spatial dimensions as input
        if x_recon.size(2) != x.size(2) or x_recon.size(3) != x.size(3):
            x_recon = F.interpolate(x_recon, size=(x.size(2), x.size(3)), mode='nearest')
        
        return x_recon, mu, logvar
    
    def sample(self, num_samples=1, device='cpu', size=(8, 8)):
        """Generate samples from the latent space"""
        # Calculate appropriate latent size based on the requested output size
        latent_h = max(1, size[0] // 4)
        latent_w = max(1, size[1] // 4)
        
        # Create random latent variable
        z = torch.randn(num_samples, self.latent_dim, latent_h, latent_w).to(device)
        
        # Decode
        logits = self.decode(z, size)
        
        # Apply softmax to get class probabilities
        probs = F.softmax(logits, dim=1)
        
        # Sample from the categorical distribution
        samples = torch.multinomial(probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes), 1)
        samples = samples.reshape(num_samples, size[0], size[1])
        
        return samples
    
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

# Loss function for categorical VAE
def vae_loss(recon_logits, x, mu, logvar, beta=1.0):
    # Convert one-hot x to class indices for cross entropy
    target = torch.argmax(x, dim=1)
    
    # Reconstruction loss: categorical cross entropy
    CE = F.cross_entropy(recon_logits, target, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return CE + beta * KLD

def get_training_data(filepath):
    json_files = glob.glob(os.path.join(filepath, "*.json"))
    
    train_data = []

    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            train_data.extend([matrix for pair in data["train"] for matrix in pair.values()])

    return train_data

# Function to train the model
def train(model, train_loader, optimizer, device, beta=1.0, epoch=0):
    model.train()
    train_loss = 0
    num_batches = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        try:
            # Forward pass
            recon_batch, mu, logvar = model(data)
            
            # Compute loss
            loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            # Backward pass
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            num_batches += 1
            
            # if batch_idx % 10 == 0:
                # print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
                # print(f'Input shape: {data.shape}, Output shape: {recon_batch.shape}')
        
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print(f"Input shape: {data.shape}")
            continue
    
    avg_loss = train_loss / max(1, num_batches)
    print(f'Epoch: {epoch}, Average loss: {avg_loss:.4f}, Batches processed: {num_batches}')
    return avg_loss

class GridDataset(Dataset):
    def __init__(self, grids, num_classes=11, target_size=(30, 30), augment=True):
        self.grids = grids
        self.num_classes = num_classes
        self.target_height, self.target_width = target_size
        initial_size = len(self.grids)
        if augment:
            self._apply_augmentations()

        print(f"Dataset size after augmentation: {initial_size} -> {len(self.grids)}")

    def _apply_augmentations(self):
        """Apply augmentations to all grids in the dataset"""
        augmenter = GridAugmenter(num_colors=self.num_classes - 1) # one class is for the backgrounds
        
        total_augmentations = []
        # for grid in self.grids:
        for grid in self.grids:
            augmented_grids = augmenter.augment_grid(grid)
            total_augmentations.extend(augmented_grids)
                
        self.grids = total_augmentations

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, index):
        grid = self.grids[index]
        grid_tensor = torch.tensor(grid, dtype=torch.long)

        grid_tensor = torch.clamp(grid_tensor, 0, self.num_classes - 2)

        h, w = grid_tensor.shape

        padded_grid = torch.full(
            (self.target_height, self.target_width), 
            fill_value=self.num_classes - 1,  # Padding with class index 10
            dtype=torch.long
        )
        padded_grid[:h, :w] = grid_tensor

        # One-hot encode and permute to (C, H, W)
        one_hot = F.one_hot(padded_grid, num_classes=self.num_classes).permute(2, 0, 1).float()

        return one_hot
    
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
    grids = get_training_data(filepath="data/training")
    print(f"Loaded {len(grids)} grids")
    
    # Create dataset and dataloader
    dataset = GridDataset(grids)
    batch_size = 4
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )

    channels = 11 # 10 channels for the colors, 1 for padding
    
    # Initialize the model
    model = VAE(
        in_channels=channels, 
        num_classes=channels, 
        base_filters=64, 
        latent_dim=32,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train the model
    epochs = 100
    losses = []
    
    for epoch in range(1, epochs + 1):
        try:
            loss = train(model, train_loader, optimizer, device, beta=0.05, epoch=epoch)
            losses.append(loss)
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'checkpoints/vae_checkpoint_epoch.pt')
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            continue
    
    # Plot training loss
    # if losses:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(losses)
    #     plt.title('Training Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.tight_layout()
    #     plt.show()


if __name__ == "__main__":
    main()