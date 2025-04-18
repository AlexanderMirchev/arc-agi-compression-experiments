import torch
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from data.augmentations import augment_grids
from data.scale_processing import scaling
from models.fully_connected_vae import FullyConnectedVAE
from utils.load_data import get_grids
from utils.train_vae import train, validate

class PCACompressor:
    """PCA-based compression for ARC grids"""
    
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca = None
        
    def fit(self, grids):
        """Fit PCA to a collection of grids"""
        # Flatten grids for PCA
        flat_grids = np.array([grid.flatten() for grid in grids])
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(flat_grids)
        return self
        
    def compress(self, grids):
        """Transform grids to PCA components"""
        if self.pca is None:
            raise ValueError("PCA model not fitted yet")
        
        flat_grids = np.array([grid.flatten() for grid in grids])
        return np.array(self.pca.transform(flat_grids))
        
    def decompress(self, components):
        """Transform PCA components back to approximate grids"""
        if self.pca is None:
            raise ValueError("PCA model not fitted yet")
            
        reconstructed = self.pca.inverse_transform(components)
        
        # Here you'd typically reshape back to original grid shape
        # For demonstration, we'll assume all grids are the same shape
        grid_shape = (10, 10)  # Update this based on your data
        return [rec.reshape(grid_shape) for rec in reconstructed]

def visualize_latent_space(model, data_loader, device='cpu'):
    """Create a visualization of the latent space using t-SNE"""
    from sklearn.manifold import TSNE
    
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = data[0].to(device)
            mu, _ = model.encode(data)
            latent_vectors.append(mu.cpu().numpy())
    
    # Combine all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.6)
    plt.title('t-SNE Visualization of VAE Latent Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()
    
    return latent_tsne


def reconstruct_samples(model, data_loader, compressor, device='cpu', num_samples=5):
    """Reconstruct a few samples and show original vs reconstruction"""
    model.eval()
    
    # Get some samples
    for batch_idx, data in enumerate(data_loader):
        original_batch = data[0]
        break
    
    with torch.no_grad():
        # Process through VAE
        data = original_batch.to(device)
        recon_batch, _, _ = model(data)
        
        # Get back to numpy for visualization
        original_compressed = original_batch.cpu().numpy()
        reconstructed_compressed = recon_batch.cpu().numpy()
        
        # Decompress back to grid form
        original_grids = compressor.decompress(original_compressed[:num_samples])
        reconstructed_grids = compressor.decompress(reconstructed_compressed[:num_samples])
    
    # Plot original vs reconstructed
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2*num_samples))
    
    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(original_grids[i], cmap='viridis')
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')
        
        # Reconstructed
        axes[i, 1].imshow(reconstructed_grids[i], cmap='viridis')
        axes[i, 1].set_title(f"Reconstructed {i+1}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def latent_space_interpolation(model, sample1, sample2, compressor, device='cpu', steps=10):
    """Interpolate between two points in the latent space"""
    model.eval()
    
    with torch.no_grad():
        # Encode both samples
        sample1_tensor = torch.tensor(sample1, dtype=torch.float32).unsqueeze(0).to(device)
        sample2_tensor = torch.tensor(sample2, dtype=torch.float32).unsqueeze(0).to(device)
        
        mu1, _ = model.encode(sample1_tensor)
        mu2, _ = model.encode(sample2_tensor)
        
        # Create interpolations
        interpolations = []
        
        for alpha in np.linspace(0, 1, steps):
            # Interpolated point
            z = alpha * mu2 + (1 - alpha) * mu1
            
            # Decode
            decoded = model.decode(z)
            interpolations.append(decoded.cpu().numpy()[0])
        
        # Convert all to grids
        grid_interpolations = compressor.decompress(interpolations)
    
    # Plot
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
    
    for i, grid in enumerate(grid_interpolations):
        axes[i].imshow(grid, cmap='viridis')
        axes[i].set_title(f"{i/(steps-1):.1f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_latent_dimensions(model, compressor, device='cpu', dim_range=(-3, 3), steps=7):
    """See how each latent dimension affects the output"""
    model.eval()
    
    # Use zero vector as base point
    base_z = torch.zeros(1, model.latent_dim).to(device)
    grid_variations = []
    
    with torch.no_grad():
        # For each dimension
        for dim in range(model.latent_dim):
            dimension_grids = []
            
            # Try different values
            for val in np.linspace(dim_range[0], dim_range[1], steps):
                # Create latent vector with just this dimension modified
                z = base_z.clone()
                z[0, dim] = val
                
                # Decode
                decoded = model.decode(z)
                decoded_np = decoded.cpu().numpy()[0]
                grid = compressor.decompress([decoded_np])[0]
                dimension_grids.append(grid)
            
            grid_variations.append(dimension_grids)
    
    # Plot
    fig, axes = plt.subplots(model.latent_dim, steps, figsize=(steps*2, model.latent_dim*2))
    
    for i in range(model.latent_dim):
        for j in range(steps):
            axes[i, j].imshow(grid_variations[i][j], cmap='viridis')
            axes[i, j].set_title(f"z[{i}]={np.linspace(dim_range[0], dim_range[1], steps)[j]:.1f}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage
def main():    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the data
    training_grids, validation_grids = get_grids(filepath="data/training")
    print(f"Loaded {len(training_grids)} training grids and {len(validation_grids)} validation grids.")
    
    initial_training_size = len(training_grids)
    training_grids = augment_grids(training_grids, num_colors=10)
    print(f"Augmentation: {initial_training_size} -> {len(training_grids)}")

    training_grids = np.array([scaling(np.array(grid), height=30, width=30) for grid in training_grids])
    validation_grids = np.array([scaling(np.array(grid), height=30, width=30) for grid in validation_grids])
    
    n_components = 20  # Number of PCA components
    compressor = PCACompressor(n_components=n_components)
    compressor.fit(training_grids)
    compressed_train_data = torch.tensor(compressor.compress(training_grids), dtype=torch.float32)
    compressed_val_data = torch.tensor(compressor.compress(validation_grids), dtype=torch.float32)
    
    train_dataset = TensorDataset(compressed_train_data)
    val_dataset = TensorDataset(compressed_val_data)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = n_components
    hidden_dim = 128
    latent_dim = 10
    
    model = FullyConnectedVAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
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
                # plot_losses(train_losses, val_losses)
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

    
    # Visualize results
    # visualize_latent_space(model, data_loader, device)
    # reconstruct_samples(model, data_loader, compressor, device)
    
    
    # Latent space interpolation (example)
    # if len(data_loader.dataset) >= 2:
    #     sample1 = compressed_data[0]
    #     sample2 = compressed_data[10]
    #     latent_space_interpolation(model, sample1, sample2, compressor, device)
    
    # # Analyze latent dimensions
    # analyze_latent_dimensions(model, compressor, device)


if __name__ == "__main__":
    main()