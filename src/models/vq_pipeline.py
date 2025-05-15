import torch
import torch.optim as optim
import torch.nn.functional as F

from data.augmentations import augment_grid_pairs
from data.scale_processing import scaling, reverse_scaling
from models.convolutional_vqvae import ConvolutionalVQVAE
from models.fully_connected_vae import FullyConnectedVAE
from models.pipeline import Pipeline
from utils.load_data import get_grids
from utils.train_vae import vae_loss_mse, train, validate
from utils.view import plot_losses

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

def get_compression_functions(saved_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvolutionalVQVAE(
        in_channels=10, 
        starting_filters=64, 
        num_embeddings=256,
        embedding_dim=64,
        commitment_cost=0.1,
    ).to(device)

    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    def compress_fn(grid):
        # print("c", model)
        model.eval()

        with torch.no_grad():
            grid = grid.to(device)
            grid = grid.unsqueeze(0)  # Add batch dimension
            z_e = model.encode(grid)
            z_q, _, _ = model.quantize(z_e)
            z_q = z_q.view(z_q.size(0), -1) # to linear
            return z_q
        
    def decompress_fn(z_e):
        # print("d", model)
        model.eval()

        with torch.no_grad():
            # z_e = z_e.to(device)
            z_q = z_e.view(1, 64, 6, 6) 
            z_q, _, _ = model.quantize(z_q)
            recon_batch = model.decode(z_q).squeeze(0)
            return recon_batch
        
    return compress_fn, decompress_fn

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    training_data, validation_data = get_grids(filepath="data/training")
    
    training_grid_pairs = [pair for task in training_data.values() for pairs in task.values() for pair in pairs]
    validation_grid_pairs = [pair for task in validation_data.values() for pairs in task.values() for pair in pairs]

    input_dim = 6*6*64
    model = FullyConnectedVAE(
        input_dim=input_dim,
        hidden_dim=1024,
        latent_dim=64
    ).to(device)
    
    print(f"Model architecture: {model}")

    training_grid_pairs = augment_grid_pairs(training_grid_pairs, target_count=15000)
    # print(f"Loaded {len(training_grid_pairs)} (after augmentation) training grid pairs and {len(validation_grid_pairs)} validation grid pairs.")

    compress_fn, decompress_fn = get_compression_functions('checkpoints/conv_vqvae_6x6x64_b001.pt')

    pipeline = Pipeline(
        model=model,
        preprocess_fn=preprocess_grid,
        postprocess_fn=postprocess_grid,
        compress_fn=compress_fn,
        decompress_fn=decompress_fn,
    )

    batch_size = 16
    train_loader = pipeline.create_data_loader(training_grid_pairs, batch_size=batch_size, shuffle=True)
    val_loader = pipeline.create_data_loader(validation_grid_pairs, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    max_epochs = 100
    patience = 5
    min_improvement = 1e-3
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(1, max_epochs + 1):
        try:
            beta = 2.0

            train_loss = train(model, 
                               train_loader, 
                               loss_fn=vae_loss_mse,
                               optimizer=optimizer, 
                               device=device, 
                               beta=beta, 
                               epoch=epoch)
            val_loss = validate(model, 
                                val_loader, 
                                loss_fn=vae_loss_mse,
                                device=device, 
                                beta=beta, 
                                epoch=epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss + min_improvement < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'checkpoints/combineishun.pt')
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
