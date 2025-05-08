import numpy as np
import pywt
import torch
import torch.optim as optim

from data.augmentations import augment_grid_pairs
from data.scale_processing import scaling, reverse_scaling
from models.pipeline import Pipeline
from models.fully_connected_vae import FullyConnectedVAE
from utils.load_data import get_grids
from utils.train_vae import vae_loss, train, validate
from utils.view import plot_losses

def preprocess_grid(grid):
    grid = scaling(grid, height=30, width=30, direction='norm')

    num_classes = 10
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    return torch.clamp(grid_tensor, 0, num_classes - 1)

def postprocess_grid(grid, grid_original):
    return reverse_scaling(grid_original, grid)

def compress_wavelet(grid):
    coeffs2 = pywt.dwt2(grid, 'haar')
    cA, (cH, cV, cD) = coeffs2

    threshold = 0.1
    cH = np.where(np.abs(cH) < threshold, 0, cH)
    cV = np.where(np.abs(cV) < threshold, 0, cV)
    cD = np.where(np.abs(cD) < threshold, 0, cD)
    
    feature_vector = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
    return feature_tensor

def decompress_wavelet(feature_tensor, grid_shape=(30, 30)):
    feature_vector = feature_tensor.detach().numpy().squeeze()
    
    cA_size = grid_shape[0] * grid_shape[1] // 4  
    cH_size = cV_size = cA_size 
    
    cA = feature_vector[:cA_size].reshape(grid_shape[0] // 2, grid_shape[1] // 2)
    cH = feature_vector[cA_size:cA_size + cH_size].reshape(grid_shape[0] // 2, grid_shape[1] // 2)
    cV = feature_vector[cA_size + cH_size:cA_size + cH_size + cV_size].reshape(grid_shape[0] // 2, grid_shape[1] // 2)
    cD = feature_vector[cA_size + cH_size + cV_size:].reshape(grid_shape[0] // 2, grid_shape[1] // 2)
    
    coeffs2 = cA, (cH, cV, cD)
    grid_reconstructed = pywt.idwt2(coeffs2, 'haar')
    
    return grid_reconstructed
def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # Load the data
    training_data, validation_data = get_grids(filepath="data/evaluation")
    # print(training_data.keys())
    key='b7fb29bc'
    # print([pair for task in training_data.values() for pairs in task.values() for pair in pairs][0])
    training_grid_pairs = [([pair for pairs in training_data[key].values() for pair in pairs])[0]]
    
    # training_grid_pairs = [pair for task in training_data.values() for pairs in task.values() for pair in pairs]
    # validation_grid_pairs = [pair for task in validation_data.values() for pairs in task.values() for pair in pairs]

    model = FullyConnectedVAE(
        input_dim=900,  # 30x30 grid flattened
        hidden_dim=512,
        latent_dim=64
    ).to(device)
    
    print(f"Model architecture: {model}")

    # training_grid_pairs = augment_grid_pairs(training_grid_pairs, target_count=5000)
    # print(f"Loaded {len(training_grid_pairs)} (after augmentation) training grid pairs and {len(validation_grid_pairs)} validation grid pairs.")

    pipeline = Pipeline(
        model=model,
        preprocess_fn=preprocess_grid,
        postprocess_fn=postprocess_grid,
        compress_fn=compress_wavelet,
        decompress_fn=decompress_wavelet,
    )

    batch_size = 1
    train_loader = pipeline.create_data_loader(training_grid_pairs, batch_size=batch_size, shuffle=True)
    # val_loader = pipeline.create_data_loader(validation_grid_pairs, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    max_epochs = 1000
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    for epoch in range(1, max_epochs + 1):
        try:
            beta = 0

            train_loss = train(model, 
                               train_loader, 
                               loss_fn=vae_loss,
                               optimizer=optimizer, 
                               device=device, 
                               beta=beta, 
                               epoch=epoch)
            # val_loss = validate(model, 
            #                     val_loader, 
            #                     loss_fn=vae_loss,
            #                     device=device, 
            #                     beta=beta, 
            #                     epoch=epoch)
            
            train_losses.append(train_loss)
            # val_losses.append(val_loss)

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                # 'val_loss': val_loss,
            }, 'checkpoints/overfit.pt')

            # else:
            #     epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation loss. Best: {best_val_loss}")
                break
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            continue
    
    # plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()