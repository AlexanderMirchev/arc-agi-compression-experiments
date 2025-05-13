import joblib
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA

from data.scale_processing import scaling, reverse_scaling
from utils.load_data import get_grids

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

# def get_compression_functions(saved_model_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ppca = PCA(n_components=50, svd_solver='full')
#     model = ConvolutionalVQVAE(
#         in_channels=10, 
#         starting_filters=64, 
#         num_embeddings=256,
#         embedding_dim=64,
#         commitment_cost=0.1
#     ).to(device)

#     checkpoint = torch.load(saved_model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])

#     def compress_fn(grid):
#         grid = grid.to(device)
#         grid = grid.unsqueeze(0)  # Add batch dimension
#         z_e = model.encode(grid)
#         z_q, _, _ = model.quantize(z_e)
#         z_q = z_q.squeeze(0).view(z_q.size(0), -1) # to linear
#         return z_q
    
#     def decompress_fn(z_e):
#         z_q = z_e.to(device)
#         z_q = z_q.view(1, 64, 6, 6) 
#         z_q2, _, _ = model.quantize(z_q) # to actually map them to quantized vectors
#         recon_batch = model.decode(z_q2)
#         return recon_batch
#     return compress_fn, decompress_fn

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    training_data, validation_data = get_grids(filepath="data/training")
    
    training_grid_pairs = [pair for task in training_data.values() for pairs in task.values() for pair in pairs]
    validation_grid_pairs = [pair for task in validation_data.values() for pairs in task.values() for pair in pairs]

    grids = [preprocess_grid(grid) for pair in training_grid_pairs for grid in pair]
    print(grids[:1])
    components = 50
    pca = PCA(n_components=components, svd_solver='full')
    pca.fit(grids)  # learns the principal components from the training data

    joblib.dump(pca, f'checkpoints/pca_{components}.joblib')

if __name__ == "__main__":
    main()
