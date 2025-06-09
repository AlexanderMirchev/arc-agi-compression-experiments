import torch
import numpy as np
from models.pipeline import Pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def grid_to_latent(pipeline: Pipeline, grid, model_type="vq", only_preprocess=False):
    grid = pipeline.preprocess_and_compress(grid)
    z = grid
    if only_preprocess:
        return z, grid.size()
    
    if model_type == 'vq':
        z = pipeline.encode(grid.unsqueeze(0).to(device))
    else:
        z, _ = pipeline.encode(grid.unsqueeze(0).to(device))
    
    z_size = z.size()
    z_flat = z.view(z.size(0), -1)

    return z_flat, z_size

def latent_to_grid(pipeline: Pipeline, z, expected_output, model_type="vq", only_postprocess=False):
    if only_postprocess:
        return pipeline.decompress_and_postprocess(z, expected_output)
    decoded = z

    if model_type == 'vq':
        z_quantized, _, _ = pipeline.model.quantize(z.to(device))
        decoded = pipeline.decode(z_quantized)
    else:
        decoded = pipeline.decode(z.unsqueeze(0).to(device))
        
    return pipeline.decompress_and_postprocess(decoded.squeeze(0), expected_output)

def process_train_pairs(pipeline: Pipeline, train_pairs, model_type="vq"):
    z_inputs = []
    z_diffs = []
    z_size = None
    for input, output in train_pairs:
        z_input, z_size = grid_to_latent(pipeline, input, model_type)
        z_output, _ = grid_to_latent(pipeline, output, model_type)
        z_inputs.append(z_input)
        z_diffs.append(z_output - z_input)
    
    return z_inputs, z_diffs, z_size

def reconstruct_grid(pipeline: Pipeline, grid, model_type="vq", first_layer_only=False):
    pipeline.model_eval()
    with torch.no_grad():
        z, z_size = grid_to_latent(pipeline, grid, model_type, only_preprocess=first_layer_only)
        z = z.view(*z_size)
        reconstructed_grid = latent_to_grid(pipeline, z, grid, model_type, only_postprocess=first_layer_only)
    return reconstructed_grid

def extract_single_object_grids(grid):
    unique_digits = np.unique(grid)
    unique_digits = unique_digits[unique_digits != 0]

    object_grids = []
    for digit in unique_digits:
        mask = (grid == digit).astype(int) * digit
        object_grids.append(mask)

    return object_grids