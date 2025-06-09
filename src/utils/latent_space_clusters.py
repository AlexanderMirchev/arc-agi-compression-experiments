import torch
import numpy as np

from utils.latent_space import latent_to_grid

def sample_nearby(z_values, num_samples_per_point=50, std_dev=0.1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    z_values = np.asarray(z_values)
    N, D = z_values.shape
    noise = np.random.normal(scale=std_dev, size=(N, num_samples_per_point, D))
    nearby_samples = z_values[:, np.newaxis, :] + noise  # Shape: (N, num_samples_per_point, D)
    sampled_points = nearby_samples.reshape(-1, D)
    combined_points = np.vstack([z_values, sampled_points])
    
    return combined_points


def find_matching_example_grid_id(decoded_grid, example_grids):
    for idx, grid in enumerate(example_grids):
        if matches_example_grid(decoded_grid, grid):
            return idx + 1
    return -1  # Not found

def matches_example_grid(decoded_grid, example_grid):
    decoded_grid = np.array(decoded_grid)
    grid = np.array(example_grid)

    return (decoded_grid.shape == grid.shape and (decoded_grid == grid).mean() == 1.0)
    
def prepare_latent_data(z_example_grids, z_all):
    z_all_np = z_all
    z_example_np = z_example_grids.detach().cpu().numpy()
    z_samples = sample_nearby(z_example_np, num_samples_per_point=500, std_dev=0.1, random_seed=42)
    return z_all_np, z_example_np, z_samples


def label_decoded_samples_single_grid(z_samples, example_grid, label, pipeline, model_type):
    matched_z = []
    for z in z_samples:
        z_tensor = torch.from_numpy(z).float()
        # issue here - doesn't work with different example grid sizes
        decoded = latent_to_grid(pipeline, z_tensor, example_grid, model_type)
        p = matches_example_grid(decoded, example_grid)
        print('label_decoded', label, p)
        if p:
            matched_z.append(z)
            
    return np.array(matched_z), label
