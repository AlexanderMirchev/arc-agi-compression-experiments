import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

from models.pipeline import Pipeline
from utils.latent_space import grid_to_latent, latent_to_grid

def extract_diff(z_diffs, z_inputs, z_test_input, comp="average"):
    if comp == 'average':
        return np.mean(z_diffs, axis=0)
    elif comp == 'scaled_euc':
        support_dists = [np.linalg.norm(support_z - z_test_input) for support_z in z_inputs]
        euc_weights = softmax(-np.array(support_dists))
        return np.sum([w * z for w, z in zip(euc_weights, z_diffs)], axis=0)
    elif comp == 'scaled_cos':
        z_inputs = np.array(z_inputs)
        if z_inputs.ndim > 2:
        # Flatten all dimensions after the first one
            z_inputs_2d = z_inputs.reshape(z_inputs.shape[0], -1)
        else:
            z_inputs_2d = z_inputs
            
        if z_test_input.ndim > 1:
            z_test_input_2d = z_test_input.reshape(1, -1)
        else:
            z_test_input_2d = z_test_input.reshape(1, -1)
        

        cos_sims = cosine_similarity(z_inputs_2d, z_test_input_2d).flatten()
        cos_weights = softmax(cos_sims)
        return np.sum([w * z for w, z in zip(cos_weights, z_diffs)], axis=0)
    else:
        return np.zeros_like(z_test_input)

def process_train_pairs(pipeline: Pipeline, train_pairs, model_type="vq"):
    z_inputs = []
    z_diffs = []
    z_size = None
    for input, output in train_pairs:
        z_input, size = grid_to_latent(pipeline, input, model_type)
        z_output, _ = grid_to_latent(pipeline, output, model_type)
        z_inputs.append(z_input)
        z_diffs.append(z_output - z_input)
        if z_size is None:
            z_size = size
    
    return z_inputs, z_diffs, z_size

def visual_analogy_single_task(pipeline: Pipeline, train_pairs, test_pair, model_type="vq", comp='average'):
    pipeline.model_eval()

    with torch.no_grad():
        test_input, test_output = test_pair
        z_inputs, z_diffs, z_size = process_train_pairs(pipeline, train_pairs, model_type)
        z_test_input, _ = grid_to_latent(pipeline, test_input, model_type)

        z_diff = extract_diff(z_diffs, z_inputs, z_test_input, comp)
        
        z_pred = z_test_input + z_diff
        z_pred_reshaped = z_pred.view(*z_size)
        
        predicted_output = latent_to_grid(pipeline, z_pred_reshaped, test_output, model_type)
    return predicted_output, test_output
