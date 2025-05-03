import numpy as np
import random

def augment_grid_pairs(
        grid_pairs, 
        target_count,      
        color_aug_prob=1.0, 
        rotation_aug_prob=0.6, 
        mirror_aug_prob=0.5,
        noise_aug_prob=1.0,
        num_colors=10,
): 
    if(len(grid_pairs) >= target_count):
        return grid_pairs
    
    expanded = grid_pairs.copy()
    
    while len(expanded) < target_count:
        pair = random.choice(grid_pairs)
        augmented = _augment_pair(pair, color_aug_prob=color_aug_prob,
                                rotation_aug_prob=rotation_aug_prob,
                                mirror_aug_prob=mirror_aug_prob,
                                noise_aug_prob=noise_aug_prob,
                                num_colors=num_colors)
        expanded.extend(augmented)
    
    return expanded

def _augment_pair(
        pair, 
        color_aug_prob, 
        rotation_aug_prob, 
        mirror_aug_prob,
        noise_aug_prob,
        num_colors=10,
):
    in_grid, out_grid = pair
    aug_pairs = [(in_grid.copy(), out_grid.copy())]

    if random.random() < color_aug_prob:
        aug_pairs.append(_apply_color_augmentation(in_grid, out_grid, num_colors))

    if random.random() < rotation_aug_prob:
        rotated = _apply_rotation_augmentation(in_grid, out_grid)
        aug_pairs.append(rotated)

    if random.random() < mirror_aug_prob:
        mirrored = _apply_mirror_augmentation(in_grid, out_grid)
        aug_pairs.append(mirrored)

    if random.random() < noise_aug_prob:
        noisy = _apply_noise_augmentation_pair(in_grid, out_grid, num_colors=num_colors)
        aug_pairs.append(noisy)

    return aug_pairs

def _apply_color_augmentation(input_grid, output_grid, num_colors=10):
    all_colors = np.unique(np.concatenate((input_grid.flatten(), output_grid.flatten())))
    if len(all_colors) <= 1:
        return input_grid.copy(), output_grid.copy()

    new_colors = np.random.permutation(num_colors)[:len(all_colors)]
    color_map = {old: new for old, new in zip(all_colors, new_colors)}

    new_input = np.zeros_like(input_grid)
    new_output = np.zeros_like(output_grid)

    for old_color, new_color in color_map.items():
        new_input[input_grid == old_color] = new_color
        new_output[output_grid == old_color] = new_color

    return new_input, new_output

def _apply_rotation_augmentation(input_grid, output_grid):
    k = random.choice([1, 2, 3])
    return np.rot90(input_grid, k=k), np.rot90(output_grid, k=k)

def _apply_mirror_augmentation(input_grid, output_grid):
    return np.fliplr(input_grid).copy(), np.fliplr(output_grid).copy()

def _apply_noise_augmentation_pair(input_grid, output_grid, noise_prob=0.05, num_colors=10):
    noisy_input = input_grid.copy()
    input_mask = np.random.rand(*input_grid.shape) < noise_prob
    input_random_values = np.random.randint(0, num_colors, size=input_grid.shape)
    noisy_input[input_mask] = input_random_values[input_mask]

    noisy_output = output_grid.copy()
    output_mask = np.random.rand(*output_grid.shape) < noise_prob
    output_random_values = np.random.randint(0, num_colors, size=output_grid.shape)
    noisy_output[output_mask] = output_random_values[output_mask]

    return noisy_input, noisy_output
