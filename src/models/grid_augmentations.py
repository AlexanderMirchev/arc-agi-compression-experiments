import numpy as np
import random

class GridAugmenter:
    def __init__(self, num_colors, color_aug_prob=1, rotation_aug_prob=0.6, mirror_aug_prob=0.5):
        self.num_colors = num_colors
        self.color_aug_prob = color_aug_prob
        self.rotation_aug_prob = rotation_aug_prob
        self.mirror_aug_prob = mirror_aug_prob
        
    def augment_grid(self, grid):
        augmented_grids = [np.array(grid)] 
        color_augs = []
    
        # Color augmentations
        if random.random() < self.color_aug_prob:
            color_augs = self._color_augmentation(grid)
            augmented_grids.extend(color_augs)
        
        # Rotation augmentations
        if random.random() < self.rotation_aug_prob:
            rotated_grids = self._rotation_augmentation(grid)
            augmented_grids.extend(rotated_grids)
            
            # Also apply rotations to color augmented grids
            for color_grid in color_augs:
                rotated_color_grids = self._rotation_augmentation(color_grid)
                augmented_grids.extend(rotated_color_grids)
        
        # Mirror augmentations
        # if random.random() < self.mirror_aug_prob:
        #     mirror_grid = self._mirror_augmentation(grid)
        #     if mirror_grid is not None:  # Only add if mirroring was successful
        #         augmented_grids.append(mirror_grid)
                
        #         # Also mirror rotated and color augmented grids
        #         for color_grid in color_augs:
        #             mirror_color = self._mirror_augmentation(color_grid)
        #             if mirror_color is not None:
        #                 augmented_grids.append(mirror_color)
                            
        return augmented_grids
    
    def _color_augmentation(self, grid):
        result = []
        original_colors = np.unique(grid)
        
        # Skip if there's only one color (nothing to remap)
        if len(original_colors) <= 1:
            return result
            
        # Create 2 color remappings
        for _ in range(2):
            new_colors = np.random.permutation(self.num_colors)[:len(original_colors)]
            color_map = {old: new for old, new in zip(original_colors, new_colors)}
            
            new_grid = np.zeros_like(grid)
            for old_color, new_color in color_map.items():
                new_grid[grid == old_color] = new_color
                
            result.append(new_grid)
            
        return result
    
    def _rotation_augmentation(self, grid):
        rotations = []

        rot90 = np.rot90(grid)
        rotations.append(rot90)
        
        rot180 = np.rot90(rot90)
        rotations.append(rot180)
        
        rot270 = np.rot90(rot180)
        rotations.append(rot270)
        
        return np.array([random.choice(rotations)])
    
    #def _mirror_augmentation(self, grid):
        # w = len(grid[0])
        
        # # Only mirror if we have an even number of columns
        # if w % 2 == 0:
        #     mid = w // 2
        #     mirrored = np.zeros_like(grid)
        #     mirrored[:, :mid] = grid[:, :mid]
        #     mirrored[:, mid:] = np.fliplr(grid[:, :mid])
        #     return mirrored
        #return None