import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from data.augmentations import GridAugmenter

class GridDataset(Dataset):
    def __init__(self, grids, num_classes=10, target_size=(30, 30), augment=True, normalize_fn=None):
        self.grids = grids
        self.num_classes = num_classes
        self.target_height, self.target_width = target_size
        self.normalize_fn = normalize_fn

        initial_size = len(self.grids)
        self.grids = [np.array(arr) for arr in self.grids]
        if augment:
            self._apply_augmentations()

        print(f"Dataset size after augmentation: {initial_size} -> {len(self.grids)}")

    def _apply_augmentations(self):
        """Apply augmentations to all grids in the dataset"""
        augmenter = GridAugmenter(num_colors=self.num_classes - 1, color_aug_count=2) # one class is for the backgrounds
        
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
        if self.normalize_fn:
            grid = self.normalize_fn(grid, (30, 30))
        grid_tensor = torch.tensor(grid, dtype=torch.long)
        grid_tensor = torch.clamp(grid_tensor, 0, self.num_classes - 1)

        # masking
        # h, w = grid_tensor.shape
        # padded_grid = torch.full(
        #     (self.target_height, self.target_width), 
        #     fill_value=0,  # Padding with class index 10
        #     dtype=torch.long
        # )
        # padded_grid[:h, :w] = grid_tensor
        # mask = torch.zeros((self.target_height, self.target_width), dtype=torch.float32)
        # mask[:h, :w] = 1.0

        # One-hot encode and permute to (C, H, W)

        one_hot = F.one_hot(grid_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()
        return one_hot
