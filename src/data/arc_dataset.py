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
        augmenter = GridAugmenter(num_colors=self.num_classes - 1, color_aug_prob=1, mirror_aug_prob=1, rotation_aug_prob=1) # one class is for the backgrounds
        
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

class PairedGridDataset(Dataset):
    def __init__(self, grid_pairs, num_classes=10, target_size=(30, 30), augment=True, normalize_fn=None):
        self.grid_pairs = [(np.array(g1), np.array(g2)) for g1, g2 in grid_pairs]
        self.num_classes = num_classes
        self.target_height, self.target_width = target_size
        self.normalize_fn = normalize_fn

        initial_size = len(self.grid_pairs)
        if augment:
            self._apply_augmentations()

        print(f"Dataset size after augmentation: {initial_size} -> {len(self.grid_pairs)}")

    def _apply_augmentations(self):
        augmenter = GridAugmenter(num_colors=self.num_classes - 1, color_aug_count=2)
        augmented_pairs = []

        for g1, g2 in self.grid_pairs:
            aug_g1 = augmenter.augment_grid(g1)
            aug_g2 = augmenter.augment_grid(g2)

            # Match each augmented g1 with each augmented g2
            for a1 in aug_g1:
                for a2 in aug_g2:
                    augmented_pairs.append((a1, a2))

        self.grid_pairs = augmented_pairs

    def __len__(self):
        return len(self.grid_pairs)

    def __getitem__(self, index):
        grid1, grid2 = self.grid_pairs[index]

        if self.normalize_fn:
            grid1 = self.normalize_fn(grid1, (self.target_height, self.target_width))
            grid2 = self.normalize_fn(grid2, (self.target_height, self.target_width))

        g1_tensor = torch.tensor(grid1, dtype=torch.long)
        g2_tensor = torch.tensor(grid2, dtype=torch.long)

        g1_tensor = torch.clamp(g1_tensor, 0, self.num_classes - 1)
        g2_tensor = torch.clamp(g2_tensor, 0, self.num_classes - 1)

        g1_one_hot = F.one_hot(g1_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()
        g2_one_hot = F.one_hot(g2_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()

        return g1_one_hot, g2_one_hot
