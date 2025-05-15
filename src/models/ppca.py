import numpy as np
import pickle
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from data.scale_processing import scaling, reverse_scaling
from data.augmentations import augment_grid_pairs
from utils.load_data import get_grids

class PPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.fitted = False

    def fit(self, X):
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        svd = TruncatedSVD(n_components=self.n_components)
        svd.fit(X_centered)

        V = svd.components_.T
        S = svd.singular_values_ 

        W = V * (S / np.sqrt(n))[np.newaxis, :]
        self.W_ = W

        X_proj = X_centered @ W @ np.linalg.inv(W.T @ W) @ W.T
        residual = X_centered - X_proj
        sigma2 = (residual ** 2).sum() / (n * (d - self.n_components))
        self.sigma2_ = sigma2

        self.M_ = W.T @ W + sigma2 * np.eye(self.n_components)
        self.M_inv_ = np.linalg.inv(self.M_)
        self.fitted = True

    def transform(self, X):
        assert self.fitted, "Model must be fitted before calling transform"
        X_centered = X - self.mean_
        Z = X_centered @ self.W_ @ self.M_inv_
        return Z

    def inverse_transform(self, Z):
        assert self.fitted, "Model must be fitted before calling inverse_transform"
        return Z @ self.W_.T + self.mean_

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.__dict__ = pickle.load(f)

def preprocess_grid(grid):
    grid = scaling(grid, height=30, width=30, direction='norm')

    num_classes = 10
    grid_tensor = torch.tensor(grid, dtype=torch.long)
    grid_tensor = torch.clamp(grid_tensor, 0, num_classes - 1)
    one_hot = F.one_hot(grid_tensor, num_classes=10).permute(2, 0, 1).float()
    # print(one_hot.shape)

    return one_hot.reshape(-1)

def postprocess_grid(grid, grid_original):
    grid = grid.reshape(10,30,30)
    _, grid = torch.max(grid, dim=0)
    grid = grid.detach().cpu().numpy()
    return reverse_scaling(grid_original, grid)

def main():
    # torch.manual_seed(42)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_data, _ = get_grids(filepath="data/training")

    training_grid_pairs = [pair for task in training_data.values() for pairs in task.values() for pair in pairs]
    # validation_grid_pairs = [pair for task in validation_data.values() for pairs in task.values() for pair in pairs]

    training_grid_pairs = augment_grid_pairs(training_grid_pairs, target_count=5000)
    training_grids = np.array([preprocess_grid(grid) for pair in training_grid_pairs for grid in pair])

    n_components = 128
    model = PPCA(n_components=n_components)
    model.fit(training_grids)
    model.save(f"checkpoints/ppca_{n_components}.pkl")

if __name__ == "__main__":
    main()