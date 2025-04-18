import glob
import os
import json
from torch.utils.data import random_split

def get_grids(filepath):
    json_files = glob.glob(os.path.join(filepath, "*.json"))

    train_dataset, val_dataset = random_split(json_files, [300, 100])

    training_data = []
    for file in train_dataset:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            training_data.extend([matrix for pairs in data.values() for pair in pairs if isinstance(pair, dict) for matrix in pair.values()])

    validation_data = []
    for file in val_dataset:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            validation_data.extend([matrix for pairs in data.values() for pair in pairs if isinstance(pair, dict)  for matrix in pair.values()])

    return training_data, validation_data
