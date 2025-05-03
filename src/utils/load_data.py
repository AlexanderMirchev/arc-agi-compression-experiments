import glob
import os
import json
import re
import numpy as np

from torch.utils.data import random_split

def _get_pairs(data):
    return {
        "train": [(np.array(pair["input"]), np.array(pair["output"])) for pair in data["train"]],
        "test": [(np.array(pair["input"]), np.array(pair["output"])) for pair in data["test"]]
    }

def _get_puzzle_id(file):
    return re.search(r"([\da-f]+)\.json$", file).group(1)

def _extract_pairs_by_puzzle_id(dataset):
    pairs_by_id = {}
    for file in dataset:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
            pairs_by_id[_get_puzzle_id(file)] = _get_pairs(data)
    
    return pairs_by_id


def get_grids(filepath):
    json_files = glob.glob(os.path.join(filepath, "*.json"))

    train_dataset, val_dataset = random_split(json_files, [300, 100])

    train_data = _extract_pairs_by_puzzle_id(train_dataset)
    validation_data = _extract_pairs_by_puzzle_id(val_dataset)

    return train_data, validation_data
