import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Bootstrap project root
def bootstrap_root():
    path = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(path, "phase1")):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    if path not in sys.path:
        sys.path.insert(0, path)
    return path

PROJECT_ROOT = bootstrap_root()
from config.paths import DATA_PHASE2, PHASE3_DIR

y = pd.read_csv(os.path.join(DATA_PHASE2, "y_final.csv")).squeeze()

indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    stratify=y,
    random_state=42
)

os.makedirs(PHASE3_DIR, exist_ok=True)

np.save(os.path.join(PHASE3_DIR, "train_indices.npy"), train_idx)
np.save(os.path.join(PHASE3_DIR, "test_indices.npy"), test_idx)

print("Saved train_indices.npy shape:", train_idx.shape)
print("Saved test_indices.npy shape:", test_idx.shape)
