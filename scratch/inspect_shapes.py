import numpy as np
import pandas as pd
import os

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE3_DIR = os.path.join(BASE_DIR, "phase3")
PHASE4_DIR = os.path.join(BASE_DIR, "phase4")
PHASE4B_DIR = os.path.join(BASE_DIR, "phase4b")

print("Checking files...")

# Load indices
if os.path.exists(os.path.join(PHASE3_DIR, "train_indices.npy")):
    train_idx = np.load(os.path.join(PHASE3_DIR, "train_indices.npy"))
    test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))
    print(f"Train indices shape: {train_idx.shape}, Test indices shape: {test_idx.shape}")

# Load CSVs
for name, path in [
    ("X_final.csv", os.path.join(BASE_DIR, "X_final.csv")),
    ("y_final.csv", os.path.join(BASE_DIR, "y_final.csv")),
    ("Phase 4A anomaly_scores.csv", os.path.join(PHASE4_DIR, "anomaly_scores.csv")),
    ("Phase 4B behavioral_anomaly_scores.csv", os.path.join(PHASE4B_DIR, "behavioral_anomaly_scores.csv"))
]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{name}: shape = {df.shape}, columns = {list(df.columns)}")
    else:
        print(f"{name}: NOT FOUND at {path}")
