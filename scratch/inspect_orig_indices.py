import pandas as pd
import numpy as np
import os

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE3_DIR = os.path.join(BASE_DIR, "phase3")

test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))

df_raw = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
df_clean = pd.read_csv(os.path.join(BASE_DIR, "dataset_cleaned.csv"))

print(f"Raw shape: {df_raw.shape}, Cleaned shape: {df_clean.shape}")

# Recreate shuffled dataset with original row index
df_clean['original_row_idx'] = df_clean.index
df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
test_orig_indices = df_shuffled.iloc[test_idx]['original_row_idx'].values

print(f"Sample of test original indices: {test_orig_indices[:10]}")
print(f"Are all original indices unique? {len(np.unique(test_orig_indices)) == len(test_orig_indices)}")
