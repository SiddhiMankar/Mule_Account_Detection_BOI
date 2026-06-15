import os
import sys
import pandas as pd
import time

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
from config.paths import DATA_PHASE2

start_time = time.time()

csv_path = os.path.join(DATA_PHASE2, "dataset_cleaned.csv")
output_path = os.path.join(DATA_PHASE2, "dataset_cleaned.csv")

print("Loading dataset_cleaned.csv...")
df = pd.read_csv(csv_path)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Calculate missing percentages
missing_pct = (df.isnull().sum() / len(df)) * 100

# Identify columns with >40% missing (excluding target)
target_col = "F3924"
cols_to_drop = missing_pct[(missing_pct > 40)].index.tolist()
cols_to_drop = [c for c in cols_to_drop if c != target_col]

print(f"\nColumns with >40% missing values: {len(cols_to_drop)}")
print(f"Sample columns to drop: {cols_to_drop[:15]}...")

# Drop the columns
df_dropped = df.drop(columns=cols_to_drop)
print(f"\nAfter dropping: {df_dropped.shape[0]} rows × {df_dropped.shape[1]} columns")
print(f"Columns removed: {len(cols_to_drop)}")

# Save the updated dataset
print(f"\nSaving to {output_path}...")
df_dropped.to_csv(output_path, index=False)
print("Saved successfully.")

# Update the dropped columns log
log_path = os.path.join(DATA_PHASE2, "dropped_columns_log.txt")
with open(log_path, "a") as f:
    f.write(f"\n--- HIGH_MISSING_40_90 ({len(cols_to_drop)} columns) ---\n")
    f.write(", ".join(sorted(cols_to_drop)) + "\n")
    f.write(f"\n=== UPDATED TOTALS ===\n")
    f.write(f"Original Columns: 3925\n")
    f.write(f"Total Columns Dropped: {3925 - df_dropped.shape[1]}\n")
    f.write(f"Final Columns: {df_dropped.shape[1]}\n")

print(f"\nUpdated dropped_columns_log.txt")

# Verify the result
remaining_missing = (df_dropped.isnull().sum() / len(df_dropped)) * 100
max_missing = remaining_missing.max()
cols_with_any_missing = (remaining_missing > 0).sum()
print(f"\n--- Verification ---")
print(f"Max missing % in remaining columns: {max_missing:.2f}%")
print(f"Columns with any missing values: {cols_with_any_missing}")
print(f"Target column present: {target_col in df_dropped.columns}")
print(f"Target distribution:\n{df_dropped[target_col].value_counts()}")

elapsed = time.time() - start_time
print(f"\nDone in {elapsed:.1f} seconds.")
