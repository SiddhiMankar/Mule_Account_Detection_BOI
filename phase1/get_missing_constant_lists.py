import os
import sys
import pandas as pd
import numpy as np

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
from config.paths import DATA_PHASE1

csv_path = os.path.join(DATA_PHASE1, "dataset.csv")
df = pd.read_csv(csv_path)

missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100

remove_cols = missing_percent[missing_percent > 40].index.tolist()
investigate_cols = missing_percent[(missing_percent >= 10) & (missing_percent <= 40)].index.tolist()

constant_non_nan_cols = []
for col in df.columns:
    if col != "F3924" and col != "Unnamed: 0":
        if df[col].nunique(dropna=True) <= 1:
            constant_non_nan_cols.append(col)

scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
with open(os.path.join(scratch_dir, "report_lists.txt"), "w") as f:
    f.write(f"--- CANDIDATE FOR REMOVAL (>40% missing): {len(remove_cols)} columns ---\n")
    f.write(", ".join(remove_cols) + "\n\n")
    f.write(f"--- INVESTIGATE (10%-40% missing): {len(investigate_cols)} columns ---\n")
    f.write(", ".join(investigate_cols) + "\n\n")
    f.write(f"--- CONSTANT FEATURES (excluding NaNs): {len(constant_non_nan_cols)} columns ---\n")
    f.write(", ".join(constant_non_nan_cols) + "\n\n")

print("Lists dumped to report_lists.txt")
