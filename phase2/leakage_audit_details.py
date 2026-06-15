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
y = df["F3924"]

features = ["F3912", "F2507", "F2506", "F2409", "F2408", "F515", "F518"]

scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
with open(os.path.join(scratch_dir, "leakage_audit_details.txt"), "w") as f:
    for col in features:
        f.write(f"=== FEATURE: {col} ===\n")
        missing_pct = df[col].isnull().mean() * 100
        f.write(f"Missing %: {missing_pct:.2f}%\n")
        f.write(f"Unique count: {df[col].nunique(dropna=False)}\n")
        if df[col].dtype != object:
            f.write(f"Correlation: {df[col].corr(y):.6f}\n")
        
        # Crosstab
        ct = pd.crosstab(df[col].fillna("Missing"), y)
        f.write("Crosstab:\n")
        f.write(ct.to_string() + "\n")
        
        # Row-normalized crosstab
        ct_norm = pd.crosstab(df[col].fillna("Missing"), y, normalize="index") * 100
        f.write("Row-normalized Crosstab (%):\n")
        f.write(ct_norm.to_string() + "\n")
        f.write("-" * 50 + "\n\n")
