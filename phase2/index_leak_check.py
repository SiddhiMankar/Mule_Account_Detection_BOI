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

# Let's inspect the indices where F3924 is 1
mule_indices = df[df["F3924"] == 1]["Unnamed: 0"].tolist()
print(f"Total mule accounts: {len(mule_indices)}")
print("First 20 mule account indices (Unnamed: 0):")
print(mule_indices[:20])
print("Last 20 mule account indices (Unnamed: 0):")
print(mule_indices[-20:])

# Let's check descriptive statistics of Unnamed: 0 for normal vs mule
print("\nSummary statistics of Unnamed: 0 for normal accounts:")
print(df[df["F3924"] == 0]["Unnamed: 0"].describe())
print("\nSummary statistics of Unnamed: 0 for mule accounts:")
print(df[df["F3924"] == 1]["Unnamed: 0"].describe())

# Find all features with corr > 0.1 or < -0.1
numeric_df = df.select_dtypes(include=[np.number])
corrs = numeric_df.corrwith(df["F3924"]).dropna()
corrs_sorted = corrs.sort_values(ascending=False).drop(labels=["F3924"], errors='ignore')

high_pos_corr = corrs_sorted[corrs_sorted > 0.1]
high_neg_corr = corrs_sorted[corrs_sorted < -0.1]

print(f"\nNumber of features with correlation > 0.1: {len(high_pos_corr)}")
print(high_pos_corr)

print(f"\nNumber of features with correlation < -0.1: {len(high_neg_corr)}")
print(high_neg_corr)
