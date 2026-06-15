import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
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

X = pd.read_csv(os.path.join(DATA_PHASE2, "X_final.csv"))
y = pd.read_csv(os.path.join(DATA_PHASE2, "y_final.csv")).squeeze()

train_idx = np.load(os.path.join(PHASE3_DIR, "train_indices.npy"))
test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train Option A
iso_a = IsolationForest(n_estimators=500, contamination=0.01, random_state=42, n_jobs=-1)
iso_a.fit(X_train)
scores_a = iso_a.decision_function(X_test)

print("--- Option A (Trained on All) ---")
print("Mule scores stats:")
print(pd.Series(scores_a[y_test == 1]).describe())
print("Normal scores stats:")
print(pd.Series(scores_a[y_test == 0]).describe())

# Train Option B
iso_b = IsolationForest(n_estimators=500, contamination=0.01, random_state=42, n_jobs=-1)
iso_b.fit(X_train[y_train == 0])
scores_b = iso_b.decision_function(X_test)

print("\n--- Option B (Trained on Normal Only) ---")
print("Mule scores stats:")
print(pd.Series(scores_b[y_test == 1]).describe())
print("Normal scores stats:")
print(pd.Series(scores_b[y_test == 0]).describe())

# Check some top features
print("\nSome feature values for normal vs mule:")
mule_feats = X_train[y_train == 1].mean()
norm_feats = X_train[y_train == 0].mean()
diff = (mule_feats - norm_feats).abs().sort_values(ascending=False)
print("Top 10 features with largest mean differences between classes:")
for col in diff.head(10).index:
    print(f"Feature: {col} | Normal Mean: {norm_feats[col]:.4f} | Mule Mean: {mule_feats[col]:.4f}")
