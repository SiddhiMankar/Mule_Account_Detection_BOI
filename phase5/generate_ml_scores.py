"""
generate_ml_scores.py
---------------------
Phase 5: Step 5.2 — Generate ML Risk Scores
Bank of India -- Mule Account Detection

Loads the final LightGBM model and predicts probabilities for the holdout test set.
Saves the results to phase5/ml_scores.csv.
"""

import os
import sys

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
from config.paths import DATA_PHASE2, PHASE3_DIR, PHASE5_DIR
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = str(PROJECT_ROOT)
# PHASE3_DIR and PHASE5_DIR are imported from config.paths
os.makedirs(PHASE5_DIR, exist_ok=True)

print("=" * 60)
print("Step 5.2: Generating ML Risk Scores...")
print("=" * 60)

# 1. Load test set indices
test_idx_path = os.path.join(PHASE3_DIR, "test_indices.npy")
if os.path.exists(test_idx_path):
    print("Loading test indices from phase3/test_indices.npy...")
    test_idx = np.load(test_idx_path)
else:
    # Safe fallback: recreate exactly using the random state 42 stratified split
    print("[WARN] test_indices.npy not found in phase3! Recreating splits...")
    y_final = pd.read_csv(os.path.join(DATA_PHASE2, "y_final.csv"))
    # Recreate the exact split from Phase 3
    _, test_idx = train_test_split(
        np.arange(len(y_final)),
        test_size=0.20,
        random_state=42,
        stratify=y_final
    )
    np.save(test_idx_path, test_idx)
    print(f"Saved test indices to {test_idx_path}")

# 2. Load preprocessed features
X_final_path = os.path.join(DATA_PHASE2, "X_final.csv")
print(f"Loading features from {X_final_path}...")
X_final = pd.read_csv(X_final_path)
X_test = X_final.iloc[test_idx]
print(f"Test features shape: {X_test.shape}")

# 3. Load final LightGBM model
model_path = os.path.join(PHASE3_DIR, "best_model.pkl")
if not os.path.exists(model_path):
    # Fallback to tuned_lightgbm.pkl
    model_path = os.path.join(PHASE3_DIR, "tuned_lightgbm.pkl")
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)

# 4. Generate predictions
print("Predicting probabilities on test set...")
ml_probability = model.predict_proba(X_test)[:, 1]
ml_score = ml_probability * 100

# Save to CSV
ml_scores_df = pd.DataFrame({
    "ml_probability": ml_probability,
    "ml_score": ml_score
})

output_path = os.path.join(PHASE5_DIR, "ml_scores.csv")
ml_scores_df.to_csv(output_path, index=False)
print(f"ML risk scores saved to: {output_path}")

# Display stats
print(f"ML scores stats:")
print(f"  Count : {len(ml_scores_df)}")
print(f"  Min   : {ml_scores_df['ml_score'].min():.6f}")
print(f"  Max   : {ml_scores_df['ml_score'].max():.6f}")
print(f"  Mean  : {ml_scores_df['ml_score'].mean():.6f}")
print("=" * 60)
