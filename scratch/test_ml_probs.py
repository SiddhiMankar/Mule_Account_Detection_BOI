import numpy as np
import pandas as pd
import joblib
import os

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE3_DIR = os.path.join(BASE_DIR, "phase3")

print("Loading indices and data...")
test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))
X_final = pd.read_csv(os.path.join(BASE_DIR, "X_final.csv"))
X_test = X_final.iloc[test_idx]

print("Loading model...")
model = joblib.load(os.path.join(PHASE3_DIR, "best_model.pkl"))
print("Model loaded:", type(model))

# Generate predictions
print("Generating probabilities...")
probs = model.predict_proba(X_test)[:, 1]
print(f"Probabilities shape: {probs.shape}")
print(f"Min: {probs.min():.6f}, Max: {probs.max():.6f}, Mean: {probs.mean():.6f}")
