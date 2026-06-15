import pandas as pd
import os

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE4B_DIR = os.path.join(BASE_DIR, "phase4b")

scores_df = pd.read_csv(os.path.join(PHASE4B_DIR, "behavioral_anomaly_scores.csv"))
signals = scores_df["anomaly_signal"]

print("Anomaly signal statistics:")
print(f"Count: {len(signals)}")
print(f"Min: {signals.min():.6f}")
print(f"Max: {signals.max():.6f}")
print(f"Mean: {signals.mean():.6f}")
print(f"Median: {signals.median():.6f}")

print("\nTop 5 highest signals:")
print(signals.sort_values(ascending=False).head(5).tolist())
