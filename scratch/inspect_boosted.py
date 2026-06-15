import numpy as np
import pandas as pd
import joblib
import os
from scipy.stats import rankdata

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE3_DIR = os.path.join(BASE_DIR, "phase3")
PHASE4_DIR = os.path.join(BASE_DIR, "phase4")
PHASE4B_DIR = os.path.join(BASE_DIR, "phase4b")

test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))
df_clean = pd.read_csv(os.path.join(BASE_DIR, "dataset_cleaned.csv"))
df_clean['original_row_idx'] = df_clean.index
df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
test_orig_indices = df_shuffled.iloc[test_idx]['original_row_idx'].values
account_ids = [f"A{idx}" for idx in test_orig_indices]

# Load ML Probabilities (Phase 3)
X_final = pd.read_csv(os.path.join(BASE_DIR, "X_final.csv"))
X_test = X_final.iloc[test_idx]
model = joblib.load(os.path.join(PHASE3_DIR, "best_model.pkl"))
ml_probability = model.predict_proba(X_test)[:, 1]
ml_score = ml_probability * 100

# Load Stat Anomaly Scores (Phase 4A)
df_4a = pd.read_csv(os.path.join(PHASE4_DIR, "anomaly_scores.csv"))
stat_score = df_4a["anomaly_score"].values
target = df_4a["target"].values

# Load Behavioral Anomaly Scores (Phase 4B)
df_4b = pd.read_csv(os.path.join(PHASE4B_DIR, "behavioral_anomaly_scores.csv"))
behavior_raw_score = df_4b["anomaly_signal"].values

# Percentile scaling for LOF
percentiles = rankdata(behavior_raw_score, method="average") / len(behavior_raw_score)
behavior_score = percentiles * 100

# Merge
df_merged = pd.DataFrame({
    "account_id": account_ids,
    "target": target,
    "ml_score": ml_score,
    "stat_score": stat_score,
    "behavior_score": behavior_score
})

# Identify boosted accounts
df_merged["boosted"] = df_merged["behavior_score"] >= 99

print("Boosted accounts details:")
boosted_df = df_merged[df_merged["boosted"]]
print(boosted_df.sort_values("behavior_score", ascending=False)[["account_id", "target", "ml_score", "stat_score", "behavior_score"]])

print(f"\nNumber of boosted accounts: {len(boosted_df)}")
print(f"Number of boosted actual mules: {boosted_df['target'].sum()}")
