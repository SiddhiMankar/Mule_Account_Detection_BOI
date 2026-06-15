import numpy as np
import pandas as pd
import joblib
import os
from scipy.stats import rankdata

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE3_DIR = os.path.join(BASE_DIR, "phase3")
PHASE4_DIR = os.path.join(BASE_DIR, "phase4")
PHASE4B_DIR = os.path.join(BASE_DIR, "phase4b")

# Load indices
test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))

# 1. Load ML Probabilities (Phase 3)
X_final = pd.read_csv(os.path.join(BASE_DIR, "X_final.csv"))
X_test = X_final.iloc[test_idx]
model = joblib.load(os.path.join(PHASE3_DIR, "best_model.pkl"))
ml_probability = model.predict_proba(X_test)[:, 1]
ml_score = ml_probability * 100

# 2. Load Stat Anomaly Scores (Phase 4A)
df_4a = pd.read_csv(os.path.join(PHASE4_DIR, "anomaly_scores.csv"))
stat_score = df_4a["anomaly_score"].values
target = df_4a["target"].values

# 3. Load Behavioral Anomaly Scores (Phase 4B)
df_4b = pd.read_csv(os.path.join(PHASE4B_DIR, "behavioral_anomaly_scores.csv"))
behavior_raw_score = df_4b["anomaly_signal"].values

# Percentile scaling for LOF
percentiles = rankdata(behavior_raw_score, method="average") / len(behavior_raw_score)
behavior_score = percentiles * 100

# 4. Recreate Shuffled Dataset for Account ID mapping
df_clean = pd.read_csv(os.path.join(BASE_DIR, "dataset_cleaned.csv"))
df_clean['original_row_idx'] = df_clean.index
df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
test_orig_indices = df_shuffled.iloc[test_idx]['original_row_idx'].values
account_ids = [f"A{idx}" for idx in test_orig_indices]

# Ensure lengths match
assert len(account_ids) == len(target) == len(ml_score) == len(stat_score) == len(behavior_score)

# 5. Risk Fusion Formula
risk_score = 0.70 * ml_score + 0.10 * stat_score + 0.20 * behavior_score

# Boost extreme anomalies (behavior_score >= 99)
boost_mask = behavior_score >= 99
risk_score[boost_mask] += 10
risk_score = np.minimum(risk_score, 100.0)

# Round risk scores to integers
risk_score_rounded = np.round(risk_score).astype(int)

# Create DataFrame
df_merged = pd.DataFrame({
    "account_id": account_ids,
    "target": target,
    "ml_probability": ml_probability,
    "ml_score": ml_score,
    "stat_score": stat_score,
    "behavior_score": behavior_score,
    "risk_score": risk_score_rounded
})

# 6. Assign Risk Bands and Actions
def assign_band(score):
    if score <= 30:
        return "Normal"
    elif score <= 60:
        return "Monitor"
    elif score <= 80:
        return "High Risk"
    else:
        return "Critical"

def get_action(band):
    actions = {
        "Normal": "No action required",
        "Monitor": "Enhanced monitoring",
        "High Risk": "Manual investigation",
        "Critical": "Immediate review"
    }
    return actions[band]

df_merged["risk_band"] = df_merged["risk_score"].apply(assign_band)
df_merged["recommended_action"] = df_merged["risk_band"].apply(get_action)

# 7. Evaluate Risk Engine
critical_df = df_merged[df_merged["risk_band"] == "Critical"]
critical_precision = critical_df["target"].mean() if len(critical_df) > 0 else 0.0

total_mules = df_merged["target"].sum()
captured = df_merged[df_merged["risk_band"].isin(["High Risk", "Critical"])]["target"].sum()
capture_rate = captured / total_mules if total_mules > 0 else 0.0

n_alerts = len(df_merged[df_merged["risk_band"] != "Normal"])

print("--- Risk Engine Evaluation Results ---")
print(f"Total test set accounts: {len(df_merged)}")
print(f"Total actual mules: {total_mules}")
print(f"Critical accounts count: {len(critical_df)}")
print(f"Critical Alert Precision: {critical_precision*100:.2f}%")
print(f"High Risk + Critical captured mules: {captured}")
print(f"Fraud Capture Rate: {capture_rate*100:.2f}%")
print(f"Alert Volume (Non-Normal): {n_alerts} ({n_alerts/len(df_merged)*100:.2f}% of test set)")

print("\nRisk Band Counts:")
print(df_merged["risk_band"].value_counts())

print("\nActual mules by risk band:")
print(df_merged.groupby("risk_band")["target"].sum())

print("\nSample of Critical accounts:")
print(df_merged[df_merged["risk_band"] == "Critical"].head(10)[["account_id", "target", "risk_score", "ml_score", "stat_score", "behavior_score"]])
