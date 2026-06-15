"""
generate_risk_scores.py
-----------------------
Phase 5: Step 5.3 to Step 5.13 — Risk Engine & Score Fusion
Bank of India -- Mule Account Detection

Executes the risk score fusion logic:
1. Loads ML risk scores, statistical anomaly scores, and behavioral anomaly raw signals.
2. Scales behavioral anomaly signals strictly on the holdout test set using percentile ranking.
3. Aligns original row indices and assigns unique account IDs.
4. Combines scores using the 70/10/20 fusion formula.
5. Applies a +10 boost for behavioral scores >= 99 (capped at 100) and rounds to 2 decimal places.
6. Maps scores to risk bands and investigator actions.
7. Saves the detailed scores table, evaluation summary, plots, and serialized engine parameters.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

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
from config.paths import DATA_PHASE2, PHASE3_DIR, PHASE4_DIR, PHASE4B_DIR, PHASE5_DIR
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = str(PROJECT_ROOT)
os.makedirs(PHASE5_DIR, exist_ok=True)

print("=" * 60)
print("Phase 5: Starting Risk Score Fusion & Engine Calibration...")
print("=" * 60)

# --- 1. Load ML Probabilities (Step 5.1 & 5.2) ---
ml_scores_path = os.path.join(PHASE5_DIR, "ml_scores.csv")
if not os.path.exists(ml_scores_path):
    print("ml_scores.csv not found! Running generate_ml_scores.py...")
    import generate_ml_scores

print(f"Loading ML risk scores from {ml_scores_path}...")
df_ml = pd.read_csv(ml_scores_path)
ml_probability = df_ml["ml_probability"].values
ml_score = df_ml["ml_score"].values

# --- 2. Load Statistical Anomaly Scores (Step 5.3) ---
stat_scores_path = os.path.join(PHASE4_DIR, "anomaly_scores.csv")
print(f"Loading statistical anomaly scores from {stat_scores_path}...")
df_stat = pd.read_csv(stat_scores_path)
stat_score = df_stat["anomaly_score"].values
target = df_stat["target"].values

# --- 3. Load Behavioral Anomaly Scores & Percentile Scale (Step 5.4) ---
behavior_scores_path = os.path.join(PHASE4B_DIR, "behavioral_anomaly_scores.csv")
print(f"Loading behavioral anomaly scores from {behavior_scores_path}...")
df_behavior = pd.read_csv(behavior_scores_path)
behavior_raw_score = df_behavior["anomaly_signal"].values

# Apply percentile scaling strictly on the holdout test set scores
print("Applying percentile ranking scaling to LOF anomaly signals on test set...")
percentiles = rankdata(behavior_raw_score, method="average") / len(behavior_raw_score)
behavior_score = percentiles * 100

# --- 4. Lineage Alignment & Account ID Generation (Step 5.5) ---
print("Recreating original index shuffle to map account IDs...")
test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))
df_clean = pd.read_csv(os.path.join(DATA_PHASE2, "dataset_cleaned.csv"))
df_clean['original_row_idx'] = df_clean.index

# Apply the same shuffle as used in train_model.py & preprocess_pipeline.py
df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
test_orig_indices = df_shuffled.iloc[test_idx]['original_row_idx'].values
account_ids = [f"A{idx}" for idx in test_orig_indices]

# Verification of shapes
print(f"Verifying input shapes:")
print(f"  ML scores length         : {len(df_ml)}")
print(f"  Stat scores length       : {len(df_stat)}")
print(f"  Behavior scores length   : {len(df_behavior)}")
print(f"  Generated Account IDs    : {len(account_ids)}")
assert len(df_ml) == len(df_stat) == len(df_behavior) == len(account_ids), "Length mismatch in score merge!"
print("  ✅ All lengths match successfully.")

# --- 5. Risk Score Fusion & Boosting (Step 5.6 & 5.7) ---
print("\nFusing scores using weights: ML (70%), Stat Anomaly (10%), Behavioral Anomaly (20%)...")
risk_score = 0.70 * ml_score + 0.10 * stat_score + 0.20 * behavior_score

# Apply behavioral boost for extreme outliers (behavior_score >= 99)
print("Applying behavioral anomaly boost (+10 score for behavior_score >= 99)...")
boost_mask = behavior_score >= 99.0
risk_score[boost_mask] += 10.0

# Cap at 100 and round to 2 decimal places
risk_score = np.minimum(risk_score, 100.0)
risk_score_rounded = np.round(risk_score, 2)

# Create final merged DataFrame
df_merged = pd.DataFrame({
    "account_id": account_ids,
    "target": target,
    "ml_probability": ml_probability,
    "ml_score": np.round(ml_score, 6),
    "stat_score": np.round(stat_score, 6),
    "behavior_score": np.round(behavior_score, 6),
    "risk_score": risk_score_rounded
})

# --- 6. Assign Risk Bands & Recommended Actions (Step 5.8 & 5.9) ---
def assign_band(score):
    if score <= 30.0:
        return "Normal"
    elif score <= 60.0:
        return "Monitor"
    elif score <= 80.0:
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

# --- 7. Save Detailed Risk Scores Table (Step 5.10) ---
scores_csv_path = os.path.join(PHASE5_DIR, "risk_scores.csv")
df_merged.to_csv(scores_csv_path, index=False)
print(f"Saved final risk scores table to: {scores_csv_path}")

# --- 8. Evaluate Risk Engine (Step 5.11) ---
critical_df = df_merged[df_merged["risk_band"] == "Critical"]
critical_precision = critical_df["target"].mean() if len(critical_df) > 0 else 0.0

total_mules = df_merged["target"].sum()
captured = df_merged[df_merged["risk_band"].isin(["High Risk", "Critical"])]["target"].sum()
capture_rate = captured / total_mules if total_mules > 0 else 0.0

n_alerts = len(df_merged[df_merged["risk_band"] != "Normal"])

print("\n" + "-" * 50)
print("Risk Engine Evaluation Summary:")
print("-" * 50)
print(f"Total Test Accounts       : {len(df_merged)}")
print(f"Total Money Mules         : {total_mules}")
print(f"Alert Volume (Non-Normal) : {n_alerts} ({n_alerts/len(df_merged)*100:.2f}%)")
print(f"Critical Accounts Volume  : {len(critical_df)} ({len(critical_df)/len(df_merged)*100:.2f}%)")
print(f"Critical Alert Precision  : {critical_precision*100:.2f}%")
print(f"Mules Captured (HR+Crit)  : {captured} out of {total_mules} ({capture_rate*100:.2f}%)")
print("-" * 50)

# Export Summary JSON
summary_stats = {
    "total_accounts": int(len(df_merged)),
    "total_mules": int(total_mules),
    "critical_count": int(len(critical_df)),
    "critical_precision": float(critical_precision),
    "captured_mules": int(captured),
    "fraud_capture_rate": float(capture_rate),
    "alert_volume": int(n_alerts),
    "alert_rate_pct": float(n_alerts / len(df_merged) * 100),
    "risk_band_counts": df_merged["risk_band"].value_counts().to_dict(),
    "mules_by_risk_band": df_merged.groupby("risk_band")["target"].sum().to_dict()
}

summary_json_path = os.path.join(PHASE5_DIR, "risk_summary.json")
with open(summary_json_path, "w") as f:
    json.dump(summary_stats, f, indent=4)
print(f"Saved risk summary JSON to: {summary_json_path}")

# --- 9. Generate Visualizations (Step 5.12) ---
print("\nGenerating visualizations...")
# Plot A: Risk score distribution histogram
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create histogram with KDE
sns.histplot(df_merged["risk_score"], bins=50, kde=True, color="#2c3e50")

# Draw thresholds
plt.axvline(30.0, color="#2ecc71", linestyle="--", linewidth=1.5, label="Normal Threshold (<= 30)")
plt.axvline(60.0, color="#f1c40f", linestyle="--", linewidth=1.5, label="Monitor Threshold (<= 60)")
plt.axvline(80.0, color="#e67e22", linestyle="--", linewidth=1.5, label="High Risk Threshold (<= 80)")

plt.title("Distribution of Fused Risk Scores (Test Set)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Fused Risk Score (0 - 100)", fontsize=12)
plt.ylabel("Number of Accounts", fontsize=12)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()

dist_plot_path = os.path.join(PHASE5_DIR, "risk_distribution.png")
plt.savefig(dist_plot_path, dpi=150)
plt.close()
print(f"  Saved risk score distribution plot to: {dist_plot_path}")

# Plot B: Bar plot of risk band counts
plt.figure(figsize=(8, 5))
band_counts = df_merged["risk_band"].value_counts().reindex(["Normal", "Monitor", "High Risk", "Critical"]).fillna(0)
colors = ["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"]

sns.barplot(x=band_counts.index, y=band_counts.values, palette=colors, hue=band_counts.index, legend=False)

# Add count labels on top of bars
for i, count in enumerate(band_counts.values):
    plt.text(i, count + max(band_counts.values)*0.01, f"{int(count)}", ha="center", va="bottom", fontweight="bold", fontsize=11)

plt.title("Account Volume by Unified Risk Band", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Risk Band", fontsize=12)
plt.ylabel("Account Count", fontsize=12)
plt.tight_layout()

counts_plot_path = os.path.join(PHASE5_DIR, "risk_band_counts.png")
plt.savefig(counts_plot_path, dpi=150)
plt.close()
print(f"  Saved risk band counts plot to: {counts_plot_path}")

# --- 10. Export Risk Engine parameters (Step 5.13) ---
engine_config = {
    "weights": {
        "ml": 0.70,
        "stat": 0.10,
        "behavior": 0.20
    },
    "boost_threshold": 99.0,
    "boost_amount": 10.0,
    "risk_bands": {
        "Normal": {"min_score": 0.0, "max_score": 30.0},
        "Monitor": {"min_score": 30.01, "max_score": 60.0},
        "High Risk": {"min_score": 60.01, "max_score": 80.0},
        "Critical": {"min_score": 80.01, "max_score": 100.0}
    },
    "recommended_actions": {
        "Normal": "No action required",
        "Monitor": "Enhanced monitoring",
        "High Risk": "Manual investigation",
        "Critical": "Immediate review"
    }
}

engine_pkl_path = os.path.join(PHASE5_DIR, "risk_engine.pkl")
joblib.dump(engine_config, engine_pkl_path)
print(f"Saved risk engine serialized config to: {engine_pkl_path}")

print("\n" + "=" * 60)
print("Phase 5 Risk Engine completed successfully.")
print("=" * 60)
