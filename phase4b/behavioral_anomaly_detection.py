"""
behavioral_anomaly_detection.py
--------------------------------
Phase 4B: Unsupervised Anomaly Detection on Behavioral Features
Bank of India -- Mule Account Detection

Covers Steps 4B.8 through 4B.14:
  - Load behavioral features & train/test splits.
  - Run 5-fold Stratified CV on train to tune Isolation Forest (Options A & B) and LOF.
  - Select best model by validation PR-AUC.
  - Train final model, save as behavioral_isolation_forest.pkl (or LOF equivalent).
  - MinMaxScaler to rescaled 0-100 risk score (fitting on train, transforming test).
  - Evaluate Alert Budgets (Top 0.5%, Top 1%, Top 100) and compute captured mules, rate, and lift.
  - Profile LightGBM False Negatives recovery on test set.
  - Save results and generate visualizations.
"""

import os
import sys
import json

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
from config.paths import PHASE3_DIR, PHASE4_DIR, PHASE4B_DIR
import warnings
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# Setup paths
BASE_DIR   = str(PROJECT_ROOT)
# PHASE3_DIR, PHASE4_DIR, and PHASE4B_DIR are imported from config.paths
os.makedirs(PHASE4B_DIR, exist_ok=True)

print("=" * 60)
print("Phase 4B Behavioral Anomaly Detection Pipeline Starting...")
print("=" * 60)

# 1. Load Data
print("\nLoading behavioral feature dataset...")
features_df = pd.read_csv(os.path.join(PHASE4B_DIR, "behavioral_features.csv"))
y = features_df["target"].copy()
X = features_df.drop(columns=["target"]).copy()

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")

train_idx = np.load(os.path.join(PHASE3_DIR, "train_indices.npy"))
test_idx  = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"  Train set: {X_train.shape[0]} rows (mules: {y_train.sum()})")
print(f"  Test set : {X_test.shape[0]} rows (mules: {y_test.sum()})")

# 2. Cross-Validation and Model Tuning
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

print("\nRunning 5-Fold Stratified CV on Training Set...")
print("-" * 60)

# Model 1: Isolation Forest Grid Search (Options A & B)
contaminations = [0.005, 0.01, 0.02]
if_options = ["A", "B"]

for opt in if_options:
    for cont in contaminations:
        fold_pr_aucs = []
        fold_roc_aucs = []
        fold_recalls_99 = []
        fold_recalls_top100 = []
        
        for fold_idx, (train_cv_idx, val_cv_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_cv_idx], X_train.iloc[val_cv_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]
            
            # Fit Option
            if opt == "B":
                X_fit = X_fold_train[y_fold_train == 0]
            else:
                X_fit = X_fold_train
                
            clf = IsolationForest(
                n_estimators=500,
                contamination=cont,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_fit)
            
            # Anomaly signal is inverted decision function
            scores_val = clf.decision_function(X_fold_val)
            signal_val = -scores_val
            
            # Metrics
            pr_auc  = average_precision_score(y_fold_val, signal_val)
            roc_auc = roc_auc_score(y_fold_val, signal_val)
            
            # Recall @ Top 1% (99th percentile threshold of validation fold)
            pct_99   = np.percentile(signal_val, 99)
            flag_val = (signal_val >= pct_99).astype(int)
            rec_99   = recall_score(y_fold_val, flag_val, zero_division=0)
            
            # Recall @ Top 100 accounts in validation fold
            top100_local_idx = np.argsort(signal_val)[::-1][:100]
            captured_mules = y_fold_val.iloc[top100_local_idx].sum()
            total_mules = y_fold_val.sum()
            rec_top100 = captured_mules / total_mules if total_mules > 0 else 0.0
            
            fold_pr_aucs.append(pr_auc)
            fold_roc_aucs.append(roc_auc)
            fold_recalls_99.append(rec_99)
            fold_recalls_top100.append(rec_top100)
            
        mean_pr      = np.mean(fold_pr_aucs)
        mean_roc     = np.mean(fold_roc_aucs)
        mean_rec_99  = np.mean(fold_recalls_99)
        mean_rec_100 = np.mean(fold_recalls_top100)
        
        model_name = f"Isolation Forest (Opt {opt}, Cont={cont:.3f})"
        print(f"  {model_name:38s} | PR-AUC: {mean_pr:.4f} | ROC-AUC: {mean_roc:.4f} | Rec@Top1%: {mean_rec_99:.4f} | Rec@Top100: {mean_rec_100:.4f}")
        
        cv_results.append({
            "model_type": "Isolation Forest",
            "option": opt,
            "param": cont,
            "name": model_name,
            "mean_pr_auc": mean_pr,
            "mean_roc_auc": mean_roc,
            "mean_rec_99": mean_rec_99,
            "mean_rec_100": mean_rec_100
        })

# Model 2: Local Outlier Factor Grid Search (Novelty=True)
n_neighbors_list = [10, 20, 50]

for nn in n_neighbors_list:
    fold_pr_aucs = []
    fold_roc_aucs = []
    fold_recalls_99 = []
    fold_recalls_top100 = []
    
    for fold_idx, (train_cv_idx, val_cv_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_cv_idx], X_train.iloc[val_cv_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]
        
        # LOF is fitted on all training accounts in the fold (LOF does not support option B naturally for novelty)
        clf = LocalOutlierFactor(
            n_neighbors=nn,
            novelty=True,
            n_jobs=-1
        )
        clf.fit(X_fold_train)
        
        scores_val = clf.decision_function(X_fold_val)
        signal_val = -scores_val
        
        # Metrics
        pr_auc  = average_precision_score(y_fold_val, signal_val)
        roc_auc = roc_auc_score(y_fold_val, signal_val)
        
        # Recall @ Top 1%
        pct_99   = np.percentile(signal_val, 99)
        flag_val = (signal_val >= pct_99).astype(int)
        rec_99   = recall_score(y_fold_val, flag_val, zero_division=0)
        
        # Recall @ Top 100 accounts
        top100_local_idx = np.argsort(signal_val)[::-1][:100]
        captured_mules = y_fold_val.iloc[top100_local_idx].sum()
        total_mules = y_fold_val.sum()
        rec_top100 = captured_mules / total_mules if total_mules > 0 else 0.0
        
        fold_pr_aucs.append(pr_auc)
        fold_roc_aucs.append(roc_auc)
        fold_recalls_99.append(rec_99)
        fold_recalls_top100.append(rec_top100)
        
    mean_pr      = np.mean(fold_pr_aucs)
    mean_roc     = np.mean(fold_roc_aucs)
    mean_rec_99  = np.mean(fold_recalls_99)
    mean_rec_100 = np.mean(fold_recalls_top100)
    
    model_name = f"Local Outlier Factor (NN={nn})"
    print(f"  {model_name:38s} | PR-AUC: {mean_pr:.4f} | ROC-AUC: {mean_roc:.4f} | Rec@Top1%: {mean_rec_99:.4f} | Rec@Top100: {mean_rec_100:.4f}")
    
    cv_results.append({
        "model_type": "Local Outlier Factor",
        "option": "A",
        "param": nn,
        "name": model_name,
        "mean_pr_auc": mean_pr,
        "mean_roc_auc": mean_roc,
        "mean_rec_99": mean_rec_99,
        "mean_rec_100": mean_rec_100
    })

# Select best model by highest CV PR-AUC
best_cfg = max(cv_results, key=lambda x: x["mean_pr_auc"])
print("-" * 60)
print(f"Best Configuration: {best_cfg['name']} | CV PR-AUC: {best_cfg['mean_pr_auc']:.4f}")

# 3. Train Final Model on Training Set & Save
print(f"\nTraining final model '{best_cfg['name']}' on full training set...")
if best_cfg["model_type"] == "Isolation Forest":
    if best_cfg["option"] == "B":
        X_final_fit = X_train[y_train == 0]
        print(f"  Fitting only on normal training accounts ({X_final_fit.shape[0]} rows)...")
    else:
        X_final_fit = X_train
        print(f"  Fitting on all training accounts ({X_final_fit.shape[0]} rows)...")
        
    final_model = IsolationForest(
        n_estimators=500,
        contamination=best_cfg["param"],
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_final_fit)
    model_file = "behavioral_isolation_forest.pkl"
else:
    # LOF
    X_final_fit = X_train
    print(f"  Fitting LOF on all training accounts ({X_final_fit.shape[0]} rows)...")
    final_model = LocalOutlierFactor(
        n_neighbors=best_cfg["param"],
        novelty=True,
        n_jobs=-1
    )
    final_model.fit(X_final_fit)
    model_file = "behavioral_lof.pkl"

model_path = os.path.join(PHASE4B_DIR, model_file)
joblib.dump(final_model, model_path)
print(f"  Saved final model object to: {model_path}")

# 4. Score Transformation (0-100 scale, fitted on train, transformed on test)
print("\nTransforming anomaly signals to risk scores (0-100)...")
train_raw_signals = -final_model.decision_function(X_train)
test_raw_signals  = -final_model.decision_function(X_test)

# Fit MinMaxScaler on training signals
scaler_0_100 = MinMaxScaler(feature_range=(0, 100))
scaler_0_100.fit(train_raw_signals.reshape(-1, 1))

# Transform train/test
train_risk_scores = scaler_0_100.transform(train_raw_signals.reshape(-1, 1)).flatten()
test_risk_scores  = scaler_0_100.transform(test_raw_signals.reshape(-1, 1)).flatten()

# Clip to exactly [0, 100] to handle out-of-bounds anomalies safely
train_risk_scores = np.clip(train_risk_scores, 0.0, 100.0)
test_risk_scores  = np.clip(test_risk_scores, 0.0, 100.0)

# Evaluate class separation on test set
normal_risk_mean = test_risk_scores[y_test == 0].mean()
mule_risk_mean   = test_risk_scores[y_test == 1].mean()

print(f"  Average anomaly risk score (Normal accounts): {normal_risk_mean:.2f}")
print(f"  Average anomaly risk score (Mule accounts)  : {mule_risk_mean:.2f}")
if mule_risk_mean > normal_risk_mean:
    print("  ✅ Separation Success: Mule risk scores > Normal risk scores on holdout test set.")
else:
    print("  ❌ Separation Failure: Mule risk scores <= Normal risk scores on holdout test set.")

# Save raw signals, transformed risk scores, ranks, and 99th percentile flags for test set
test_ranks = pd.Series(test_raw_signals).rank(ascending=False, method="min").astype(int)

# 99th percentile threshold fitted on test risk scores
test_threshold_99 = np.percentile(test_risk_scores, 99)
test_flags = (test_risk_scores >= test_threshold_99).astype(int)

results_df = pd.DataFrame({
    "target": y_test.values,
    "anomaly_signal": test_raw_signals,
    "risk_score": test_risk_scores,
    "rank": test_ranks.values,
    "predicted_flag": test_flags
})
scores_path = os.path.join(PHASE4B_DIR, "behavioral_anomaly_scores.csv")
results_df.to_csv(scores_path, index=False)
print(f"  Saved behavioral anomaly scores to: {scores_path}")

# 5. Alert Budget Capacity Analysis (Step 4B.11)
print("\nRunning alert budget (Top-K) capacity analysis on test set...")
total_test_accounts = len(y_test)
total_test_mules = y_test.sum()

alert_budgets = {
    "Top 0.5%": int(np.ceil(total_test_accounts * 0.005)),
    "Top 1.0%": int(np.ceil(total_test_accounts * 0.01)),
    "Top 100 Accounts": 100
}

alert_records = []
overall_mule_rate = total_test_mules / total_test_accounts

for label, k in alert_budgets.items():
    top_k_df = results_df.sort_values(by="risk_score", ascending=False).head(k)
    captured_mules = top_k_df["target"].sum()
    capture_rate = captured_mules / total_test_mules
    
    # Lift = (Mule rate in alerts) / (Overall mule rate)
    alert_mule_rate = captured_mules / k
    lift = alert_mule_rate / overall_mule_rate
    
    print(f"  {label:16s} (k={k:3d} accounts) | Captured Mules: {captured_mules:2d}/{total_test_mules:2d} | "
          f"Rate: {capture_rate*100:.2f}% | Lift: {lift:.2f}x")
          
    alert_records.append({
        "alert_budget": label,
        "k_accounts": k,
        "captured_mules": int(captured_mules),
        "capture_rate": float(capture_rate),
        "lift": float(lift)
    })

# 6. LightGBM False Negative Recovery Analysis (Step 4B.12)
print("\nLoading LightGBM predictions to assess False Negative recovery...")
lgb_preds_path = os.path.join(PHASE4_DIR, "phase3_test_predictions.csv")

if not os.path.exists(lgb_preds_path):
    print("  [WARN] Phase 3 test predictions file not found in phase4/phase3_test_predictions.csv. Skipping recovery profiling.")
    fn_recovery_records = []
    num_fn = 0
else:
    lgb_df = pd.read_csv(lgb_preds_path)
    # Align rows
    eval_df = pd.DataFrame({
        "target": y_test.values,
        "lgb_prediction": lgb_df["lgb_prediction"].values,
        "lgb_probability": lgb_df["lgb_probability"].values,
        "risk_score": test_risk_scores,
        "rank": test_ranks.values
    })
    
    # Identify False Negatives
    lgb_fn_mask = (eval_df["target"] == 1) & (eval_df["lgb_prediction"] == 0)
    lgb_fn_df = eval_df[lgb_fn_mask]
    num_fn = len(lgb_fn_df)
    print(f"  LightGBM missed {num_fn} mule accounts (False Negatives) on test set.")
    
    fn_recovery_records = []
    if num_fn > 0:
        print("  Profiling missed mules and Anomaly Risk Scores:")
        print(lgb_fn_df[["target", "lgb_probability", "risk_score", "rank"]].to_string(index=True))
        
        percentile_thresholds = [95, 97.5, 99]
        for p in percentile_thresholds:
            pct_val = np.percentile(test_risk_scores, p)
            recovered_df = lgb_fn_df[lgb_fn_df["risk_score"] >= pct_val]
            num_recovered = len(recovered_df)
            rec_rate = num_recovered / num_fn
            print(f"    Anomaly Model threshold {p}th percentile ({pct_val:.2f} risk score) catches {num_recovered} out of {num_fn} missed mules ({rec_rate*100:.2f}%).")
            
            fn_recovery_records.append({
                "percentile": p,
                "score_threshold": pct_val,
                "recovered_count": int(num_recovered),
                "recovery_rate": float(rec_rate)
            })

# 7. Generate Visualizations (Step 4B.13)
print("\nGenerating visualizations...")
# Distribution of Rescaled Anomaly Scores
plt.figure(figsize=(8, 5))
sns.histplot(results_df["risk_score"], bins=50, kde=True, color="teal")
plt.axvline(test_threshold_99, color="red", linestyle="--", label="99th Percentile Threshold")
plt.title("Distribution of Rescaled Behavioral Anomaly Scores (Test Set)", fontsize=13)
plt.xlabel("Behavioral Anomaly Risk Score (0-100)", fontsize=11)
plt.ylabel("Count of Accounts", fontsize=11)
plt.legend()
plt.tight_layout()
dist_img_path = os.path.join(PHASE4B_DIR, "behavioral_distribution.png")
plt.savefig(dist_img_path, dpi=150)
plt.close()
print(f"  Saved: {dist_img_path}")

# Box plot of anomaly scores by target class
plt.figure(figsize=(6, 5))
sns.boxplot(x="target", y="risk_score", data=results_df, hue="target", palette={0: "#4C72B0", 1: "#C44E52"}, legend=False)
plt.xticks([0, 1], ["Normal (0)", "Mule (1)"])
plt.title("Behavioral Anomaly Score Comparison by Class", fontsize=13)
plt.xlabel("Actual Class", fontsize=11)
plt.ylabel("Behavioral Anomaly Risk Score (0-100)", fontsize=11)
plt.tight_layout()
vs_img_path = os.path.join(PHASE4B_DIR, "behavioral_vs_target.png")
plt.savefig(vs_img_path, dpi=150)
plt.close()
print(f"  Saved: {vs_img_path}")

# 8. Save Summary Metrics (Step 4B.14)
print("\nSaving analysis table...")
final_test_roc_auc = roc_auc_score(y_test, test_raw_signals)
final_test_pr_auc  = average_precision_score(y_test, test_raw_signals)

analysis_metrics = [
    {"Metric": "Best CV Configuration", "Value": best_cfg["name"]},
    {"Metric": "CV PR-AUC", "Value": f"{best_cfg['mean_pr_auc']:.4f}"},
    {"Metric": "CV ROC-AUC", "Value": f"{best_cfg['mean_roc_auc']:.4f}"},
    {"Metric": "Final Test ROC-AUC", "Value": f"{final_test_roc_auc:.4f}"},
    {"Metric": "Final Test PR-AUC", "Value": f"{final_test_pr_auc:.4f}"},
    {"Metric": "Avg anomaly risk score (Normal)", "Value": f"{normal_risk_mean:.2f}"},
    {"Metric": "Avg anomaly risk score (Mule)", "Value": f"{mule_risk_mean:.2f}"},
    {"Metric": "Mules captured in Top 0.5% alerts", "Value": f"{alert_records[0]['captured_mules']}/{total_test_mules} (Rate: {alert_records[0]['capture_rate']*100:.2f}%, Lift: {alert_records[0]['lift']:.2f}x)"},
    {"Metric": "Mules captured in Top 1% alerts", "Value": f"{alert_records[1]['captured_mules']}/{total_test_mules} (Rate: {alert_records[1]['capture_rate']*100:.2f}%, Lift: {alert_records[1]['lift']:.2f}x)"},
    {"Metric": "Mules captured in Top 100 alerts", "Value": f"{alert_records[2]['captured_mules']}/{total_test_mules} (Rate: {alert_records[2]['capture_rate']*100:.2f}%, Lift: {alert_records[2]['lift']:.2f}x)"},
    {"Metric": "LightGBM False Negatives Missed", "Value": str(num_fn)}
]

if num_fn > 0:
    for item in fn_recovery_records:
        analysis_metrics.append({
            "Metric": f"LGBM False Negatives Flagged by Anomaly Model (>={item['percentile']}% pct)",
            "Value": f"{item['recovered_count']} out of {num_fn} (Rate: {item['recovery_rate']*100:.2f}%)"
        })
else:
    analysis_metrics.append({
        "Metric": "LGBM False Negatives Flagged by Anomaly Model",
        "Value": "N/A (0 missed)"
    })

analysis_df = pd.DataFrame(analysis_metrics)
analysis_csv_path = os.path.join(PHASE4B_DIR, "behavioral_analysis.csv")
analysis_df.to_csv(analysis_csv_path, index=False)
print(f"  Saved behavioral analysis summary to: {analysis_csv_path}")

print("\n" + "=" * 60)
print("Phase 4B Pipeline completed successfully.")
print("=" * 60)
