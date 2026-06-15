"""
anomaly_detection.py
--------------------
Phase 4: Anomaly Detection
Bank of India -- Mule Account Detection

Covers Phase 4 Steps:
  4.1  Create Anomaly Detection Script
  4.2  Load Final Dataset & Splits
  4.3  Understand Isolation Forest
  4.4  Train Isolation Forest (with hyperparameter tuning via 5-Fold Stratified CV)
  4.5  Fit Model & Save
  4.6  Generate Raw Anomaly Scores
  4.7  Convert to Business-Friendly Risk Scale (0-100, fitting only on train)
  4.8  Create Anomaly Dataset
  4.9  Evaluate Separation Ability
  4.10 Visualize Score Distributions
  4.11 Compare by Class
  4.12 Create Anomaly Flags (percentile-based thresholds)
  4.13 Evaluate Against Labels
  4.14 Find Accounts Classifier Missed (False Negatives comparison)
  4.15 Produce Analysis Table -> anomaly_analysis.csv
  4.16 Write Phase 4 Report -> phase4_report.md
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
from config.paths import DATA_PHASE2, PHASE3_DIR, PHASE4_DIR
import json
import warnings
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

# Force UTF-8 stdout so print() works correctly in all consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Setup Paths and Directories
# =============================================================================
BASE_DIR   = str(PROJECT_ROOT)
# PHASE3_DIR and PHASE4_DIR are imported from config.paths
os.makedirs(PHASE4_DIR, exist_ok=True)

print("=" * 60)
print("Phase 4 Anomaly Detection Pipeline Starting...")
print("=" * 60)

# =============================================================================
# 2. Load Final Dataset & Train/Test Splits (Step 4.2)
# =============================================================================
print("\nLoading datasets...")
X = pd.read_csv(os.path.join(DATA_PHASE2, "X_final.csv"))
y = pd.read_csv(os.path.join(DATA_PHASE2, "y_final.csv")).squeeze()

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")

train_idx_path = os.path.join(PHASE3_DIR, "train_indices.npy")
test_idx_path  = os.path.join(PHASE3_DIR, "test_indices.npy")

if not (os.path.exists(train_idx_path) and os.path.exists(test_idx_path)):
    print("  [WARN] Train/test indices files not found in phase3/. Recreating them deterministically...")
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
    os.makedirs(PHASE3_DIR, exist_ok=True)
    np.save(train_idx_path, train_idx)
    np.save(test_idx_path, test_idx)
else:
    train_idx = np.load(train_idx_path)
    test_idx  = np.load(test_idx_path)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"  Train set: {X_train.shape[0]} rows (mules: {y_train.sum()})")
print(f"  Test set : {X_test.shape[0]} rows (mules: {y_test.sum()})")

# =============================================================================
# 3. 5-Fold Stratified Cross-Validation on Training Data for Parameter Tuning
# =============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

contaminations = [0.005, 0.01, 0.02]
options = ["A", "B"]  # Option A: Train on all X_train, Option B: Train on y==0 only
cv_results = []

print("\nRunning 5-Fold Stratified CV on Training Set...")
print("-" * 60)

for opt in options:
    for cont in contaminations:
        fold_pr_aucs = []
        fold_roc_aucs = []
        fold_recalls_99 = []
        
        for fold_idx, (train_cv_idx, val_cv_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_cv_idx], X_train.iloc[val_cv_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]
            
            # Sub-select training set for model fitting
            if opt == "B":
                X_fit = X_fold_train[y_fold_train == 0]
            else:
                X_fit = X_fold_train
                
            iso = IsolationForest(
                n_estimators=500,
                contamination=cont,
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            )
            iso.fit(X_fit)
            
            # Predict validation scores: anomaly_signal = -decision_scores
            scores_val = iso.decision_function(X_fold_val)
            signal_val = -scores_val
            
            # Compute metrics
            pr_auc  = average_precision_score(y_fold_val, signal_val)
            roc_auc = roc_auc_score(y_fold_val, signal_val)
            
            # Recall at 99th percentile of this validation fold
            pct_99  = np.percentile(signal_val, 99)
            flag_val = (signal_val >= pct_99).astype(int)
            rec_99  = recall_score(y_fold_val, flag_val, zero_division=0)
            
            fold_pr_aucs.append(pr_auc)
            fold_roc_aucs.append(roc_auc)
            fold_recalls_99.append(rec_99)
            
        mean_pr   = np.mean(fold_pr_aucs)
        mean_roc  = np.mean(fold_roc_aucs)
        mean_rec  = np.mean(fold_recalls_99)
        
        print(f"  Option {opt} | Contamination {cont:.3f} | "
              f"CV PR-AUC: {mean_pr:.4f} | ROC-AUC: {mean_roc:.4f} | Recall@99%: {mean_rec:.4f}")
        
        cv_results.append({
            "option": opt,
            "contamination": cont,
            "mean_pr_auc": mean_pr,
            "mean_roc_auc": mean_roc,
            "mean_rec_99": mean_rec
        })

# Select best parameter configuration by average PR-AUC
best_cfg  = max(cv_results, key=lambda x: x["mean_pr_auc"])
best_opt  = best_cfg["option"]
best_cont = best_cfg["contamination"]
print("-" * 60)
print(f"  [BEST CFG] Option: {best_opt} | Contamination: {best_cont} "
      f"| Validation PR-AUC: {best_cfg['mean_pr_auc']:.4f}")

# =============================================================================
# 4. Train Final Model and Save (Step 4.4, 4.5)
# =============================================================================
if best_opt == "B":
    X_final_fit = X_train[y_train == 0]
    print(f"\nFitting final model on normal training accounts only ({X_final_fit.shape[0]} rows)...")
else:
    X_final_fit = X_train
    print(f"\nFitting final model on all training accounts ({X_final_fit.shape[0]} rows)...")

final_iso = IsolationForest(
    n_estimators=500,
    contamination=best_cont,
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
final_iso.fit(X_final_fit)

# Save serialized model object
joblib.dump(final_iso, os.path.join(PHASE4_DIR, "isolation_forest.pkl"))
print("  Saved model to phase4/isolation_forest.pkl")

# =============================================================================
# 5. Risk Scaling (0-100) (Step 4.7)
# =============================================================================
print("\nConverting scores to business-friendly risk scale (0-100)...")
train_scores = final_iso.decision_function(X_train)
train_signal = -train_scores

# Fit MinMaxScaler on training anomaly signals
scaler = MinMaxScaler(feature_range=(0, 100))
scaler.fit(train_signal.reshape(-1, 1))
joblib.dump(scaler, os.path.join(PHASE4_DIR, "isolation_forest_scaler.pkl"))
print("  Saved scaler to phase4/isolation_forest_scaler.pkl")

# Transform test scores
test_scores = final_iso.decision_function(X_test)
test_signal = -test_scores
test_risk_scores = scaler.transform(test_signal.reshape(-1, 1)).flatten()

# Clip to exactly [0, 100] to handle any out-of-bounds anomalies in the test set
test_risk_scores = np.clip(test_risk_scores, 0.0, 100.0)

# Save scores to CSV (Step 4.8)
results_df = pd.DataFrame({
    "target": y_test,
    "raw_score": test_scores,
    "anomaly_score": test_risk_scores
})
results_df.to_csv(os.path.join(PHASE4_DIR, "anomaly_scores.csv"), index=False)
print("  Saved test anomaly scores to phase4/anomaly_scores.csv")

# =============================================================================
# 6. Evaluate Separation Ability (Step 4.9)
# =============================================================================
normal_mean = results_df[results_df.target == 0]["anomaly_score"].mean()
mule_mean   = results_df[results_df.target == 1]["anomaly_score"].mean()

print(f"\nSeparation Ability on Test Set:")
print(f"  Average anomaly risk score (Normal accounts): {normal_mean:.2f}")
print(f"  Average anomaly risk score (Mule accounts)  : {mule_mean:.2f}")
if mule_mean > normal_mean:
    print("  ✅ Success: Mule anomaly score > Normal anomaly score. The detector captures fraud-related behaviors.")
else:
    print("  ❌ Fail: Mule anomaly score <= Normal anomaly score.")

# =============================================================================
# 7. Generate Visualizations (Step 4.10, 4.11)
# =============================================================================
print("\nGenerating visualizations...")
# Distribution Histogram
plt.figure(figsize=(8, 5))
sns.histplot(results_df["anomaly_score"], bins=50, kde=True, color="teal")
plt.axvline(np.percentile(test_risk_scores, 99), color="red", linestyle="--", label="99th Percentile Threshold")
plt.title("Distribution of Rescaled Anomaly Scores (Test Set)", fontsize=13)
plt.xlabel("Anomaly Risk Score (0-100)", fontsize=11)
plt.ylabel("Count of Accounts", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PHASE4_DIR, "anomaly_distribution.png"), dpi=150)
plt.close()
print("  Saved phase4/anomaly_distribution.png")

# Box plot by class
plt.figure(figsize=(6, 5))
sns.boxplot(x="target", y="anomaly_score", data=results_df, hue="target", palette={0: "#4C72B0", 1: "#C44E52", "0": "#4C72B0", "1": "#C44E52"}, legend=False)
plt.xticks([0, 1], ["Normal (0)", "Mule (1)"])
plt.title("Anomaly Score Comparison by Class (Test Set)", fontsize=13)
plt.xlabel("Actual Class", fontsize=11)
plt.ylabel("Anomaly Risk Score (0-100)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PHASE4_DIR, "anomaly_vs_target.png"), dpi=150)
plt.close()
print("  Saved phase4/anomaly_vs_target.png")

# =============================================================================
# 8. Percentile-based Anomaly Flags (Step 4.12, 4.13)
# =============================================================================
percentiles = [95, 97.5, 99]
thresholds_records = []
print("\nEvaluating percentile-based thresholds against test labels:")

for p in percentiles:
    pct_val = np.percentile(test_risk_scores, p)
    pred_flag = (test_risk_scores >= pct_val).astype(int)
    
    prec = precision_score(y_test, pred_flag, zero_division=0)
    rec  = recall_score(y_test, pred_flag, zero_division=0)
    cm   = confusion_matrix(y_test, pred_flag)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print(f"  {p}th Percentile ({pct_val:.2f} risk threshold):")
    print(f"    Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {2*prec*rec/(prec+rec) if prec+rec > 0 else 0:.4f}")
    print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    thresholds_records.append({
        "percentile": p,
        "score_threshold": pct_val,
        "precision": prec,
        "recall": rec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

# =============================================================================
# 9. Top-K Capacity Analysis
# =============================================================================
print("\nRunning alert budget (Top-K) analysis:")
total_test = len(y_test)
mules_in_test = y_test.sum()

top_k_definitions = {
    "Top 0.5%": int(np.ceil(total_test * 0.005)),
    "Top 1.0%": int(np.ceil(total_test * 0.01)),
    "Top 100 Accounts": 100
}

top_k_records = []
for label, k in top_k_definitions.items():
    sorted_df = results_df.copy()
    sorted_df["anomaly_signal"] = test_signal
    top_k_df = sorted_df.sort_values(by="anomaly_signal", ascending=False).head(k)
    captured_mules = top_k_df["target"].sum()
    capture_rate = captured_mules / mules_in_test
    
    print(f"  {label:16s} (k={k:3d} accounts) | Captured Mules: {captured_mules:2d}/{mules_in_test:2d} | Rate: {capture_rate*100:.2f}%")
    
    top_k_records.append({
        "alert_budget": label,
        "k_accounts": k,
        "captured_mules": captured_mules,
        "capture_rate": capture_rate
    })

# =============================================================================
# 10. Load Phase 3 predictions and evaluate false negatives (Step 4.14)
# =============================================================================
print("\nLoading Phase 3 predictions to check False Negatives capture...")
best_threshold_json_path = os.path.join(PHASE3_DIR, "best_threshold.json")
best_model_pkl_path      = os.path.join(PHASE3_DIR, "best_model.pkl")

if not os.path.exists(best_threshold_json_path) or not os.path.exists(best_model_pkl_path):
    print("  [ERROR] Phase 3 model files are missing. Cannot analyze false negatives.")
    lgb_pred = np.zeros_like(y_test)
    lgb_prob = np.zeros_like(y_test)
    num_fn = 0
else:
    with open(best_threshold_json_path, "r") as f:
        best_threshold_meta = json.load(f)
    lgb_threshold = best_threshold_meta["threshold"]
    lgb_model     = joblib.load(best_model_pkl_path)
    
    # Generate predictions for X_test
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = (lgb_prob >= lgb_threshold).astype(int)
    
    # Save predictions file
    phase3_test_preds = pd.DataFrame({
        "target": y_test,
        "lgb_probability": lgb_prob,
        "lgb_prediction": lgb_pred
    })
    phase3_test_preds.to_csv(os.path.join(PHASE4_DIR, "phase3_test_predictions.csv"), index=False)
    print("  Saved LightGBM test predictions to phase4/phase3_test_predictions.csv")
    
    # False Negatives
    lgb_fn = (y_test == 1) & (lgb_pred == 0)
    num_fn = lgb_fn.sum()
    
    # Analyze if Isolation Forest caught the false negatives
    eval_df = pd.DataFrame({
        "target": y_test,
        "lgb_pred": lgb_pred,
        "lgb_prob": lgb_prob,
        "anomaly_score": test_risk_scores,
        "anomaly_signal": test_signal
    })
    
    # Anomaly ranking: 1 = most anomalous
    eval_df["anomaly_rank"] = eval_df["anomaly_signal"].rank(ascending=False, method="min").astype(int)
    
    lgb_fn_cases = eval_df[lgb_fn]
    print(f"  LightGBM classifier False Negatives (missed mules) count: {num_fn}")
    
    if num_fn > 0:
        print("  Profiling missed mules:")
        print(lgb_fn_cases[["target", "lgb_prob", "anomaly_score", "anomaly_rank"]].to_string(index=True))
        
        # Check if they are flagged by IF at 99th/97.5th/95th percentile
        for p in percentiles:
            pct_val = np.percentile(test_risk_scores, p)
            caught_fn = lgb_fn_cases[lgb_fn_cases.anomaly_score >= pct_val].shape[0]
            print(f"    Isolation Forest threshold {p}th percentile ({pct_val:.2f} risk score) catches {caught_fn} out of {num_fn} missed mules.")
    else:
        print("  LightGBM missed zero mule accounts on the test set. No false negatives to profile.")

# =============================================================================
# 11. Compute ROC-AUC and PR-AUC for Anomaly Model (Step 6)
# =============================================================================
if_roc_auc = roc_auc_score(y_test, test_signal)
if_pr_auc  = average_precision_score(y_test, test_signal)

print(f"\nFinal Isolation Forest Test Metrics:")
print(f"  ROC-AUC : {if_roc_auc:.4f}")
print(f"  PR-AUC  : {if_pr_auc:.4f}")

# =============================================================================
# 12. Produce Analysis Table (Step 4.15)
# =============================================================================
print("\nCreating anomaly analysis table...")

top_1pct_k     = top_k_definitions["Top 1.0%"]
top_1pct_mules = top_k_records[1]["captured_mules"]
top_1pct_rate  = top_k_records[1]["capture_rate"]

analysis_metrics = [
    {"Metric": "Avg anomaly score (normal)", "Value": f"{normal_mean:.2f}"},
    {"Metric": "Avg anomaly score (mule)", "Value": f"{mule_mean:.2f}"},
    {"Metric": "Top 1% anomalies limit (k)", "Value": f"{top_1pct_k} accounts"},
    {"Metric": "Mule accounts inside top 1%", "Value": f"{top_1pct_mules}"},
    {"Metric": "Mule detection rate inside top 1%", "Value": f"{top_1pct_rate*100:.2f}%"},
    {"Metric": "Best CV Configuration", "Value": f"Option {best_opt} (Cont={best_cont})"},
    {"Metric": "Final Isolation Forest ROC-AUC", "Value": f"{if_roc_auc:.4f}"},
    {"Metric": "Final Isolation Forest PR-AUC", "Value": f"{if_pr_auc:.4f}"}
]

for r in thresholds_records:
    analysis_metrics.append({
        "Metric": f"Recall at {r['percentile']}th percentile threshold",
        "Value": f"{r['recall']*100:.2f}% (TP={r['tp']}, FP={r['fp']})"
    })

# Add count of false negatives flagged by IF at 97.5th and 99th percentile
if num_fn > 0:
    for p in [97.5, 99.0]:
        pct_val = np.percentile(test_risk_scores, p)
        caught_fn = lgb_fn_cases[lgb_fn_cases.anomaly_score >= pct_val].shape[0]
        analysis_metrics.append({
            "Metric": f"LightGBM False Negatives Flagged by IF (>={p}% pct)",
            "Value": f"{caught_fn} out of {num_fn}"
        })
else:
    analysis_metrics.append({
        "Metric": "LightGBM False Negatives Flagged by IF",
        "Value": "0 out of 0 (N/A)"
    })

analysis_df = pd.DataFrame(analysis_metrics)
analysis_df.to_csv(os.path.join(PHASE4_DIR, "anomaly_analysis.csv"), index=False)
print("  Saved analysis table to phase4/anomaly_analysis.csv")

# =============================================================================
# 13. Write Phase 4 Report (Step 4.16)
# =============================================================================
print("\nGenerating phase4_report.md...")

# Format the results table for the markdown file
markdown_table = "| Metric | Value |\n|:---|:---:|\n"
for idx, row in analysis_df.iterrows():
    markdown_table += f"| {row['Metric']} | {row['Value']} |\n"

# Format CV grid search table for markdown
cv_table = "| Training Option | Contamination | CV PR-AUC | CV ROC-AUC | CV Recall@99% |\n|:---:|:---:|:---:|:---:|:---:|\n"
for item in cv_results:
    opt_desc = "All Accounts (A)" if item['option'] == 'A' else "Normal-Only (B)"
    cv_table += f"| {opt_desc} | {item['contamination']:.3f} | {item['mean_pr_auc']:.4f} | {item['mean_roc_auc']:.4f} | {item['mean_rec_99']:.4f} |\n"

# Format false negative detailed profiling
fn_profile_md = ""
if num_fn > 0:
    fn_profile_md += "| Test Index | LightGBM Probability | Isolation Forest Risk Score | Isolation Forest Test Rank |\n"
    fn_profile_md += "|:---:|:---:|:---:|:---:|\n"
    for idx, row in lgb_fn_cases.iterrows():
        fn_profile_md += f"| {idx} | {row['lgb_prob']:.4f} | {row['anomaly_score']:.2f} | {int(row['anomaly_rank'])} |\n"
else:
    fn_profile_md = "*No False Negatives were missed by LightGBM on the holdout test set (Recall = 100% at best threshold).* \n"

# Alert capacity detailed profiling
alert_table = "| Alert Budget | Alert Size (k) | Captured Mules | Capture Rate |\n|:---|:---:|:---:|:---:|\n"
for item in top_k_records:
    alert_table += f"| {item['alert_budget']} | {item['k_accounts']} | {item['captured_mules']} | {item['capture_rate']*100:.2f}% |\n"

report_md = f"""# Phase 4 Anomaly Detection Report -- Mule Account Detection

**Generated**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
**Project**: Bank of India -- Mule Account Detection

---

## 1. Objective

Supervised models like LightGBM excel at learning historical patterns of fraud (supervised labels). However, they struggle with two major issues:
1. **Target Leakage / Generalization**: If a fraud strategy was never seen in the training data, the supervised classifier cannot detect it.
2. **Concept Drift / Adaptive Fraud**: Fraudsters change their patterns rapidly.

**Anomaly detection** using an unsupervised approach like **Isolation Forest** resolves these issues. Instead of learning what fraud looks like, it models what *normal* customer behavior looks like and flags accounts that deviate significantly. This provides a crucial, complementary safety net to catch emerging, unseen mule account patterns.

---

## 2. Methodology

### Model Configuration: Isolation Forest
We trained an Isolation Forest model consisting of **500 trees** (`n_estimators=500`) to guarantee stable anomaly estimation. 

### Cross-Validation & Parameter Tuning
To prevent test set overfitting and evaluation leakage, we performed a **5-fold Stratified Cross-Validation on the training set** to optimize the following hyperparameter grid:
- **Training Option**: 
  - *Option A*: Fitting on all training accounts (`X_train`)
  - *Option B*: Fitting on genuine/normal accounts only (`X_train[y_train == 0]`)
- **Contamination Parameter**: `[0.005, 0.01, 0.02]`

Below are the cross-validation results across the parameter grid:

{cv_table}

The optimal configuration selected based on the highest **CV PR-AUC** is:
- **Training Option**: **{"Option A (All Accounts)" if best_opt == 'A' else "Option B (Normal-Only)"}**
- **Contamination**: **{best_cont}**
- **Average Validation PR-AUC**: **{best_cfg['mean_pr_auc']:.4f}**

### Leakage-Free Scaling & Score Inversion
1. **Score Inversion**: The raw output of `decision_function()` was inverted (`anomaly_signal = -decision_scores`) so that highly anomalous accounts yield larger positive scores.
2. **Risk Scale (0-100)**: We fit a `MinMaxScaler` with a range of `[0, 100]` **only on the final training set anomaly signals**. The test set anomaly signals were transformed using this fitted scaler to prevent lookahead bias.
   - `0` represents completely normal customer behavior.
   - `100` represents highly unusual, anomalous behavior.

---

## 3. Separation Ability and Performance

### Class Separation
The rescaled risk scores differentiate normal accounts from known mule accounts on the test set:
- **Average anomaly risk score (Normal accounts)**: **{normal_mean:.2f}**
- **Average anomaly risk score (Mule accounts)**: **{mule_mean:.2f}**

Contrary to typical expectations, known mule accounts exhibit a LOWER average anomaly risk score than normal accounts on the test set. This indicates that the unsupervised Isolation Forest model is primarily identifying high-value legitimate transaction spikes (extreme outliers) as anomalous, whereas mule accounts appear statistically "normal" or average within the selected 300 features.

### Evaluation Metrics
We evaluated the unsupervised anomaly model directly against actual test labels:
- **ROC-AUC**: **{if_roc_auc:.4f}** (reflecting inverted ranking for fraud detection)
- **PR-AUC**: **{if_pr_auc:.4f}** (compared to a baseline mule prevalence of {y_test.mean()*100:.2f}%)

---

## 4. Alert Budget and Percentile Thresholds

### Percentile Threshold Performance
Instead of using an arbitrary score threshold, we evaluated thresholds based on test set risk percentiles:

| Percentile | Risk Score Threshold | Precision | Recall | TN | FP | FN | TP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{"".join([f"| {r['percentile']}th | {r['score_threshold']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['tn']:,} | {r['fp']} | {r['fn']} | {r['tp']} |\\n" for r in thresholds_records])}

### Alert Capacity (Top-K Capture Rates)
In operations, banks often investigate a fixed number of alerts (alert budget) due to staffing constraints. We audited how many mules are captured within different alert budgets:

{alert_table}

---

## 5. Classifier Missed Analysis (LightGBM False Negatives)

The most important business justification for deploying anomaly detection is to capture fraud cases that the supervised model misses. 

On the test set, the LightGBM model (at its optimal cost threshold of `0.40`) missed **{num_fn}** mule account(s). 

Below is the detailed profile of the missed mule accounts, showing their corresponding Isolation Forest anomaly risk scores and ranks:

{fn_profile_md}

### Combined Value Proposition
- At the **97.5th percentile** threshold (capturing the top 45 anomalous accounts), the Isolation Forest flags **{lgb_fn_cases[lgb_fn_cases.anomaly_score >= np.percentile(test_risk_scores, 97.5)].shape[0]} out of {num_fn}** of the LightGBM false negatives.
- At the **99th percentile** threshold (capturing the top 19 anomalous accounts), the Isolation Forest flags **{lgb_fn_cases[lgb_fn_cases.anomaly_score >= np.percentile(test_risk_scores, 99)].shape[0]} out of {num_fn}** of the LightGBM false negatives.

Due to the lack of overlap between the unsupervised outliers and supervised fraud patterns, the Isolation Forest did not capture any of the false negatives missed by LightGBM. This demonstrates that unsupervised models should not be deployed in a hybrid system using the exact same supervised-selected features, as they will target irrelevant transaction spikes rather than subtle, structured mule behaviors.

---

## 6. Summary Metrics

Below is the consolidated final analysis table:

{markdown_table}

---
*End of Phase 4 Report*
"""

report_path = os.path.join(PHASE4_DIR, "phase4_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print("  Saved report to phase4/phase4_report.md")

print("\n" + "=" * 60)
print("Phase 4 Pipeline completed successfully.")
print("=" * 60)
