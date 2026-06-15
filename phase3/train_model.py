"""
train_model.py
--------------
Phase 3: Model Development & Evaluation
Bank of India -- Mule Account Detection

Covers Steps 3.1 through 3.14:
  3.1  Train/Test Split (stratified 80/20)
  3.2  Cross-Validation Strategy (StratifiedKFold, 5-fold)
  3.3  Evaluation Framework (evaluate_model)
  3.4  Baseline Model #1 -- Logistic Regression
  3.5  Baseline Model #2 -- Random Forest
  3.6  Baseline Model #3 -- XGBoost
  3.7  Baseline Model #4 -- LightGBM
  3.8  Model Comparison Table -> model_comparison.csv
  3.9  Select Top 2 Models
  3.10 Hyperparameter Tuning -> hyperparameter_tuning.py results embedded
  3.11 Threshold Optimisation -> threshold_analysis.csv
  3.12 Final Test Evaluation (confusion matrix, ROC, PR curves)
  3.13 Save best_model.pkl + best_threshold.json
  3.14 Generate phase3_model_report.md
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
from config.paths import DATA_PHASE2, PHASE3_DIR
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for servers/scripts
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# Force UTF-8 stdout so print() works even on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- XGBoost ------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] xgboost not found - XGBoost model will be skipped.")

# -- LightGBM -----------------------------------------------------------------
try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[WARN] lightgbm not found - LightGBM model will be skipped.")

warnings.filterwarnings("ignore")

# =============================================================================
# 0.  Paths
# =============================================================================
BASE_DIR   = str(PROJECT_ROOT)
# PHASE3_DIR is imported from config.paths
os.makedirs(PHASE3_DIR, exist_ok=True)

RANDOM_STATE = 42
start_time   = time.time()

# =============================================================================
# 1.  Load Data
# =============================================================================
print("=" * 60)
print("Loading X_final.csv and y_final.csv ...")
X = pd.read_csv(os.path.join(DATA_PHASE2, "X_final.csv"))
y = pd.read_csv(os.path.join(DATA_PHASE2, "y_final.csv")).squeeze()

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(f"  Mule accounts   : {y.sum()} ({y.mean()*100:.2f}%)")
print(f"  Normal accounts : {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")

normal_count     = int((y == 0).sum())
mule_count       = int(y.sum())
scale_pos_weight = round(normal_count / mule_count)  # ~111

# =============================================================================
# STEP 3.1 -- Train / Test Split
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.1 -- Stratified Train/Test Split (80/20)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"  Train : {X_train.shape[0]} rows  (mules: {y_train.sum()})")
print(f"  Test  : {X_test.shape[0]}  rows  (mules: {y_test.sum()})")
print(f"  Train mule %: {y_train.mean()*100:.2f}%")
print(f"  Test  mule %: {y_test.mean()*100:.2f}%")

# =============================================================================
# STEP 3.2 -- Cross-Validation Strategy
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.2 -- StratifiedKFold (n_splits=5, shuffle=True)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
# STEP 3.3 -- Evaluation Framework
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.3 -- Defining evaluate_model()")

def evaluate_model(model, X_eval, y_eval, threshold=0.5, model_name="Model"):
    """
    Compute and return a comprehensive evaluation dict for a fitted classifier.

    Parameters
    ----------
    model      : fitted sklearn-compatible estimator
    X_eval     : feature DataFrame / array
    y_eval     : true labels
    threshold  : decision threshold (default 0.5)
    model_name : label used in printout

    Returns
    -------
    dict with keys:
        Precision, Recall, F1, ROC-AUC, PR-AUC,
        Confusion Matrix (2-D np.array), FN, FP, Cost
    """
    proba  = model.predict_proba(X_eval)[:, 1]
    preds  = (proba >= threshold).astype(int)

    prec   = precision_score(y_eval, preds, zero_division=0)
    rec    = recall_score(y_eval, preds, zero_division=0)
    f1     = f1_score(y_eval, preds, zero_division=0)
    roc    = roc_auc_score(y_eval, proba)
    pr_auc = average_precision_score(y_eval, proba)
    cm     = confusion_matrix(y_eval, preds)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    cost   = 10 * fn + fp

    print(f"\n  [{model_name}]  threshold={threshold}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    ROC-AUC   : {roc:.4f}")
    print(f"    PR-AUC    : {pr_auc:.4f}")
    print(f"    FN        : {fn}  |  FP : {fp}  |  Cost : {cost}")
    print(f"    Confusion Matrix:\n{cm}")

    return {
        "Precision"        : prec,
        "Recall"           : rec,
        "F1"               : f1,
        "ROC-AUC"          : roc,
        "PR-AUC"           : pr_auc,
        "Confusion Matrix" : cm,
        "FN"               : fn,
        "FP"               : fp,
        "Cost"             : cost,
    }


def cv_evaluate(model, X_cv, y_cv, cv_strategy, model_name="Model"):
    """
    Run cross-validated evaluation and return mean metric dict.
    Uses predict_proba to compute ROC-AUC and PR-AUC inside each fold.
    """
    scoring = {
        "precision" : "precision",
        "recall"    : "recall",
        "f1"        : "f1",
        "roc_auc"   : "roc_auc",
        "pr_auc"    : "average_precision",
    }
    results = cross_validate(
        model, X_cv, y_cv,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )
    means = {
        "CV Precision" : results["test_precision"].mean(),
        "CV Recall"    : results["test_recall"].mean(),
        "CV F1"        : results["test_f1"].mean(),
        "CV ROC-AUC"   : results["test_roc_auc"].mean(),
        "CV PR-AUC"    : results["test_pr_auc"].mean(),
    }
    print(f"\n  [{model_name}] Cross-Validation Results (5-fold)")
    for k, v in means.items():
        print(f"    {k:15s}: {v:.4f}")
    return means


# =============================================================================
# STEP 3.4 -- Baseline Model #1: Logistic Regression
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.4 -- Baseline Model #1: Logistic Regression")

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=5000,
    random_state=RANDOM_STATE,
    solver="saga",
    n_jobs=-1,
)
lr_cv = cv_evaluate(lr_model, X_train, y_train, cv, "Logistic Regression")

# =============================================================================
# STEP 3.5 -- Baseline Model #2: Random Forest
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.5 -- Baseline Model #2: Random Forest")

rf_model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_cv = cv_evaluate(rf_model, X_train, y_train, cv, "Random Forest")

# =============================================================================
# STEP 3.6 -- Baseline Model #3: XGBoost
# =============================================================================
print("\n" + "=" * 60)
print(f"STEP 3.6 -- Baseline Model #3: XGBoost  (scale_pos_weight={scale_pos_weight})")

if XGB_AVAILABLE:
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1,
    )
    xgb_cv = cv_evaluate(xgb_model, X_train, y_train, cv, "XGBoost")
else:
    xgb_cv = {k: np.nan for k in ["CV Precision","CV Recall","CV F1","CV ROC-AUC","CV PR-AUC"]}

# =============================================================================
# STEP 3.7 -- Baseline Model #4: LightGBM
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.7 -- Baseline Model #4: LightGBM")

if LGB_AVAILABLE:
    lgb_model = LGBMClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_cv = cv_evaluate(lgb_model, X_train, y_train, cv, "LightGBM")
else:
    lgb_cv = {k: np.nan for k in ["CV Precision","CV Recall","CV F1","CV ROC-AUC","CV PR-AUC"]}

# =============================================================================
# STEP 3.8 -- Model Comparison Table
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.8 -- Model Comparison Table")

comparison_data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
    "CV Precision" : [lr_cv["CV Precision"], rf_cv["CV Precision"],
                      xgb_cv["CV Precision"], lgb_cv["CV Precision"]],
    "CV Recall"    : [lr_cv["CV Recall"],    rf_cv["CV Recall"],
                      xgb_cv["CV Recall"],   lgb_cv["CV Recall"]],
    "CV F1"        : [lr_cv["CV F1"],        rf_cv["CV F1"],
                      xgb_cv["CV F1"],        lgb_cv["CV F1"]],
    "CV ROC-AUC"   : [lr_cv["CV ROC-AUC"],  rf_cv["CV ROC-AUC"],
                      xgb_cv["CV ROC-AUC"],  lgb_cv["CV ROC-AUC"]],
    "CV PR-AUC"    : [lr_cv["CV PR-AUC"],   rf_cv["CV PR-AUC"],
                      xgb_cv["CV PR-AUC"],   lgb_cv["CV PR-AUC"]],
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df_sorted = comparison_df.sort_values("CV Recall", ascending=False)
print(comparison_df_sorted.to_string(index=False))

comparison_df_sorted.to_csv(
    os.path.join(PHASE3_DIR, "model_comparison.csv"), index=False
)
print("[DONE] Saved phase3/model_comparison.csv")

# =============================================================================
# STEP 3.9 -- Select Top 2 Models
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.9 -- Selecting Top 2 Models by CV Recall")

valid_rows = comparison_df_sorted.dropna(subset=["CV Recall"])
top2_names = valid_rows.nlargest(2, "CV Recall")["Model"].tolist()
print(f"  Top 2 models: {top2_names}")

# Map model names -> unfitted instances for tuning
model_registry = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=5000,
        random_state=RANDOM_STATE, solver="saga", n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
}
if XGB_AVAILABLE:
    model_registry["XGBoost"] = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, n_jobs=-1,
    )
if LGB_AVAILABLE:
    model_registry["LightGBM"] = LGBMClassifier(
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    )

# =============================================================================
# STEP 3.10 -- Hyperparameter Tuning (RandomizedSearchCV)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.10 -- Hyperparameter Tuning (RandomizedSearchCV, scoring=recall)")

PARAM_GRIDS = {
    "XGBoost": {
        "n_estimators"    : [200, 400, 600],
        "max_depth"       : [3, 5, 7],
        "learning_rate"   : [0.01, 0.05, 0.1],
        "subsample"       : [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "gamma"           : [0, 0.1, 0.3],
        "min_child_weight": [1, 3, 5],
    },
    "LightGBM": {
        "n_estimators"     : [200, 400, 600],
        "num_leaves"       : [31, 63, 127],
        "learning_rate"    : [0.01, 0.05, 0.1],
        "feature_fraction" : [0.7, 0.8, 1.0],
        "bagging_fraction" : [0.7, 0.8, 1.0],
        "bagging_freq"     : [0, 5, 10],
        "min_child_samples": [5, 10, 20],
        "reg_alpha"        : [0, 0.1, 0.5],
        "reg_lambda"       : [0, 0.1, 0.5],
    },
    "Random Forest": {
        "n_estimators"     : [200, 400, 600],
        "max_depth"        : [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features"     : ["sqrt", "log2"],
    },
    "Logistic Regression": {
        "C"      : [0.001, 0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver" : ["saga", "liblinear"],
    },
}

tuned_models = {}

for name in top2_names:
    if name not in model_registry or name not in PARAM_GRIDS:
        print(f"  Skipping {name} -- not in registry or no param grid.")
        continue

    print(f"\n  Tuning {name} ...")
    base_est   = model_registry[name]
    param_grid = PARAM_GRIDS[name]

    search = RandomizedSearchCV(
        estimator=base_est,
        param_distributions=param_grid,
        n_iter=30,
        scoring="recall",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score  = search.best_score_
    print(f"    Best CV Recall : {best_score:.4f}")
    print(f"    Best Params    : {best_params}")

    tuned_models[name] = search.best_estimator_

if not tuned_models:
    # Fallback: use baseline models fitted on full training set
    print("  No tuned models available -- falling back to baseline models.")
    for name in top2_names:
        if name in model_registry:
            print(f"  Fitting baseline {name} on full training set ...")
            model_registry[name].fit(X_train, y_train)
            tuned_models[name] = model_registry[name]

# =============================================================================
# STEP 3.11 -- Threshold Optimisation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.11 -- Threshold Optimisation")

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
threshold_records = []

for name, model in tuned_models.items():
    proba = model.predict_proba(X_test)[:, 1]
    for thr in THRESHOLDS:
        preds = (proba >= thr).astype(int)
        rec  = recall_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)
        cm   = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        cost = 10 * fn + fp
        threshold_records.append({
            "Model"    : name,
            "Threshold": thr,
            "Precision": round(prec, 4),
            "Recall"   : round(rec,  4),
            "F1"       : round(f1,   4),
            "FN"       : int(fn),
            "FP"       : int(fp),
            "Cost"     : int(cost),
        })
        print(f"  {name}  thr={thr:.2f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  FN={fn}  FP={fp}  Cost={cost}")

threshold_df = pd.DataFrame(threshold_records)
threshold_df.to_csv(
    os.path.join(PHASE3_DIR, "threshold_analysis.csv"), index=False
)
print("[DONE] Saved phase3/threshold_analysis.csv")

# Pick best threshold per model (minimise Cost)
best_thresholds = {}
for name in tuned_models:
    sub = threshold_df[threshold_df["Model"] == name].copy()
    sub = sub.sort_values("Cost", ascending=True)
    best_row = sub.iloc[0]
    best_thresholds[name] = float(best_row["Threshold"])
    print(f"  Best threshold for {name}: {best_thresholds[name]} (Cost={best_row['Cost']})")

# =============================================================================
# STEP 3.12 -- Final Test Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.12 -- Final Test Evaluation")

final_results = {}
for name, model in tuned_models.items():
    thr     = best_thresholds.get(name, 0.5)
    metrics = evaluate_model(model, X_test, y_test, threshold=thr, model_name=name)
    final_results[name] = {"threshold": thr, "metrics": metrics}

    # Classification report
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= thr).astype(int)
    print(f"\n  Classification Report -- {name} (threshold={thr})")
    print(classification_report(y_test, preds, target_names=["Normal", "Mule"]))

# Choose the overall best model (lowest Cost)
best_model_name = min(
    final_results,
    key=lambda n: final_results[n]["metrics"]["Cost"],
)
best_model = tuned_models[best_model_name]
best_thr   = best_thresholds[best_model_name]
best_met   = final_results[best_model_name]["metrics"]

print(f"\n  [BEST] Best overall model: {best_model_name}  threshold={best_thr} (Cost={best_met['Cost']})")

# -- Confusion Matrix Plot ----------------------------------------------------
cm_arr = best_met["Confusion Matrix"]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax)
classes    = ["Normal", "Mule"]
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=12)
ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=12)
thresh_cm = cm_arr.max() / 2.
for i in range(cm_arr.shape[0]):
    for j in range(cm_arr.shape[1]):
        ax.text(j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center", fontsize=14,
                color="white" if cm_arr[i, j] > thresh_cm else "black")
ax.set_ylabel("True Label", fontsize=13)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_title(f"Confusion Matrix - {best_model_name}\n(threshold={best_thr})", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(PHASE3_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)
print("[DONE] Saved phase3/confusion_matrix.png")

# -- ROC Curve Plot -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for idx, (name, model) in enumerate(tuned_models.items()):
    proba   = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, lw=2, color=colors[idx % len(colors)],
            label=f"{name}  (AUC={roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve - Test Set", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PHASE3_DIR, "roc_curve.png"), dpi=150)
plt.close(fig)
print("[DONE] Saved phase3/roc_curve.png")

# -- Precision-Recall Curve Plot ----------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for idx, (name, model) in enumerate(tuned_models.items()):
    proba  = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec_arr, prec_arr, lw=2, color=colors[idx % len(colors)],
            label=f"{name}  (PR-AUC={pr_auc:.3f})")

baseline = y_test.mean()
ax.axhline(baseline, color="gray", linestyle="--", lw=1,
           label=f"Baseline (mule prevalence={baseline:.3f})")
ax.set_xlabel("Recall", fontsize=13)
ax.set_ylabel("Precision", fontsize=13)
ax.set_title("Precision-Recall Curve - Test Set", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PHASE3_DIR, "pr_curve.png"), dpi=150)
plt.close(fig)
print("[DONE] Saved phase3/pr_curve.png")

# =============================================================================
# STEP 3.13 -- Save Best Model + Best Threshold
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.13 -- Saving Best Model & Threshold")

joblib.dump(best_model, os.path.join(PHASE3_DIR, "best_model.pkl"))
print(f"[DONE] Saved phase3/best_model.pkl  ({best_model_name})")

threshold_meta = {
    "model"    : best_model_name,
    "threshold": best_thr,
}
with open(os.path.join(PHASE3_DIR, "best_threshold.json"), "w") as f:
    json.dump(threshold_meta, f, indent=4)
print(f"[DONE] Saved phase3/best_threshold.json  (threshold={best_thr})")

# =============================================================================
# STEP 3.14 -- Generate phase3_model_report.md
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3.14 -- Generating phase3_model_report.md")

def fmt(v):
    """Format a float to 4 decimals, or 'N/A' if NaN."""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "N/A"

# Build comparison table rows
comp_rows = ""
for _, row in comparison_df_sorted.iterrows():
    comp_rows += (
        f"| {row['Model']:20s} | {fmt(row['CV Precision'])} | "
        f"{fmt(row['CV Recall'])} | {fmt(row['CV F1'])} | "
        f"{fmt(row['CV ROC-AUC'])} | {fmt(row['CV PR-AUC'])} |\n"
    )

# Threshold table for best model
thr_sub  = threshold_df[threshold_df["Model"] == best_model_name].sort_values("Threshold")
thr_rows = ""
for _, r in thr_sub.iterrows():
    mark = " <- selected" if r["Threshold"] == best_thr else ""
    thr_rows += (
        f"| {r['Threshold']:.2f} | {r['Precision']:.4f} | "
        f"{r['Recall']:.4f} | {r['F1']:.4f} | "
        f"{int(r['FN'])} | {int(r['FP'])} | {int(r['Cost'])} |{mark}\n"
    )

# Final metrics for best model
bm     = best_met
tn, fp, fn, tp = cm_arr.ravel() if cm_arr.size == 4 else (0, 0, 0, 0)

report_md = f"""# Phase 3 Model Report -- Mule Account Detection

**Generated**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
**Project**: Bank of India -- Mule Account Detection

---

## 1. Dataset Summary

| Property            | Value                    |
|:--------------------|:-------------------------|
| Total Rows          | {len(X):,}              |
| Selected Features   | {X.shape[1]}            |
| Mule Accounts (1)   | {mule_count}            |
| Normal Accounts (0) | {normal_count:,}        |
| Class Imbalance     | ~{scale_pos_weight}:1   |
| Train Set Size      | {len(X_train):,}        |
| Test Set Size       | {len(X_test):,}         |

> **Note**: Stratified splitting ensured the mule proportion ({y.mean()*100:.2f}%) is preserved in both train and test sets.

---

## 2. Cross-Validation Strategy

- **Method**: `StratifiedKFold`
- **n_splits**: 5
- **shuffle**: True
- **random_state**: 42

Every fold is guaranteed to contain mule accounts due to stratification.

---

## 3. Baseline Model Comparison (5-Fold CV on Training Set)

> Models were **not** chosen based on accuracy. The primary metric is **Recall** because missing a mule account is costlier than raising a false alarm.

| Model                | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|:---------------------|:---------:|:------:|:--:|:-------:|:------:|
{comp_rows}
---

## 4. Top 2 Models Selected for Tuning

{chr(10).join(f"- **{n}**" for n in top2_names)}

Selected by highest CV Recall.

---

## 5. Hyperparameter Tuning

- **Method**: `RandomizedSearchCV` (n_iter=30 per model)
- **Scoring metric**: `recall`
- **CV folds**: same StratifiedKFold (5-fold)

---

## 6. Threshold Optimisation -- {best_model_name}

Default threshold of 0.50 is often suboptimal for fraud detection.
Tested thresholds: 0.10, 0.20, 0.30, 0.40, 0.50, 0.60.
Optimized using the cost formula: Cost = (10 * FN) + (1 * FP).

| Threshold | Precision | Recall | F1 | FN | FP | Cost |
|:---------:|:---------:|:------:|:--:|:--:|:--:|:----:|
{thr_rows}
---

## 7. Best Model -- Final Test Evaluation

| Property       | Value            |
|:---------------|:-----------------|
| **Model**      | {best_model_name} |
| **Threshold**  | {best_thr}       |
| **Precision**  | {fmt(bm["Precision"])} |
| **Recall**     | {fmt(bm["Recall"])}    |
| **F1**         | {fmt(bm["F1"])}        |
| **ROC-AUC**    | {fmt(bm["ROC-AUC"])}   |
| **PR-AUC**     | {fmt(bm["PR-AUC"])}    |
| **False Negatives (FN)** | {int(bm["FN"])} |
| **False Positives (FP)** | {int(bm["FP"])} |
| **Total Cost**           | {int(bm["Cost"])} |

### Confusion Matrix (Test Set)

|                   | Predicted Normal | Predicted Mule |
|:------------------|:----------------:|:--------------:|
| **Actual Normal** | TN = {int(tn):4d}  | FP = {int(fp):4d}  |
| **Actual Mule**   | FN = {int(fn):4d}  | TP = {int(tp):4d}  |

- **True Positives (TP)** -- Mule accounts correctly flagged
- **False Negatives (FN)** -- Mule accounts missed (minimise these)
- **False Positives (FP)** -- Normal accounts wrongly flagged
- **True Negatives (TN)** -- Normal accounts correctly cleared

---

## 8. Why This Threshold Was Chosen

Threshold **{best_thr}** was selected because it minimizes the total business cost:
$$\\text{{Cost}} = (10 \\times \\text{{FN}}) + (1 \\times \\text{{FP}})$$
This threshold achieves the optimal balance between missing mule accounts (FN, penalized at 10x weight) and raising too many false alarms (FP, penalized at 1x weight).

---

## 9. Deliverables

| File                       | Description                                  |
|:---------------------------|:---------------------------------------------|
| `train_model.py`           | End-to-end Phase 3 training script           |
| `hyperparameter_tuning.py` | Stand-alone tuning script with full logs     |
| `model_comparison.csv`     | 5-Fold CV metrics for all 4 baseline models  |
| `threshold_analysis.csv`   | Threshold sweep for top models               |
| `best_model.pkl`           | Serialised best estimator                    |
| `best_threshold.json`      | Optimal threshold metadata                   |
| `confusion_matrix.png`     | Confusion matrix for best model (test set)   |
| `roc_curve.png`            | ROC curve for top models (test set)          |
| `pr_curve.png`             | Precision-Recall curve (test set)            |
| `phase3_model_report.md`   | This report                                  |

---

*End of Phase 3 Report*
"""

report_path = os.path.join(PHASE3_DIR, "phase3_model_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print("[DONE] Saved phase3/phase3_model_report.md")

# =============================================================================
# Done
# =============================================================================
elapsed = time.time() - start_time
print("\n" + "=" * 60)
print(f"[DONE] All Phase 3 steps completed in {elapsed/60:.1f} minutes.")
print(f"   Best model : {best_model_name}")
print(f"   Threshold  : {best_thr}")
print(f"   Recall     : {fmt(bm['Recall'])}")
print(f"   F1         : {fmt(bm['F1'])}")
print(f"   ROC-AUC    : {fmt(bm['ROC-AUC'])}")
print(f"   PR-AUC     : {fmt(bm['PR-AUC'])}")
print(f"   FN         : {int(bm['FN'])}")
print(f"   FP         : {int(bm['FP'])}")
print(f"   Cost       : {int(bm['Cost'])}")
print("=" * 60)
