"""
hyperparameter_tuning.py
------------------------
Phase 3, Step 3.10 — Stand-alone Hyperparameter Tuning Script
Bank of India — Mule Account Detection

Purpose:
  Run an exhaustive RandomizedSearchCV sweep for XGBoost and LightGBM
  (or whichever two models were top-ranked after baseline evaluation).
  Results are logged to a CSV and the best estimators are saved as .pkl files.

Usage:
  python phase3/hyperparameter_tuning.py

Outputs (all in phase3/):
  tuning_results_xgboost.csv
  tuning_results_lightgbm.csv
  tuned_xgboost.pkl
  tuned_lightgbm.pkl
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
import time
import warnings
import joblib
import json
import numpy as np
import pandas as pd

# Force UTF-8 stdout so print() works even on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = str(PROJECT_ROOT)
# PHASE3_DIR is imported from config.paths
os.makedirs(PHASE3_DIR, exist_ok=True)

RANDOM_STATE = 42
start_time   = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading X_final.csv and y_final.csv ...")
X = pd.read_csv(os.path.join(DATA_PHASE2, "X_final.csv"))
y = pd.read_csv(os.path.join(DATA_PHASE2, "y_final.csv")).squeeze()

normal_count = int((y == 0).sum())
mule_count   = int(y.sum())
scale_pos_weight = round(normal_count / mule_count)

print(f"  Dataset : {X.shape[0]} rows x {X.shape[1]} features")
print(f"  Mule    : {mule_count}  |  Normal : {normal_count}  |  spw={scale_pos_weight}")

# ─────────────────────────────────────────────────────────────────────────────
# Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
)
print(f"  Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation Strategy
# ─────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# Helper — evaluate a fitted model on a hold-out set
# ─────────────────────────────────────────────────────────────────────────────
def holdout_metrics(model, X_eval, y_eval, threshold=0.5):
    proba = model.predict_proba(X_eval)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_eval, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    cost = 10 * fn + fp
    return {
        "Precision" : round(precision_score(y_eval, preds, zero_division=0), 4),
        "Recall"    : round(recall_score(y_eval, preds, zero_division=0), 4),
        "F1"        : round(f1_score(y_eval, preds, zero_division=0), 4),
        "ROC-AUC"   : round(roc_auc_score(y_eval, proba), 4),
        "PR-AUC"    : round(average_precision_score(y_eval, proba), 4),
        "FN"        : int(fn),
        "FP"        : int(fp),
        "Cost"      : int(cost),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Helper — run tuning and save results
# ─────────────────────────────────────────────────────────────────────────────
def run_tuning(name, estimator, param_distributions, n_iter=30, scoring="recall"):
    print(f"\n{'='*60}")
    print(f"  Tuning {name}  (n_iter={n_iter}, scoring={scoring})")
    print(f"{'='*60}")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    # ── Save CV results ────────────────────────────────────────────────────
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")

    # Keep only useful columns
    keep_cols = [c for c in results_df.columns if not c.startswith("split")]
    results_df = results_df[keep_cols]

    out_csv = os.path.join(PHASE3_DIR, f"tuning_results_{name.lower().replace(' ', '_')}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"  [DONE] Saved {out_csv}")

    # ── Print best params ──────────────────────────────────────────────────
    print(f"\n  Best CV {scoring}: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    # ── Evaluate on test set ───────────────────────────────────────────────
    best_est = search.best_estimator_
    test_met = holdout_metrics(best_est, X_test, y_test, threshold=0.5)
    print(f"  Test-set metrics (threshold=0.50): {test_met}")

    # ── Save the best estimator ────────────────────────────────────────────
    pkl_path = os.path.join(PHASE3_DIR, f"tuned_{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(best_est, pkl_path)
    print(f"  [DONE] Saved {pkl_path}")

    return best_est, search.best_params_, search.best_score_, test_met


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Tuning
# ─────────────────────────────────────────────────────────────────────────────
xgb_best = None
try:
    from xgboost import XGBClassifier

    xgb_param_grid = {
        "n_estimators"    : [200, 400, 600],
        "max_depth"       : [3, 5, 7],
        "learning_rate"   : [0.01, 0.05, 0.1],
        "subsample"       : [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "gamma"           : [0, 0.1, 0.3],
        "min_child_weight": [1, 3, 5],
    }

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1,
    )

    xgb_best, xgb_params, xgb_cv_score, xgb_test = run_tuning(
        "XGBoost", xgb_base, xgb_param_grid, n_iter=30, scoring="recall"
    )

except ImportError:
    print("[WARN] xgboost not installed - skipping XGBoost tuning.")

# ─────────────────────────────────────────────────────────────────────────────
# LightGBM Tuning
# ─────────────────────────────────────────────────────────────────────────────
lgb_best = None
try:
    from lightgbm import LGBMClassifier

    lgb_param_grid = {
        "n_estimators"     : [200, 400, 600],
        "num_leaves"       : [31, 63, 127],
        "learning_rate"    : [0.01, 0.05, 0.1],
        "feature_fraction" : [0.7, 0.8, 1.0],
        "bagging_fraction" : [0.7, 0.8, 1.0],
        "bagging_freq"     : [0, 5, 10],
        "min_child_samples": [5, 10, 20],
        "reg_alpha"        : [0, 0.1, 0.5],
        "reg_lambda"       : [0, 0.1, 0.5],
    }

    lgb_base = LGBMClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_best, lgb_params, lgb_cv_score, lgb_test = run_tuning(
        "LightGBM", lgb_base, lgb_param_grid, n_iter=30, scoring="recall"
    )

except ImportError:
    print("[WARN] lightgbm not installed - skipping LightGBM tuning.")

# ─────────────────────────────────────────────────────────────────────────────
# Threshold Sweep for tuned models
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
tuned_candidates = {}
if xgb_best is not None:
    tuned_candidates["XGBoost"] = xgb_best
if lgb_best is not None:
    tuned_candidates["LightGBM"] = lgb_best

threshold_records = []
best_thresholds   = {}

print("\n" + "="*60)
print("Threshold Sweep on Test Set")
print("="*60)

for name, model in tuned_candidates.items():
    proba = model.predict_proba(X_test)[:, 1]
    best_cost = float("inf")
    best_thr = 0.5
    for thr in THRESHOLDS:
        preds = (proba >= thr).astype(int)
        rec  = recall_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)
        cm   = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        cost = 10 * fn + fp
        print(f"  {name}  thr={thr:.2f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  FN={fn}  FP={fp}  Cost={cost}")
        threshold_records.append({
            "Model": name, "Threshold": thr,
            "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4),
            "FN": int(fn), "FP": int(fp), "Cost": int(cost),
        })
        if cost < best_cost:
            best_cost = cost
            best_thr = thr
    best_thresholds[name] = best_thr
    print(f"  -> Best threshold for {name}: {best_thr} (Cost={best_cost})\n")

if threshold_records:
    thr_df = pd.DataFrame(threshold_records)
    thr_df.to_csv(os.path.join(PHASE3_DIR, "threshold_analysis.csv"), index=False)
    print("[DONE] Updated phase3/threshold_analysis.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Pick global best model and save threshold meta
# ─────────────────────────────────────────────────────────────────────────────
if tuned_candidates:
    best_model_name = min(
        tuned_candidates,
        key=lambda n: holdout_metrics(
            tuned_candidates[n], X_test, y_test,
            threshold=best_thresholds.get(n, 0.5),
        )["Cost"],
    )
    best_model = tuned_candidates[best_model_name]
    best_thr   = best_thresholds.get(best_model_name, 0.5)

    joblib.dump(best_model, os.path.join(PHASE3_DIR, "best_model.pkl"))
    print(f"\n[DONE] Saved best_model.pkl  ({best_model_name})")

    meta = {"model": best_model_name, "threshold": best_thr}
    with open(os.path.join(PHASE3_DIR, "best_threshold.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print(f"[DONE] Saved best_threshold.json  -> threshold={best_thr}")

    final_m = holdout_metrics(best_model, X_test, y_test, threshold=best_thr)
    print(f"\nFinal Test Metrics - {best_model_name} (thr={best_thr})")
    for k, v in final_m.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.4f}")
        else:
            print(f"  {k:12s}: {v}")

elapsed = time.time() - start_time
print(f"\n[DONE] Hyperparameter tuning completed in {elapsed/60:.1f} minutes.")
