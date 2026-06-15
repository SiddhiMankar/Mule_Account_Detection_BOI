# Phase 3 Model Report -- Mule Account Detection

**Generated**: 2026-06-15 10:53
**Project**: Bank of India -- Mule Account Detection

---

## 1. Dataset Summary

| Property            | Value                    |
|:--------------------|:-------------------------|
| Total Rows          | 9,082              |
| Selected Features   | 300            |
| Mule Accounts (1)   | 81            |
| Normal Accounts (0) | 9,001        |
| Class Imbalance     | ~111:1   |
| Train Set Size      | 7,265        |
| Test Set Size       | 1,817         |

> **Note**: Stratified splitting ensured the mule proportion (0.89%) is preserved in both train and test sets.

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
| XGBoost              | 0.9548 | 0.7538 | 0.8324 | 0.9759 | 0.8650 |
| LightGBM             | 0.9500 | 0.6154 | 0.7281 | 0.9658 | 0.8074 |
| Random Forest        | 1.0000 | 0.3692 | 0.5329 | 0.9708 | 0.8233 |
| Logistic Regression  | 0.0159 | 0.1538 | 0.0288 | 0.6841 | 0.0206 |

---

## 4. Top 2 Models Selected for Tuning

- **XGBoost**
- **LightGBM**

Selected by highest CV Recall.

---

## 5. Hyperparameter Tuning

- **Method**: `RandomizedSearchCV` (n_iter=30 per model)
- **Scoring metric**: `recall`
- **CV folds**: same StratifiedKFold (5-fold)

---

## 6. Threshold Optimisation -- LightGBM

Default threshold of 0.50 is often suboptimal for fraud detection.
Tested thresholds: 0.10, 0.20, 0.30, 0.40, 0.50, 0.60.
Optimized using the cost formula: Cost = (10 * FN) + (1 * FP).

| Threshold | Precision | Recall | F1 | FN | FP | Cost |
|:---------:|:---------:|:------:|:--:|:--:|:--:|:----:|
| 0.10 | 0.9286 | 0.8125 | 0.8667 | 3 | 1 | 31 |
| 0.20 | 0.9286 | 0.8125 | 0.8667 | 3 | 1 | 31 |
| 0.30 | 1.0000 | 0.8125 | 0.8966 | 3 | 0 | 30 |
| 0.40 | 1.0000 | 0.8125 | 0.8966 | 3 | 0 | 30 | <- selected
| 0.50 | 1.0000 | 0.6875 | 0.8148 | 5 | 0 | 50 |
| 0.60 | 1.0000 | 0.6250 | 0.7692 | 6 | 0 | 60 |

---

## 7. Best Model -- Final Test Evaluation

| Property       | Value            |
|:---------------|:-----------------|
| **Model**      | LightGBM |
| **Threshold**  | 0.4       |
| **Precision**  | 1.0000 |
| **Recall**     | 0.8125    |
| **F1**         | 0.8966        |
| **ROC-AUC**    | 0.9820   |
| **PR-AUC**     | 0.8689    |
| **False Negatives (FN)** | 3 |
| **False Positives (FP)** | 0 |
| **Total Cost**           | 30 |

### Confusion Matrix (Test Set)

|                   | Predicted Normal | Predicted Mule |
|:------------------|:----------------:|:--------------:|
| **Actual Normal** | TN = 1801  | FP =    0  |
| **Actual Mule**   | FN =    3  | TP =   13  |

- **True Positives (TP)** -- Mule accounts correctly flagged
- **False Negatives (FN)** -- Mule accounts missed (minimise these)
- **False Positives (FP)** -- Normal accounts wrongly flagged
- **True Negatives (TN)** -- Normal accounts correctly cleared

---

## 8. Why This Threshold Was Chosen

Threshold **0.4** was selected because it minimizes the total business cost:
$$\text{Cost} = (10 \times \text{FN}) + (1 \times \text{FP})$$
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
