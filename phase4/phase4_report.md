# Phase 4 Anomaly Detection Report -- Mule Account Detection

**Generated**: 2026-06-15 10:55
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

| Training Option | Contamination | CV PR-AUC | CV ROC-AUC | CV Recall@99% |
|:---:|:---:|:---:|:---:|:---:|
| All Accounts (A) | 0.005 | 0.0087 | 0.4316 | 0.0000 |
| All Accounts (A) | 0.010 | 0.0087 | 0.4316 | 0.0000 |
| All Accounts (A) | 0.020 | 0.0087 | 0.4316 | 0.0000 |
| Normal-Only (B) | 0.005 | 0.0088 | 0.4235 | 0.0000 |
| Normal-Only (B) | 0.010 | 0.0088 | 0.4235 | 0.0000 |
| Normal-Only (B) | 0.020 | 0.0088 | 0.4235 | 0.0000 |


The optimal configuration selected based on the highest **CV PR-AUC** is:
- **Training Option**: **Option B (Normal-Only)**
- **Contamination**: **0.005**
- **Average Validation PR-AUC**: **0.0088**

### Leakage-Free Scaling & Score Inversion
1. **Score Inversion**: The raw output of `decision_function()` was inverted (`anomaly_signal = -decision_scores`) so that highly anomalous accounts yield larger positive scores.
2. **Risk Scale (0-100)**: We fit a `MinMaxScaler` with a range of `[0, 100]` **only on the final training set anomaly signals**. The test set anomaly signals were transformed using this fitted scaler to prevent lookahead bias.
   - `0` represents completely normal customer behavior.
   - `100` represents highly unusual, anomalous behavior.

---

## 3. Separation Ability and Performance

### Class Separation
The rescaled risk scores differentiate normal accounts from known mule accounts on the test set:
- **Average anomaly risk score (Normal accounts)**: **24.18**
- **Average anomaly risk score (Mule accounts)**: **17.74**

Contrary to typical expectations, known mule accounts exhibit a LOWER average anomaly risk score than normal accounts on the test set. This indicates that the unsupervised Isolation Forest model is primarily identifying high-value legitimate transaction spikes (extreme outliers) as anomalous, whereas mule accounts appear statistically "normal" or average within the selected 300 features.

### Evaluation Metrics
We evaluated the unsupervised anomaly model directly against actual test labels:
- **ROC-AUC**: **0.3305** (reflecting inverted ranking for fraud detection)
- **PR-AUC**: **0.0084** (compared to a baseline mule prevalence of 0.88%)

---

## 4. Alert Budget and Percentile Thresholds

### Percentile Threshold Performance
Instead of using an arbitrary score threshold, we evaluated thresholds based on test set risk percentiles:

| Percentile | Risk Score Threshold | Precision | Recall | TN | FP | FN | TP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 95th | 49.52 | 0.0110 | 0.0625 | 1,711 | 90 | 15 | 1 |\n| 97.5th | 54.41 | 0.0217 | 0.0625 | 1,756 | 45 | 15 | 1 |\n| 99th | 60.93 | 0.0000 | 0.0000 | 1,782 | 19 | 16 | 0 |\n

### Alert Capacity (Top-K Capture Rates)
In operations, banks often investigate a fixed number of alerts (alert budget) due to staffing constraints. We audited how many mules are captured within different alert budgets:

| Alert Budget | Alert Size (k) | Captured Mules | Capture Rate |
|:---|:---:|:---:|:---:|
| Top 0.5% | 10 | 0 | 0.00% |
| Top 1.0% | 19 | 0 | 0.00% |
| Top 100 Accounts | 100 | 1 | 6.25% |


---

## 5. Classifier Missed Analysis (LightGBM False Negatives)

The most important business justification for deploying anomaly detection is to capture fraud cases that the supervised model misses. 

On the test set, the LightGBM model (at its optimal cost threshold of `0.40`) missed **3** mule account(s). 

Below is the detailed profile of the missed mule accounts, showing their corresponding Isolation Forest anomaly risk scores and ranks:

| Test Index | LightGBM Probability | Isolation Forest Risk Score | Isolation Forest Test Rank |
|:---:|:---:|:---:|:---:|
| 1435 | 0.0000 | 8.21 | 1715 |
| 4381 | 0.0001 | 22.51 | 883 |
| 6341 | 0.0140 | 0.00 | 1817 |


### Combined Value Proposition
- At the **97.5th percentile** threshold (capturing the top 45 anomalous accounts), the Isolation Forest flags **0 out of 3** of the LightGBM false negatives.
- At the **99th percentile** threshold (capturing the top 19 anomalous accounts), the Isolation Forest flags **0 out of 3** of the LightGBM false negatives.

Due to the lack of overlap between the unsupervised outliers and supervised fraud patterns, the Isolation Forest did not capture any of the false negatives missed by LightGBM. This demonstrates that unsupervised models should not be deployed in a hybrid system using the exact same supervised-selected features, as they will target irrelevant transaction spikes rather than subtle, structured mule behaviors.

---

## 6. Summary Metrics

Below is the consolidated final analysis table:

| Metric | Value |
|:---|:---:|
| Avg anomaly score (normal) | 24.18 |
| Avg anomaly score (mule) | 17.74 |
| Top 1% anomalies limit (k) | 19 accounts |
| Mule accounts inside top 1% | 0 |
| Mule detection rate inside top 1% | 0.00% |
| Best CV Configuration | Option B (Cont=0.005) |
| Final Isolation Forest ROC-AUC | 0.3305 |
| Final Isolation Forest PR-AUC | 0.0084 |
| Recall at 95th percentile threshold | 6.25% (TP=1, FP=90) |
| Recall at 97.5th percentile threshold | 6.25% (TP=1, FP=45) |
| Recall at 99th percentile threshold | 0.00% (TP=0, FP=19) |
| LightGBM False Negatives Flagged by IF (>=97.5% pct) | 0 out of 3 |
| LightGBM False Negatives Flagged by IF (>=99.0% pct) | 0 out of 3 |


---
*End of Phase 4 Report*
