# Phase 6 Report: Explainability & Investigation Reports

This report presents the implementation and findings of **Phase 6: Explainability & Investigation Reports** for Bank of India's unified money mule detection system. Using SHAP (SHapley Additive exPlanations) values combined with statistical and behavioral anomaly profiles, we provide a complete, transparent, and actionable audit trail for every flagged account.

---

## 1. Executive Summary

Phase 6 introduces explainability to the unified risk engine by breaking down final fused risk scores into clear, quantifiable drivers from each of the three analytical pillars:
1. **Supervised Predictive Model (LightGBM)**: Explained via local SHAP contributions.
2. **Statistical Anomaly Detector (Isolation Forest)**: Assessed against baseline test-set behavior.
3. **Behavioral Anomaly Detector (Local Outlier Factor)**: Profiled via percentile rank and the application of the $+10.0$ behavioral boost.

### Key Outcomes
- **100% Precision in Critical & High Risk Bands**: All 10 accounts flagged in these bands (5 Critical, 5 High Risk) are actual money mules. 
- **Explainable Fraud Alerts**: Every alert in the **Monitor (22)**, **High Risk (5)**, and **Critical (5)** risk bands has an associated **Investigation Card** containing a 3-pillar risk breakdown and a natural language narrative explaining the root cause of the alert.
- **Systematic Audit of Misses**: Analyzed the 18 False Positives and the 2 False Negatives to identify specific model calibration improvements and feature engineering gaps.
- **Visual Artifacts**: Exported a global SHAP summary plot, 10 individual waterfall plots for the top-risk accounts, and 5 interactive HTML/JS force plots for dynamic exploration.

---

## 2. SHAP Methodology & Explainability Approach

To explain the supervised LightGBM model, we utilized **SHAP (SHapley Additive exPlanations)**, a game-theoretic approach that assigns each feature an importance value for a specific prediction. 

- **Explainer Selection**: We used `shap.TreeExplainer(model)` which is optimized for tree-based ensemble models (LightGBM/XGBoost) and computes mathematically exact Shapley values.
- **Target Value**: Predictions are explained in terms of their log-odds contributions (the raw output of the LightGBM decision trees prior to the sigmoid function).
- **Interpretation**: 
  - A **positive SHAP value** ($>0$) increases the log-odds of the prediction, pushing the account towards being flagged as a mule.
  - A **negative SHAP value** ($<0$) reduces the log-odds, pulling the account towards a normal classification.
- **Visualizations**: 
  - **Summary Beeswarm Plot**: Illustrates the global distribution of feature impacts across the entire test set.
  - **Waterfall Plots**: Provide a step-by-step additive trace from the base value (expected average model output) to the final prediction log-odds for individual accounts.
  - **Force Plots**: Render interactive, horizontal force vectors showing the push-and-pull dynamics of features on a single account.

---

## 3. Global Feature Importance (SHAP vs. Random Forest)

We compared the global importance of the top SHAP features against the Random Forest (RF) importance rankings computed in Phase 2. This comparison highlights how features behave differently in local predictions versus global splits.

The top 7 features driving predictions in the test set are analyzed below:

| Feature | Feature Type | RF Importance Rank | Model Role & Interpretation |
| :--- | :--- | :---: | :--- |
| **`F3801`** | Continuous | #1 | Global split driver; measures core transaction volume consistency. |
| **`F3807`** | Continuous | #2 | Secondary global continuous feature; relates to fund velocity. |
| **`F1922`** | Continuous | #5 | High continuous signal; measures value dispersion. |
| **`F3898`** | Continuous | #6 | **Primary Local Driver**: In both caught and missed mules, a low value (e.g., `-1.0` or `-0.67`) yields a massive positive SHAP contribution ($+1.20$ to $+3.06$), making it the strongest indicator of risk. |
| **`F1863`** | Continuous | #12 | Continuous feature; measures velocity deviation. |
| **`F3750`** | Continuous | #25 | Continuous feature; flags timing anomalies. |
| **`F3914`** | Binary | #118 | **Primary Local Risk Reducer**: Acts as a "normalizer" switch. When `F3914 = 1.0`, it exerts a strong risk-reducing SHAP contribution ($-0.56$ to $-0.89$), driving the ML probability to near-zero. When `F3914 = 0.0`, it yields a strong risk-increasing SHAP contribution ($+1.38$ to $+1.84$). |

---

## 4. Per-Risk-Band Feature Dynamics

The features that drive accounts into different risk bands differ significantly:

### Critical ($>80.00$) & High Risk ($60.01 - 80.00$) Bands
- **Driver**: Exclusively driven by the supervised ML model (`ml_score` $>95\%$, contributing $>65.0$ to the fused score).
- **Feature Profile**: Characterized by extreme values in the top continuous features:
  - `F3898` is highly negative (typically $-1.0$ or $-0.67$), indicating extreme variance.
  - `F3914` is strictly `0.0`, providing no risk reduction.
  - `F3908` (Binary feature, RF Rank #39) is strictly `1.0`, contributing a positive SHAP boost of $+0.70$ to $+1.14$.
  - Continuous features like `F3801`, `F3807`, `F1922`, and `F1863` are heavily skewed towards abnormal ranges.

### Monitor Band ($30.01 - 60.00$)
- **Driver**: Driven primarily by the behavioral anomaly score (Local Outlier Factor) and the $+10.0$ boost, rather than the supervised ML model.
- **Exception (Borderline ML Detections)**: Contains 4 actual money mules (`A9078`, `A9043`, `A9068`, `A9051`) where the ML model had moderate confidence (e.g., `A9043` has ML = $52.68$, `A9068` has ML = $48.19$). These represent borderline profiles that are correctly captured under enhanced monitoring.

---

## 5. False Positive Investigation

A total of **18 false positives** (actual normal accounts classified in the Monitor band) were identified.

### Root Cause Analysis
Every single one of the 18 false positives shares the same risk score signature:
- **`ml_score` = `0.00`** (Supervised LightGBM correctly identified these as normal accounts).
- **`stat_score` = Low to Moderate** (Isolation Forest score was typically below the test-set average of $24.12$).
- **`behavior_score` $\ge$ `99.00`** (Local Outlier Factor identified these accounts as being in the top 1% of behavioral outliers on the ledger).
- **Boost Applied**: A $+10.0$ behavioral boost was applied, pushing the final score from $\sim 21.50$ to $\sim 31.50$, crossing the $30.00$ Monitor threshold.

Because the fused score formula places a $20\%$ weight on behavioral anomalies and adds a $+10.0$ boost to the top $1\%$, any account in the top $1\%$ of LOF outliers is guaranteed to score at least $30.00$ ($20 \times 1.00 + 10.0 = 30.00$), placing it in the **Monitor** band even if the ML model score is $0.00$.

### Steps for Investigation of False Positives
For investigators handling these accounts:
1. **Verify Ledger Outliers**: Look for sudden, sharp increases in transaction frequency, large pass-through fund flows, or changes in customer occupation/location. Since these are in the top 1% of outliers, they are doing something highly atypical.
2. **Review High-Risk Activities**: Check if the anomaly is driven by legitimate business activities (e.g., a student receiving tuition funds, a retail merchant's holiday sales spike).
3. **Log Context**: If the activity is benign, tag the account profile type (e.g., "high-volume merchant") to prevent future false alerts.

### Recommendations for Model Calibration
To reduce the volume of behavioral false alerts without missing actual mules:
1. **Conditional Boosting**: Apply the $+10.0$ boost only if the account has a minimum statistical anomaly score (e.g., `stat_score` $\ge 20.0$) or a non-zero ML probability (`ml_score` $>1.0$).
2. **Refine LOF Features**: Add context-aware scaling to the behavioral features (e.g., scaling transaction velocity by the average historical volume for that specific customer segment) to filter out benign high-value transactions.

---

## 6. False Negative Investigation

Only **2 actual money mule accounts** were missed by the unified risk engine, remaining in the **Normal** risk band.

### Account `A9047` Analysis (Risk Score: `13.37`)
- **Component Scores**: ML = `0.01` | Stat = `22.51` | Behavior = `55.59`
- **Supervised Defeat**: The ML model assigned a probability of just $0.01\%$.
- **SHAP Breakdown**:
  - *Risk-Increasing*: `F3898 = -1.0` (SHAP: $+0.75$), `F1863 = -0.29` (SHAP: $+0.37$).
  - *Risk-Reducing*: `F3914 = 1.0` (SHAP: $-0.56$), `F3750 = -0.51` (SHAP: $-0.21$), `F996 = 0.0` (SHAP: $-0.20$).
- **Root Cause**: The presence of `F3914 = 1.0` (which is typically $0.0$ for caught money mules) acted as a powerful risk-reducer, completely overriding the risk-increasing signals from `F3898`. Additionally, the behavioral LOF score was in the 55th percentile, which did not trigger an alert.

### Account `A9076` Analysis (Risk Score: `2.71`)
- **Component Scores**: ML = `1.40` | Stat = `0.00` | Behavior = `8.64`
- **Supervised Defeat**: The ML model assigned a probability of $1.40\%$.
- **SHAP Breakdown**:
  - *Risk-Increasing*: `F3898 = -1.0` (SHAP: $+1.21$), `F3908 = 1.0` (SHAP: $+0.70$).
  - *Risk-Reducing*: `F3914 = 1.0` (SHAP: $-0.89$), `F1863 = 0.31` (SHAP: $-0.31$), `F3913 = 1.0` (SHAP: $-0.21$).
- **Root Cause**: Similar to `A9047`, the normalizer switch `F3914 = 1.0` drove a huge negative SHAP contribution ($-0.89$), canceling out the positive drivers. Furthermore, this account exhibited zero statistical anomaly (`stat_score = 0.0`) and very low behavioral anomaly (`behavior_score = 8.64`), indicating its ledger activity was completely indistinguishable from normal accounts.

### Steps for Investigation of False Negatives
1. **Trigger Manual Audits**: Perform a deep forensic ledger audit on `A9047` and `A9076` to identify if they represent "sleepy" accounts (low initial activity designed to bypass detection) or sophisticated mules with slow, steady transaction rates.
2. **Review Out-of-Sample Features**: Examine transaction descriptions, device logins, or IP addresses (features not in our tabular dataset) for anomalous patterns.

### Recommendations for Model Improvement
1. **Interactive Feature Engineering**: Create interaction terms between `F3914` and key risk features (e.g., `F3898 * F3914`) to prevent the model from over-relying on `F3914` as a blanket normalizer.
2. **Separate Model Segment**: Train a dedicated sub-model for accounts where `F3914 = 1.0` to detect the subtle differences between normal accounts and the subset of mules that display this feature.

---

## 7. Sample Investigation Cards

Below are three sample investigation cards representing actual money mules in the Critical band, showing how raw feature code signals are translated into factual, verified details:

### Card 1: Account `A9044` (Critical — Fused Score: `83.49`)

> [!NOTE]
> **Recommended Action**: Immediate review & debit freeze.
>
> **3-Pillar Score Breakdown**:
> - ML Score: `99.96` (Weighted: `69.97`)
> - Statistical Score: `30.77` (Weighted: `3.08`) — Above test-set average.
> - Behavioral Score: `52.23` (Weighted: `10.45`) — Moderate anomaly.
>
> **Top SHAP Contributors (LightGBM)**:
> 1. `F3898` (Continuous feature, 0.0% missing; RF rank #6): Value = `-0.67` | SHAP = `+2.46` (Risk-Increasing)
> 2. `F3914` (Binary feature, 0.0% missing; RF rank #118): Value = `0.00` | SHAP = `+1.39` (Risk-Increasing)
> 3. `F3801` (Continuous feature, 0.0% missing; RF rank #1): Value = `-0.30` | SHAP = `+0.80` (Risk-Increasing)
> 4. `F1922` (Continuous feature, 0.14% missing; RF rank #5): Value = `-0.19` | SHAP = `+0.79` (Risk-Increasing)
> 5. `F1863` (Continuous feature, 0.04% missing; RF rank #12): Value = `-0.26` | SHAP = `+0.77` (Risk-Increasing)
>
> **Narrative**:
> ML Component (weight: 70%): ML score = 99.96. Weighted contribution = 69.97. High-risk profile.
> Statistical Anomaly (weight: 10%): Score = 30.77. Weighted contribution = 3.08. Above test-set average (24.12).
> Behavioral Anomaly (weight: 20%): LOF percentile = 52.23. Weighted contribution = 10.45.
> Assessment: Alert primarily driven by ML and Behavioral Anomaly.

---

### Card 2: Account `A9075` (Critical — Fused Score: `81.20`)

> [!NOTE]
> **Recommended Action**: Immediate review & debit freeze.
>
> **3-Pillar Score Breakdown**:
> - ML Score: `100.00` (Weighted: `70.00`)
> - Statistical Score: `27.29` (Weighted: `2.73`) — Above test-set average.
> - Behavioral Score: `42.38` (Weighted: `8.48`) — Moderate anomaly.
>
> **Top SHAP Contributors (LightGBM)**:
> 1. `F3898` (Continuous feature, 0.0% missing; RF rank #6): Value = `-1.00` | SHAP = `+3.06` (Risk-Increasing)
> 2. `F3914` (Binary feature, 0.0% missing; RF rank #118): Value = `0.00` | SHAP = `+1.60` (Risk-Increasing)
> 3. `F3908` (Binary feature, 0.0% missing; RF rank #39): Value = `1.00` | SHAP = `+1.14` (Risk-Increasing)
> 4. `F3750` (Continuous feature, 0.02% missing; RF rank #25): Value = `0.79` | SHAP = `+1.01` (Risk-Increasing)
> 5. `F3807` (Continuous feature, 0.0% missing; RF rank #2): Value = `-0.23` | SHAP = `+0.72` (Risk-Increasing)
>
> **Narrative**:
> ML Component (weight: 70%): ML score = 100.00. Weighted contribution = 70.00. Definite historic mule profile.
> Statistical Anomaly (weight: 10%): Score = 27.29. Weighted contribution = 2.73. Above test-set average (24.12).
> Behavioral Anomaly (weight: 20%): LOF percentile = 42.38. Weighted contribution = 8.48.
> Assessment: Alert primarily driven by ML.

---

### Card 3: Account `A9080` (Critical — Fused Score: `80.41`)

> [!NOTE]
> **Recommended Action**: Immediate review & debit freeze.
>
> **3-Pillar Score Breakdown**:
> - ML Score: `95.72` (Weighted: `67.00`)
> - Statistical Score: `12.07` (Weighted: `1.21`) — Below test-set average.
> - Behavioral Score: `60.98` (Weighted: `12.20`) — Elevated anomaly.
>
> **Top SHAP Contributors (LightGBM)**:
> 1. `F3898` (Continuous feature, 0.0% missing; RF rank #6): Value = `-0.67` | SHAP = `+1.90` (Risk-Increasing)
> 2. `F3914` (Binary feature, 0.0% missing; RF rank #118): Value = `0.00` | SHAP = `+1.84` (Risk-Increasing)
> 3. `F3908` (Binary feature, 0.0% missing; RF rank #39): Value = `1.00` | SHAP = `+0.77` (Risk-Increasing)
> 4. `F3750` (Continuous feature, 0.02% missing; RF rank #25): Value = `0.01` | SHAP = `+0.60` (Risk-Increasing)
> 5. `F435` (Continuous feature, 6.99% missing; RF rank #168): Value = `0.88` | SHAP = `+0.54` (Risk-Increasing)
>
> **Narrative**:
> ML Component (weight: 70%): ML score = 95.72. Weighted contribution = 67.00.
> Statistical Anomaly (weight: 10%): Score = 12.07. Weighted contribution = 1.21. Below test-set average (24.12).
> Behavioral Anomaly (weight: 20%): LOF percentile = 60.98. Weighted contribution = 12.20.
> Assessment: Alert primarily driven by ML and Behavioral Anomaly.

---

## 8. Explainability Visualizations

The explainability script generated multiple key plots to help understand model behavior:

### A. Global Feature Impact
The **SHAP Summary Plot** (`shap_summary_plot.png`) shows the top 20 features sorted by their average absolute SHAP values. 
- Features like `F3898` and `F3914` appear at the very top, confirming their massive impact on model predictions across the test set.
- Continuous features exhibit distinct color gradients representing feature values (red for high values, blue for low values), illustrating how value ranges correspond directly to risk increases or decreases.

![Global SHAP Summary Plot](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/shap_summary_plot.png)

### B. Individual Risk Drivers
For each High Risk and Critical account, we generated **SHAP Waterfall Plots** (stored as `waterfall_{account_id}.png`). These plots map out exactly how the features pushed the log-odds of a specific account from the base expected value of the training set up to the high-probability prediction.

For example, on account **`A9044`**:
- The baseline log-odds starts near $-6.0$ (representing the very low prior probability of mule accounts in the dataset).
- Feature `F3898` (value = $-0.67$) adds $+2.46$ to the log-odds.
- Feature `F3914` (value = $0.00$) adds $+1.39$ to the log-odds.
- The final log-odds is pushed to $+5.83$, resulting in a prediction probability of $99.96\%$.

![SHAP Waterfall Plot for A9044](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/waterfall_A9044.png)
![SHAP Waterfall Plot for A9075](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/waterfall_A9075.png)

### C. Interactive Exploration
Interactive **Force Plots** were generated as HTML files (`force_{account_id}.html`) for all 5 Critical accounts. These files use JavaScript to enable interactive exploration of the push-and-pull factors. Investigators can hover over features to see detailed values and exact mathematical contributions, offering an excellent presentation layer for the Bank of India fraud team.
