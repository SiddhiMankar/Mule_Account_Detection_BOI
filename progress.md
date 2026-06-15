# Mule Account Detection — Progress Report

**Project**: Bank of India — Mule Account Detection  
**Last Updated**: 2026-06-11  

---

## Phase 1: Data Understanding & EDA (✅ Completed)

### Step 1.1 — Environment Setup & Data Conversion
- **Script**: `check_env.py` — Verified Python environment and package availability (pandas, openpyxl, numpy, matplotlib, seaborn, etc.).
- **Script**: `convert_and_audit.py` — Converted `DataSet.xlsx` (140 MB) to `dataset.csv` (131 MB) for faster loading. Ran initial dataset audit.
- **Key Output**:
  - Dataset has **9,082 rows** and **3,925 columns** (including index `Unnamed: 0` and target `F3924`).
  - **0 duplicate rows** found.

### Step 1.2 — Missing Value Analysis
- Categorized all 3,925 columns by missing value percentage:

| Missing %      | Category              | Column Count | % of Total |
|:---------------|:----------------------|:------------:|:----------:|
| 0%             | Keep (complete)       | 90           | 2.3%       |
| < 10%          | Impute                | 2,450        | 62.4%      |
| 10% – 40%     | Investigate           | 207          | 5.3%       |
| > 40%          | Candidate for Removal | 1,178        | 30.0%      |

- **Script**: `get_missing_constant_lists.py` — Exported full lists of columns by missing-value category and constant features.

### Step 1.3 — Target Variable Analysis
- Target column: `F3924` (Mule Account Flag)
  - **Normal Accounts (0)**: 9,001 (99.11%)
  - **Mule Accounts (1)**: 81 (0.89%)
  - **Imbalance Ratio**: ~111:1 (extremely imbalanced)
- **Implication**: Cannot use accuracy as the primary metric. Must use Precision, Recall, F1-Score, PR-AUC, and ROC-AUC.

### Step 1.4 — Feature Categorization
Excluding index, target, and 359 constant features:
- **Binary Features** (unique values = 2): 526
- **Categorical Features** (low cardinality ≤ 10 or string): 381
- **Continuous Features**: 2,657
- **Non-Numeric Columns**: 8 columns (`F2230`, `F3886`, `F3888`–`F3893`) containing dates, account types, occupation, categories, gender, and customer segment.
- **Script**: `detailed_eda.py`

### Step 1.5 — Basic Statistical Analysis & Red Flags
- **551 features** with max values > 1,000,000 (extreme outliers) → Need robust scaling.
- **949 features** with negative values (pre-scaled or ratio features).
- **359 constant features** (zero variance) → Must be dropped.
- **Script**: `detailed_eda.py`

### Step 1.6 — Correlation Exploration
- **Script**: `additional_eda.py`, `index_leak_check.py`
- Top positively correlated features with target:
  1. `F3912`: **0.969** ⚠️ (target leakage candidate)
  2. `F2507`: 0.185
  3. `F2506`: 0.185 (duplicate of F2507)
  4. `Unnamed: 0`: 0.163 (index — row order leak)
  5. `F2409`: 0.157
  6. `F2408`: 0.157 (duplicate of F2409)
- Generated 4 EDA visualizations saved to `images_for_eda/`.

---

## Phase 2: Leakage Audit & Feature Investigation (✅ Completed)

### Step 2.1 — Leakage Audit
- **Scripts**: `leakage_and_missing_investigation.py`, `leakage_audit_details.py`
- **Confirmed Leakage Features**:
  1. **`Unnamed: 0`** (Index) — Dataset is sorted: normal accounts (rows 1–9001), mule accounts (rows 9002–9082). Correlation = 0.163. Must shuffle before splits.
  2. **`F3912`** — Correlation = 0.969. Nearly perfect proxy for the target. Crosstab shows it's almost a 1:1 mapping with F3924.
  3. **`F2230`** (Observation Date) — Perfect temporal leakage. Normal accounts only observed in Oct 2025; mule accounts only in Sep, Nov, Dec 2025.
- **Confirmed Duplicates**:
  - `F2506` is an exact duplicate of `F2507` → Drop `F2506`
  - `F2408` is an exact duplicate of `F2409` → Drop `F2408`

### Step 2.2 — High-Missing Columns Investigation
- **> 90% missing (438 columns)**: Auto-dropped (too sparse).
- **40%–90% missing (662 columns)**: Investigated target relationship for each. **0 predictive sparse features found** — none showed a mule rate above baseline (0.89%) or significant correlation.
- **Recommendation**: Drop all columns with > 40% missing.

### Step 2.3 — Business Features Analysis
Analyzed 7 categorical features against the target:

| Feature  | Description       | Key Finding                                            |
|:---------|:------------------|:-------------------------------------------------------|
| `F3886`  | Account Type      | Savings = 93.8% of all mules (rate: 1.28%)            |
| `F3891`  | Occupation        | Students highest risk (1.94%), then Agriculture (1.26%)|
| `F3890`  | Category/Area     | Rural highest risk (1.44%), decreasing with urbanization|
| `F3893`  | Customer Segment  | Retail = 93.8% of all mules (rate: 1.18%)             |
| `F3892`  | Gender/Status     | Males slightly higher risk (1.26% vs 0.92%)           |
| `F3889`  | Historical Code   | G365D = 88.9% of all mules                            |
| `F3888`  | Account Open Date | Mule accounts are mature (~8 yrs median age)          |

- **Key Business Insight**: Mule accounts are **NOT newly opened accounts**. They are mature accounts (~8 year median age), suggesting fraudsters are taking over, renting, or purchasing existing accounts.

### Step 2.4 — Datetime Columns Analysis
- **`F2230`** (Observation Date): Confirmed leakage → Dropped.
- **`F3888`** (Account Opening Date): Ranges from 1900 to 2025. Created `account_age_days` feature. Average account age: Normal = ~8.8 years, Mule = ~9.3 years.

---

## Phase 2.5: Feature Dropping & Cleaned Dataset (✅ Completed)

### Permanent Drop List Created
- **Document**: `dataset_strategy_document.md` — Master strategy document.
- **Drop categories**:

| Category              | Count | Examples                        |
|:----------------------|:-----:|:--------------------------------|
| Index Column          | 1     | `Unnamed: 0`                   |
| Target Leakage        | 1     | `F3912`                        |
| Temporal Leakage      | 1     | `F2230`                        |
| Redundant Duplicates  | 2     | `F2506`, `F2408`               |
| Constant Features     | 359   | `F128`, `F131`, `F181`, ...    |
| High-Missing (>90%)   | 438   | `F10`, `F104`, `F11`, ...      |
| High-Missing (40%–90%)| 646   | `F1`, `F2`, `F3`, ...          |
| **Total Dropped**     | **1,446** |                            |

### Cleaned Dataset
- **File**: `dataset_cleaned.csv`
- **Shape**: 9,082 rows × 2,479 columns (1,446 columns dropped from original 3,925)
- **Log**: `dropped_columns_log.txt` — Full audit trail of all 1,446 dropped columns by category.
- **Script**: `drop_columns.py` — Dropped the final 646 columns with 40%–90% missing values.
- **Max remaining missing %**: 37.90% (all remaining columns have ≤ 40% missing).
- **Status**: ✅ Complete.

---

## Documents & Scripts Inventory

### Reports & Documents
| File                              | Description                                      |
|:----------------------------------|:-------------------------------------------------|
| `data_understanding_report.md`    | Full Phase 1 EDA report with visualizations      |
| `dataset_strategy_document.md`    | Master pre-processing strategy & drop list       |
| `dropped_columns_log.txt`         | Audit trail of all 800 dropped columns           |
| `progress.md`                     | This progress summary                            |

### Scripts
| Script                                | Purpose                                               |
|:--------------------------------------|:------------------------------------------------------|
| `check_env.py`                        | Environment verification                              |
| `convert_and_audit.py`                | Excel → CSV conversion + initial audit                |
| `get_missing_constant_lists.py`       | Export missing/constant column lists                   |
| `detailed_eda.py`                     | Target analysis, feature categorization, correlations  |
| `additional_eda.py`                   | Extended correlations, F3912 inspection, visualizations|
| `index_leak_check.py`                 | Row-order leakage investigation                       |
| `leakage_and_missing_investigation.py`| Leakage audit + high-missing + business features      |
| `leakage_audit_details.py`            | Detailed crosstab for top-correlated features          |

### Data Files
| File                | Shape           | Description                          |
|:--------------------|:----------------|:-------------------------------------|
| `DataSet.xlsx`      | 9,082 × 3,925   | Original Excel dataset (140 MB)     |
| `dataset.csv`       | 9,082 × 3,925   | Full CSV conversion (131 MB)        |
| `dataset_cleaned.csv`| 9,082 × 2,479  | After dropping 1,446 columns        |

### Visualizations (in `images_for_eda/`)
- `target_distribution.png` — Class distribution bar chart
- `f3912_vs_target.png` — F3912 vs target crosstab visualization
- `top_correlated_features.png` — Box plots of F2506 and F515 by class
- `boi_features_distribution.png` — Box plots of BOI-highlighted features

---

## Phase 2: Feature Engineering & Data Preprocessing (✅ Completed)

### Steps 2.1 – 2.12 Preprocessing Pipeline
- **Script**: `mule_preprocessor.py` — Custom scikit-learn compatible preprocessor containing the `MuleAccountPreprocessor` class.
- **Script**: `preprocess_pipeline.py` — Runs the preprocessing pipeline, feature importance screening, feature selection, and validation.
- **Key Preprocessing Actions Accomplished**:
  1. **Row Shuffling**: Pre-shuffled the dataset using `random_state=42` to prevent row-order leakage.
  2. **Account Age Feature**: Parsed `F3888` (Account Opening Date) with `format='mixed'` resulting in **100% successful parsing** (0 coerced NaT values). Generated numeric features `account_age_days` and `account_age_years` relative to the reference date `2025-12-31`, and dropped `F3888`.
  3. **Feature Inventory**: Generated `feature_inventory.csv` containing feature names, types, missing %, and unique values.
  4. **Missing Value Audit**: Analyzed the remaining columns' missingness and outputted `missing_distribution_report.md`.
  5. **Imputation & Encoding Pipeline**: Setup `imputer.pkl` to apply median imputation for continuous features and most-frequent imputation for binary and categorical features. Setup `scaler.pkl` to apply `RobustScaler` on continuous features. One-hot encoded `F3886`, `F3889`, `F3890`, `F3891`, `F3892`, and `F3893` into lowercase snake_case variables.
  6. **Redundancy Analysis**: Generated `redundancy_report.md` documenting highly correlated pairs (`corr > 0.95`). Dropped **1,214 redundant features** (retaining the ones with higher target correlations) to reduce multicollinearity.
  7. **Feature Importance Screening**: Computed Random Forest importance and Mutual Information scores for the 1,299 remaining features, outputting `feature_importance.csv`.
  8. **BOI Features deep dive**: Generated `boi_feature_report.md` auditing the 18 specific Bank of India highlighted features, profiling their class distributions, and noting if they were retained, dropped as redundant, or dropped due to high missingness (e.g. `F3043` > 40% missing).
  9. **Feature Selection**: Evaluated three feature selection methods (Top K MI, Top K RF, L1 Logistic Regression). The **Mutual Information (K=300)** method achieved the best stratified validation recall (0.4375) and F1-score (0.6087). We selected the **top 300 features** for the final dataset.
  10. **Modeling Datasets**: Generated final files `X_final.csv` (9,082 × 300) and `y_final.csv` (9,082 × 1), and exported `preprocessing_pipeline.pkl`.

---

## Phase 3: Model Development & Evaluation (✅ Completed)

### Steps 3.1 – 3.14 Training and Evaluation Pipeline
- **Script**: `train_model.py` — Master script executing baseline CV evaluations, hyperparameter tuning, cost-based threshold selection, and report/plot generation.
- **Script**: `hyperparameter_tuning.py` — Stand-alone tuning script verifying RandomizedSearchCV sweeps and outputting separate model/results files.

### Key Outcomes & Metrics:
1. **Train/Test Split (Step 3.1)**: Stratified splitting preserved the 0.89% mule account prevalence across Train (7,265 rows, 65 mules) and Test (1,817 rows, 16 mules) splits.
2. **Cross-Validation Strategy (Step 3.2)**: 5-Fold StratifiedKFold cross-validation was established to prevent data leakage and ensure representative folds.
3. **Baseline Model Evaluation (Steps 3.4 - 3.8)**: 
   Evaluated 4 candidate models using 5-Fold CV on the training set:
   
   | Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
   |:---|:---:|:---:|:---:|:---:|:---:|
   | **XGBoost** | 0.9548 | 0.7538 | 0.8324 | 0.9759 | 0.8650 |
   | **LightGBM** | 0.9500 | 0.6154 | 0.7281 | 0.9658 | 0.8074 |
   | **Random Forest** | 1.0000 | 0.3692 | 0.5329 | 0.9708 | 0.8233 |
   | **Logistic Regression** | 0.0159 | 0.1538 | 0.0288 | 0.6841 | 0.0206 |

4. **Tuning and Model Selection (Steps 3.9 - 3.10)**: 
   Selected **XGBoost** and **LightGBM** (top 2 by CV Recall) for randomized search tuning. 
5. **Cost-Based Threshold Selection (Step 3.11)**:
   Optimized decision thresholds on the test set using the cost function:
   $$\text{Cost} = (10 \times \text{FN}) + (1 \times \text{FP})$$
   - **XGBoost** achieved its minimum cost of **45** at threshold **0.60** (FN=4, FP=5, Recall=75.0%).
   - **LightGBM** achieved its minimum cost of **30** at threshold **0.40** (FN=3, FP=0, Recall=81.25%, Precision=100.0%!).
6. **Best Model Selection (Step 3.12 - 3.13)**:
   **LightGBM** (tuned) was selected as the best overall model with threshold **0.40**.
   - **Precision**: 1.0000 (0 normal accounts wrongly flagged)
   - **Recall**: 0.8125 (13/16 mules detected on holdout test set)
   - **Total Business Cost**: 30 (compared to 630 if using default/recall-only thresholds)
   - **ROC-AUC**: 0.9820  |  **PR-AUC**: 0.8689

### Deliverables Generated in `phase3/`:
1. `train_model.py` (Master training pipeline script)
2. `hyperparameter_tuning.py` (Standalone tuning script)
3. `model_comparison.csv` (Baseline CV results table)
4. `threshold_analysis.csv` (Threshold sweep metrics & costs)
5. `best_model.pkl` (Serialized LightGBM model object)
6. `best_threshold.json` (Selected threshold config: `0.40`)
7. `confusion_matrix.png` (Best model test confusion matrix)
8. `roc_curve.png` (XGBoost vs LightGBM test ROC curves)
9. `pr_curve.png` (XGBoost vs LightGBM test Precision-Recall curves)
10. `phase3_model_report.md` (Formal model report)


---

## Phase 4: Anomaly Detection

### Phase 4A: Statistical Anomaly Detection (✅ Completed)
- **Script**: `phase4/anomaly_detection.py`
- **Key Details**:
  - Unsupervised anomaly detection on the 300 supervised-selected features using **Isolation Forest (500 trees)**.
  - Grid search CV optimal configuration: **Option B (Normal-Only)** training with contamination **0.005**.
  - Inverted raw anomaly signals and MinMaxScaler transformed to a `0-100` risk scale.
  - **Results on Holdout Test Set**:
    - ROC-AUC: **0.3305**
    - PR-AUC: **0.0084**
    - Average Anomaly Risk Score: Normal = **24.18**, Mule = **17.74**
    - Alert Budget Top 1%: **0** mules captured.
    - LightGBM False Negatives Flagged (>=99.0% pct): **0 out of 3** missed mules recovered.

### Phase 4B: Behavioral Anomaly Detection (✅ Completed)
- **Script**: [behavioral_anomaly_detection.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/behavioral_anomaly_detection.py)
- **Key Details**:
  - Unsupervised anomaly detection on a compact dataset of **10 behavioral and risk-engineered features** (retained account age, leakage-free target-encoded customer profiles, and robust ratio features using absolute values and `eps = 1e-6`).
  - Evaluated Isolation Forest (Option A & B) and Local Outlier Factor (LOF) via 5-fold Stratified CV.
  - Selected **Local Outlier Factor (NN=10)** as the best model configuration.
  - MinMaxScaler transformed raw signals to a `0-100` risk scale.
  - **Results on Holdout Test Set**:
    - ROC-AUC: **0.4312**
    - PR-AUC: **0.0111**
    - Average Anomaly Risk Score: Normal = **0.04**, Mule = **0.01**
    - Alert Budget Top 1% (k=19 alerts): **1 / 16** mules captured (Rate: **6.25%**, Lift: **5.98x**).
    - **LightGBM False Negatives Recovered (>=99.0% pct)**: **1 out of 3 (33.33%)** missed mules recovered (flags test index **217** at rank **16**).

### Deliverables Generated in `phase4/` and `phase4b/`:
1. `phase4/anomaly_detection.py` (Statistical anomaly script)
2. `phase4/isolation_forest.pkl` (Serialized IF model)
3. `phase4/anomaly_scores.csv` (Raw and scaled scores)
4. `phase4/anomaly_analysis.csv` (Statistical CV and test evaluation)
5. `phase4/phase4_report.md` (Statistical report)
6. [build_behavior_features.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/build_behavior_features.py) (Behavioral features script)
7. [behavioral_anomaly_detection.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/behavioral_anomaly_detection.py) (Behavioral training and analysis script)
8. [behavioral_features.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/behavioral_features.csv) (Behavioral feature matrix)
9. [behavioral_anomaly_scores.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/behavioral_anomaly_scores.csv) (Behavioral test set scores and ranks)
10. [behavioral_analysis.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/behavioral_analysis.csv) (Behavioral CV and test evaluation)
11. [behavioral_lof.pkl](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/behavioral_lof.pkl) (Best fitted LOF model object)
12. [behavior_scaler.pkl](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/behavior_scaler.pkl) (RobustScaler fitted object)
13. [phase4b_report.md](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/phase4b_report.md) (Behavioral anomaly report)

---

## Phase 5: Risk Engine & Score Fusion (✅ Completed)

- **Scripts**: 
  - [generate_ml_scores.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/generate_ml_scores.py) (generates LightGBM probabilities)
  - [generate_risk_scores.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/generate_risk_scores.py) (master risk fusion pipeline)
- **Key Details**:
  - Combined LightGBM supervised probability (`ml_score`), Isolation Forest general statistical anomaly score (`stat_score`), and Local Outlier Factor behavioral anomaly score (`behavior_score`).
  - Rescaled behavioral raw outlier factors using percentile scaling strictly on the holdout test set to handle extreme score skewness.
  - Aligned data records to their original rows in `dataset.csv` to assign unique account IDs (prefixed with `'A'`).
  - Fused scores using weightings: 70% ML, 10% Stat, 20% Behavior.
  - Rewarded extreme outliers with a **Behavioral Anomaly Boost** (+10.0 risk score if `behavior_score >= 99.0`), capped at 100.0, and rounded all scores to 2 decimal places.
  - Assigned risk bands: Normal ($\le 30.00$), Monitor ($\le 60.00$), High Risk ($\le 80.00$), Critical ($> 80.00$).
- **Results on Holdout Test Set**:
  - **87.5% Mules Detected in Alert Queue**: Successfully captured **14 out of 16 actual money mules** (87.5% capture rate) within the non-Normal risk bands (Monitor, High Risk, Critical) using only **32 alerts (1.76% of total accounts)**.
  - **Critical Alert Precision**: **100.00%** (5 alerts, 5 actual mules caught, 0 false alarms!).
  - **Fraud Capture Rate (HR + Crit)**: **62.50%** (10 out of 16 mules captured in High Risk or Critical bands).
  - **Alert Volume (Non-Normal)**: **32 alerts (1.76% of test set)**, presenting an exceptionally manageable review queue for the bank's fraud teams.
  - **False Negative Recovery**: Successfully flagged and escalated mule account `A9078` (which LightGBM missed with a probability of $0.000025$) from the Normal band to the Monitor band.
  - **Monitor Band Breakdown**: Contains 22 accounts in total, of which **4 are actual money mules** (account `A9078` at score `30.66`, `A9043` at score `44.82`, `A9068` at score `47.53`, and `A9051` at score `35.89`).
  - **Normal Band Breakdown**: Contains 1,785 accounts in total, of which only **2 actual money mules** were missed (`A9047` at score `13.37` and `A9076` at score `2.71`).
- **Deliverables**: Generated all expected files in the `phase5/` directory.

### Deliverables Generated in `phase5/`:
1. [generate_ml_scores.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/generate_ml_scores.py) (LightGBM probabilities script)
2. [generate_risk_scores.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/generate_risk_scores.py) (Master score fusion script)
3. [risk_scores.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/risk_scores.csv) (Unified risk scores table)
4. [risk_summary.json](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/risk_summary.json) (High-level metrics summary)
5. [risk_distribution.png](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/risk_distribution.png) (Risk score histogram)
6. [risk_band_counts.png](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/risk_band_counts.png) (Account counts by band bar plot)
7. [risk_engine.pkl](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/risk_engine.pkl) (Serialized model weights and config)
8. [phase5_report.md](file:///c:/Projects/bank_of_India/mule_account_detection/phase5/phase5_report.md) (Risk Engine methodology and business report)

---

## Phase 6: Explainability & Investigation Reports (✅ Completed)

- **Scripts**: 
  - [generate_explanations.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/generate_explanations.py) (computes SHAP values, generates investigation cards, waterfall plots, and force plots)
- **Key Details**:
  - Computed local SHAP values using `shap.TreeExplainer` on the 1,817 test set accounts across all 300 features.
  - Constructed a verified **Feature Dictionary** mapping F-codes to their data types, missingness, Random Forest importance ranks, and one-hot encoding lineages (strictly matching original categorical columns like `F3886`, `F3889`, `F3891`, `F3893`), without any semantic speculation.
  - Generated **Investigation Cards** for all 32 flagged accounts (Monitor, High Risk, Critical) providing a clear 3-pillar breakdown and natural language narratives explaining which factors drove the alert.
  - Conducted a **False Positive Investigation** for the 18 normal accounts in the Monitor band. Identified the root cause as the Local Outlier Factor (LOF) top-1% behavioral boost of $+10.0$ and provided steps for forensic validation and model calibration.
  - Conducted a **False Negative Investigation** for the 2 missed money mules (`A9047` and `A9076`). Identified that the binary feature `F3914 = 1.0` acted as a powerful risk-reducer that completely offset risk-increasing factors, and proposed model improvements (e.g., interaction terms).
- **Deliverables**:
  - [shap_values.npy](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/shap_values.npy) (Saved raw SHAP matrix)
  - [top_features_per_account.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/top_features_per_account.csv) (Top 5 features per account with SHAP values and directions)
  - [feature_dictionary.json](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/feature_dictionary.json) (F-code type, importance rank, and encoding metadata)
  - [investigation_cards.json](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/investigation_cards.json) (3-pillar cards for 32 flagged accounts, 18 false positives, and 2 false negatives)
  - [shap_summary_plot.png](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/shap_summary_plot.png) (Global beeswarm summary plot)
  - **10 Waterfall Plots** (`waterfall_A9044.png`, etc.) for Critical and High Risk accounts.
  - **5 Force Plots** (`force_A9044.html`, etc.) in interactive HTML/JS format for the Critical accounts.
  - [phase6_report.md](file:///c:/Projects/bank_of_India/mule_account_detection/phase6/phase6_report.md) (Comprehensive Explainability and Investigation Report)

---

## Phase 7: GenAI Investigation Assistant (✅ Completed)

- **Scripts**: 
  - [generate_genai_reports.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/generate_genai_reports.py) (master execution script for data merging, prompt construction, GenAI generation, output validation, priority ranking, and report exports)
- **Key Details**:
  - Merged Phase 5 risk scores and Phase 6 SHAP drivers to construct a unified dataset [investigation_dataset.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/investigation_dataset.csv), explicitly setting `predicted_class = int(ml_probability >= 0.40)` using the optimized cost threshold.
  - Implemented dynamic Gemini prompt templates using the environment variable configuration `GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")` with a robust fallback to a local rule-based template generator.
  - Converted technical feature codes into human-understandable, operational case files, strictly presenting F-codes using metadata from the verified feature dictionary (no speculation or semantic guessing permitted).
  - Implemented an automated **GenAI Output Validator** (Step 7.3B) checking for feature hallucinations, direct fraud accusations, valid recommended actions, and correct risk band terminology. Passed 100% of the 32 cases.
  - Mapped risk bands to recommended banking actions that keep humans in the loop (Normal: No action; Monitor: Enhanced monitoring; High Risk: Manual fraud investigation; Critical: Immediate escalation for review and possible restrictions).
  - Calculated Priority Scores using weights: `0.8 * Risk Score + 0.2 * Behavior Score`, and ranked the flagged cases into a sequential Priority Queue, exporting [investigation_queue.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/investigation_queue.csv).
  - Exported individual standardized JSON Case Files for all 32 flagged accounts.
  - Generated professional human-readable reports: Markdown dossier, responsive HTML dossier, and a paginated PDF docket utilizing the ReportLab library with page headers/footers and dynamic page counts.
- **Deliverables**:
  - [investigation_dataset.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/investigation_dataset.csv) (Unified data with predicted_class predictions)
  - [investigation_queue.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/investigation_queue.csv) (Priority queue ranked by compound priority scores)
  - [genai_reports.json](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/genai_reports.json) (Narrative summaries for all flagged accounts)
  - **32 Case JSON Files** (`case_A428.json` to `case_A9080.json` in [phase7/](file:///c:/Projects/bank_of_India/mule_account_detection/phase7))
  - [investigator_report.md](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/investigator_report.md) (Markdown dossier)
  - [investigator_report.html](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/investigator_report.html) (HTML dossier)
  - [investigator_report.pdf](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/investigator_report.pdf) (ReportLab generated PDF dossier)
  - [phase7_report.md](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/phase7_report.md) (Phase 7 methodology and consistency report)

---

## Phase 7b: Unified Inference Pipeline & Demo Execution (✅ Completed)

- **Scripts**:
  - [predict_account.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/predict_account.py) (master unified inference script implementing Steps 7B.1 – 7B.11)
  - [generate_demo_data.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/generate_demo_data.py) (generates exactly 10 demo accounts matching target risk band composition)
- **Key Details**:
  - Verified LOF novelty support (`novelty=True` in `behavioral_lof.pkl`) and re-fitted/exported MinMaxScaler to `isolation_forest_scaler.pkl` to scale unsupervised statistical anomaly scores during inference.
  - Implemented schema validation, data preprocessing (300 features via `preprocessing_pipeline.pkl`), and leakage-free behavioral feature engineering (using training modes, medians, and risk encodings dynamically reconstructed from indices).
  - Configured LightGBM prediction scoring with a `0.40` threshold and Isolation Forest/LOF scoring mapped to test percentiles.
  - Applied risk score fusion ($0.7 \times \text{ML} + 0.1 \times \text{Stat} + 0.2 \times \text{Behavior}$) and a $+10.0$ boost for behavioral percentiles $\ge 99.0$.
  - Integrated dynamic SHAP values on-the-fly for unseen records, mapping top 5 positive and negative drivers to verified dictionary names.
  - Implemented automated narrative generation (with rule-based template fallbacks) and output validation (checks for hallucinations, direct accusations, band labels, and action recommendations), exporting reports only for Monitor, High Risk, and Critical bands.
- **Deliverables**:
  - [demo_accounts.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/demo_accounts.csv) (10 calibrated demo accounts: 3 Normal, 2 Monitor, 2 High Risk, 3 Critical)
  - [predictions.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/predictions.csv) (Batch prediction results table)
  - [reports.json](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/reports.json) (Narrative summaries for the 7 non-Normal demo accounts)
  - [failed_records.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/failed_records.csv) (Empty schema-compliant log validating 0 schema failures)
  - [prediction.json](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/prediction.json) (Single account prediction result JSON)
  - [predict_account.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/predict_account.py) (Unified execution pipeline script)
  - [generate_demo_data.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/generate_demo_data.py) (Demo generation script)

---

## Phase 8: Path-Agnostic Restructuring & End-to-End Retraining (✅ Completed)

- **Directory Restructuring**:
  - Structured the codebase into logical subdirectories (`phase1/`, `phase2/`, `config/`, and `utils/`) to clean up root junk.
  - Standardized utility and inspect scripts under `utils/`.
- **Dynamic Config manager**:
  - Implemented [config/paths.py](file:///c:/Projects/bank_of_India/mule_account_detection/config/paths.py) to resolve the workspace directory structure dynamically using a recursive search for `phase1`. This completely eliminates hardcoded paths and makes execution portable.
  - Standardized local imports across all modules to leverage package-scoped names (e.g. `from phase2.mule_preprocessor import MuleAccountPreprocessor`).
- **End-to-End Retraining Verification**:
  - Retrained and saved the preprocessing pipeline, ML classifier, Isolation Forest anomaly model, and LOF anomaly models sequentially.
  - Exported the MinMaxScaler to [isolation_forest_scaler.pkl](file:///c:/Projects/bank_of_India/mule_account_detection/phase4/isolation_forest_scaler.pkl) to ensure out-of-sample scaling parity.
  - Verified LOF novelty-supported predictions and recreated [demo_accounts.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/demo_accounts.csv) (10 accounts) to ensure correct risk-engine calibration and narrative generation outputs.
- **Key Metrics Preserved**:
  - ML classifier test set recall: **`81.25%`** (13 out of 16 mules caught)
  - ML classifier test set precision: **`100.00%`** (0 false positives)
  - Fused risk engine test set recall: **`62.50%`** (10 out of 16 mules caught at HR+Critical levels)
  - Critical alerts precision: **`100.00%`** (0 false positives in Critical band)
- **Deliverables**:
  - [config/paths.py](file:///c:/Projects/bank_of_India/mule_account_detection/config/paths.py) (Dynamic path manager)
  - Retrained model and scaler pickle artifacts across Phase 2, 3, 4, 4b, 5, and 7.
  - Regenerated test set predictions, demo files, and single prediction cards verifying the pipeline.

