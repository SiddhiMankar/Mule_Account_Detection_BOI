# Data Understanding Report

This report presents the Phase 1 — Data Understanding analysis for the Mule Account Detection dataset.

---

## 1. Dataset Dimensions

- **Total Rows**: 9,082
- **Total Columns**: 3,925
  - *Note*: This includes an index column (`Unnamed: 0`), 3,923 feature columns (`F1` to `F3923`), and the target column (`F3924`).
- **Duplicate Rows**: 0 (No duplicate rows were found in the dataset).

---

## 2. Missing Values Summary

Features have been categorized based on their missing value percentages:

| Missing % | Category / Action | Count | Percentage of Columns | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **0%** | Keep | 90 | 2.3% | Complete columns. Includes Target (`F3924`) and Index (`Unnamed: 0`). |
| **< 10%** | Impute | 2,450 | 62.4% | Can be handled with standard imputation (mean, median, mode). |
| **10% - 40%** | Investigate | 207 | 5.3% | Require careful handling or advanced imputation (e.g. KNN, MICE). |
| **> 40%** | Candidate for Removal | 1,178 | 30.0% | High missing rates. Likely to be dropped to avoid noise. |

> [!TIP]
> The list of 1,178 columns with > 40% missing values (e.g. `F1`, `F2`, `F3`, ...) and 207 columns with 10%-40% missing values (e.g. `F20`, `F23`, `F26`, ...) can be found in the scratch directory: [report_lists.txt](file:///C:/Users/Siddhi/.gemini/antigravity-ide/brain/69f36285-2b47-443e-8111-f89150a3d33c/scratch/report_lists.txt).

---

## 3. Target Analysis (F3924)

The target variable is `F3924` (Mule Account Flag).

- **Normal Accounts (`0`)**: 9,001 (99.11%)
- **Mule Accounts (`1`)**: 81 (0.89%)
- **Imbalance Ratio**: ~111:1 (Highly imbalanced)

![Target Distribution](file:///C:/Users/Siddhi/.gemini/antigravity-ide/brain/69f36285-2b47-443e-8111-f89150a3d33c/target_distribution.png)

### Why This Matters

1. **Evaluation Metrics**: Standard **Accuracy** will be a misleading metric (a dummy model predicting `0` always achieves 99.11% accuracy). We must evaluate models using:
   - **Precision** & **Recall**
   - **F1-Score**
   - **Precision-Recall AUC (PR-AUC)**
   - **Receiver Operating Characteristic AUC (ROC-AUC)**
2. **Sampling Strategy**: To help models learn the minority class, we should explore:
   - **Class Weights** (cost-sensitive learning)
   - **SMOTE** (Synthetic Minority Over-sampling Technique)
   - **Undersampling** of the majority class

---

## 4. Feature Type Summary

Excluding the index column (`Unnamed: 0`), target column (`F3924`), and 359 constant features:

- **Binary Features (unique values = 2)**: 526
- **Categorical Features (low cardinality <= 10 or string)**: 381
- **Continuous Features**: 2,657

### Non-Numeric Columns

There are **8 non-numeric columns** containing strings, dates, and categories:

| Column | Type | Unique Count | Sample Values | Notes |
| :--- | :--- | :---: | :--- | :--- |
| `F2230` | String (Date) | 4 | `'2025-10-01'` | Date format. |
| `F3886` | String | 17 | `'Savings'` | Account type. |
| `F3888` | String (Datetime) | 4292 | `'2011-01-08 00:00:00'` | Datetime format. |
| `F3889` | String | 7 | `'G365D'`, `'L365D'`, `'L7D'` | Code categories. |
| `F3890` | String | 4 | `'R'`, `'SU'` | Short categories. |
| `F3891` | String | 7 | `'selfemployed'`, `'student'` | Employment type. |
| `F3892` | String | 3 | `'M'` | Gender/Status category. |
| `F3893` | String | 2 | `'RETAIL'` | Customer segment category. |

---

## 5. Basic Statistical Analysis & Red Flags

A summary description of features revealed several important patterns:

1. **Extremely Large Values (Outliers)**:
   - 551 features have maximum values greater than 1,000,000.
   - For example:
     - `F625`: max = 411,684,475.0 (mean = 107,929.7)
     - `F627`: max = 45,446,587.0 (mean = 12,897.2)
     - `F631`: max = 30,000,000.0 (mean = 26,439.3)
   - These features will require robust scaling (e.g. RobustScaler) or clipping/log transformation.
2. **Negative Values**:
   - 949 features contain negative values (e.g., `F2522` min = -0.75, `F2523` min = -0.9). These appear to be pre-scaled or ratio features, which is normal but requires care during scaling.
3. **Constant Features (Useless Columns)**:
   - **359 features** have 1 or fewer unique values when ignoring NaNs (meaning they are completely constant or contain only one value plus NaNs). Examples include `F128`, `F131`, `F181`, `F182`, etc.
   - These columns provide zero variance and will be removed.

---

## 6. Correlation Exploration

### Top Positively Correlated Features with F3924
1. **`F3912`**: **0.96907** (Extremely high correlation)
2. **`F2507`**: **0.18452**
3. **`F2506`**: **0.18452**
4. **`Unnamed: 0`**: **0.16284** (Index column)
5. **`F2409`**: **0.15715**
6. **`F2408`**: **0.15715**
7. **`F515`**: **0.13699**
8. **`F518`**: **0.12691**

### Top Negatively Correlated Features with F3924
1. **`F2502`**: **-0.09807**
2. **`F2503`**: **-0.09787**
3. **`F2474`**: **-0.08811**
4. **`F2472`**: **-0.08508**
5. **`F144`**: **-0.07771**

### Critical Target Leakage & Row Order Findings

#### 1. Row Order Leakage (`Unnamed: 0`)
The row index (`Unnamed: 0`) shows a correlation of **0.16284** with the target. Analyzing the indices of the target classes shows:
- **Normal Accounts (`0`)**: Rows **1 to 9,001**
- **Mule Accounts (`1`)**: Rows **9,002 to 9,082**
- **Insight**: The dataset is completely sorted by the target! The normal accounts are at the top, and all mule accounts are appended at the bottom.
- **Action**: **We MUST shuffle the dataset before any train-test splits or cross-validation.** Also, `Unnamed: 0` (the index) must be dropped as a feature.

#### 2. Target Leakage candidate (`F3912`)
The feature `F3912` has a **0.96907** correlation with the target:
- When `F3912 = 0`, target is `0` (normal) for 8,998 rows and `1` (mule) for 2 rows.
- When `F3912 = 1`, target is `1` (mule) for 79 rows and `0` (normal) for 3 rows.

![F3912 vs Target](file:///C:/Users/Siddhi/.gemini/antigravity-ide/brain/69f36285-2b47-443e-8111-f89150a3d33c/f3912_vs_target.png)

- **Insight**: This feature is almost a perfect proxy for the target. Unless it is a feature available at the exact moment of inference (unlikely in real-world settings, or perhaps it represents a preliminary fraud score), using it will make model training trivial but might cause massive leakage. We should plan to build models **both with and without `F3912`** to test performance and prevent leakage.

#### 3. BOI-Highlighted Features
We inspected the features highlighted:
- **`F527`**: Numeric, Correlation = -0.00281, Missing = 8.68%.
- **`F115`**: Numeric, Correlation = 0.05759, Missing = 3.94%.
- **`F2082`**: Numeric, Correlation = -0.02426, Missing = 0.08%.
- **`F3889`**: Categorical, split into codes like `G365D` (7,544 rows), `L365D` (397 rows), `L7D` (386 rows), etc.
  - Target distribution by `F3889` category:
    - `'G365D'`: 0.95% mule accounts (72 out of 7,544)
    - `'L365D'`: 1.26% mule accounts (5 out of 397)
    - `'L180D'`: 0.96% mule accounts (3 out of 313)
    - `'L31D'`: 0.68% mule accounts (1 out of 148)
    - Others: 0.0% mule accounts.

![BOI Features Distribution](file:///C:/Users/Siddhi/.gemini/antigravity-ide/brain/69f36285-2b47-443e-8111-f89150a3d33c/boi_features_distribution.png)
![Top Correlated Features](file:///C:/Users/Siddhi/.gemini/antigravity-ide/brain/69f36285-2b47-443e-8111-f89150a3d33c/top_correlated_features.png)

---

## 7. Preprocessing Steps Required Later

Based on our findings, the following preprocessing steps will be required in Phase 2:

1. **Row Shuffling**: The dataset must be shuffled before any train-test splitting or cross-validation.
2. **Feature Dropping**:
   - Drop the index column `Unnamed: 0`.
   - Drop the **359 constant columns** (no variance).
   - Drop columns with **> 40% missing values** (1,178 columns).
3. **Imputation**:
   - Impute missing values for remaining columns. For columns with < 10% missing, simple median/mode imputation is appropriate. For columns with 10-40% missing, we can investigate KNN/iterative imputation.
4. **Encoding Categorical Columns**:
   - Convert string/categorical columns (`F2230`, `F3886`, `F3888`, `F3889`, `F3890`, `F3891`, `F3892`, `F3893`) using One-Hot Encoding or target encoding.
   - For datetime columns (`F3888`, `F2230`), parse them into year/month/day/day-of-week numeric features.
5. **Scaling**:
   - Since many features have large outlier values, use robust scaling methods (like `RobustScaler`) to prevent outliers from dominating the model coefficients/gradients.
6. **Class Imbalance Mitigation**:
   - Apply class weights, SMOTE, or undersampling during model training.
