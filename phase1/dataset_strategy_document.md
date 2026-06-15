# Dataset Strategy Document

This document defines the master pre-processing strategy for the Mule Account Detection dataset, outlining features to drop, features to keep, and specific feature engineering and preprocessing steps.

---

## 1. Permanent Drop List (`DROP_COLUMNS`)

Based on our audit of leakage, duplicate, constant, and high-missing (>90% missing) columns, the following list of columns must be dropped from the dataset before training.

### Dropping Categories Summary
- **Index Column**: `Unnamed: 0` (ordering leak)
- **Target Leakage**: `F3912` (proxy target)
- **Temporal Leakage**: `F2230` (perfect class separator by month)
- **Constant Features**: 359 columns (zero variance)
- **High-Missing Features (>90%)**: 516 columns (too sparse to impute or model)
- **Redundant Duplicates**: `F2506` (duplicate of `F2507`), `F2408` (duplicate of `F2409`)

### Python Copy-Paste List
Below is the python list containing the exact column names to drop:

```python
DROP_COLUMNS = [
    # Index and Leakage Columns
    "Unnamed: 0", "F3912", "F2230",
    
    # Redundant Duplicates
    "F2506", "F2408",

    # Constant Features (359 columns)
    "F128", "F131", "F181", "F182", "F183", "F184", "F185", "F186", "F189", "F192", 
    "F236", "F239", "F290", "F293", "F390", "F393", "F437", "F440", "F492", "F495", 
    "F539", "F542", "F594", "F597", "F616", "F617", "F618", "F619", "F620", "F621", 
    "F640", "F641", "F642", "F643", "F644", "F645", "F646", "F647", "F648", "F649", 
    "F650", "F651", "F652", "F653", "F654", "F655", "F656", "F657", "F658", "F659", 
    "F660", "F661", "F662", "F663", "F688", "F689", "F690", "F691", "F692", "F693", 
    "F694", "F695", "F696", "F697", "F698", "F699", "F700", "F701", "F702", "F703", 
    "F704", "F705", "F724", "F725", "F726", "F727", "F728", "F729", "F748", "F749", 
    "F750", "F751", "F752", "F753", "F754", "F755", "F756", "F757", "F758", "F759", 
    "F760", "F761", "F762", "F763", "F764", "F765", "F766", "F767", "F768", "F769", 
    "F770", "F771", "F791", "F794", "F796", "F797", "F798", "F799", "F800", "F801", 
    "F802", "F803", "F804", "F805", "F806", "F807", "F810", "F813", "F820", "F821", 
    "F822", "F823", "F824", "F825", "F833", "F834", "F836", "F837", "F856", "F857", 
    "F858", "F859", "F860", "F861", "F863", "F866", "F868", "F869", "F870", "F871", 
    "F872", "F873", "F874", "F875", "F876", "F877", "F878", "F879", "F894", "F897", 
    "F899", "F900", "F902", "F903", "F904", "F905", "F906", "F907", "F908", "F909", 
    "F910", "F911", "F912", "F913", "F914", "F915", "F918", "F921", "F923", "F926", 
    "F928", "F929", "F930", "F931", "F932", "F933", "F965", "F968", "F1018", "F1019", 
    "F1020", "F1021", "F1022", "F1023", "F1073", "F1076", "F1126", "F1127", "F1128", 
    "F1129", "F1130", "F1131", "F1181", "F1184", "F1234", "F1235", "F1236", "F1237", 
    "F1238", "F1239", "F1289", "F1292", "F1342", "F1343", "F1344", "F1345", "F1346", 
    "F1347", "F1397", "F1400", "F1450", "F1451", "F1452", "F1453", "F1454", "F1455", 
    "F1505", "F1508", "F1558", "F1559", "F1560", "F1561", "F1562", "F1563", "F1613", 
    "F1616", "F1666", "F1667", "F1668", "F1669", "F1670", "F1671", "F1674", "F1677", 
    "F1721", "F1724", "F1774", "F1775", "F1776", "F1777", "F1778", "F1779", "F1782", 
    "F1785", "F1829", "F1832", "F1882", "F1883", "F1884", "F1885", "F1886", "F1887", 
    "F1890", "F1893", "F1937", "F1940", "F1990", "F1991", "F1992", "F1993", "F1994", 
    "F1995", "F1998", "F2001", "F2045", "F2048", "F2098", "F2099", "F2100", "F2101", 
    "F2102", "F2103", "F2106", "F2109", "F2153", "F2156", "F2206", "F2207", "F2208", 
    "F2209", "F2210", "F2211", "F2214", "F2217", "F2312", "F2360", "F2406", "F2455", 
    "F2458", "F2552", "F2555", "F2600", "F2603", "F2607", "F2655", "F2707", "F2753", 
    "F2756", "F2801", "F2804", "F2807", "F2810", "F2814", "F2860", "F2863", "F2908", 
    "F2911", "F2914", "F2917", "F2921", "F2924", "F2968", "F2971", "F3027", "F3030", 
    "F3074", "F3077", "F3133", "F3179", "F3182", "F3230", "F3233", "F3236", "F3240", 
    "F3243", "F3287", "F3290", "F3338", "F3341", "F3344", "F3348", "F3351", "F3395", 
    "F3398", "F3449", "F3452", "F3456", "F3459", "F3503", "F3506", "F3557", "F3560", 
    "F3564", "F3567", "F3611", "F3614", "F3665", "F3668", "F3672", "F3675", "F3719", 
    "F3722", "F3773", "F3776", "F3780", "F3783", "F3844", "F3845", "F3846", "F3850", 
    "F3851", "F3852", "F3853", "F3854", "F3855", "F3875", "F3876", "F3877", "F3878", 
    "F3879", "F3911",
    
    # High-Missing Features (>90% missing - 516 columns)
    # Note: These columns are detailed in report_lists.txt (e.g. F1, F2, F3, ...)
    # Add all 516 columns here in production pre-processing
]
```

---

## 2. High-Missing Columns Investigation

We investigated all **1,178 columns** with >40% missing values:
1. **Missing > 90% (516 columns)**: Automatically dropped (added to `DROP_COLUMNS`). Imputing features that are 90%+ missing is mathematically unstable and yields near-zero variance.
2. **Missing 40% - 90% (662 columns)**: Evaluated the relationship of these sparse features with the target variable `F3924`.
   - **Methodology**: Calculated the target class distribution for non-missing entries vs missing entries for all 662 features.
   - **Findings**: **0 predictive sparse features** were found in this group. None of the columns showed a target rate in non-null entries significantly higher than the baseline fraud rate (0.89%), nor did they show a significant correlation within non-null values.
   - **Strategy**: Drop these columns or apply median imputation if any are retained for specific business reasons. For maximum noise reduction, we recommend dropping the 662 features in the 40%-90% range as well.

---

## 3. Business Features Analysis

We analyzed the relationship between 7 key categorical business features and the target variable `F3924`:

### A. Account Type (`F3886`)
- **Findings**:
  - **Savings accounts** contain **93.8% of all mule accounts** (76 out of 81) with a class-specific mule rate of **1.28%**.
  - **Current accounts** contain 4 mules (mule rate **0.20%**).
  - **MSME Medium accounts** contain 1 mule (mule rate **1.41%**).
  - All other loan, corporate, micro, and term deposit account types have **0.00%** mule accounts.
- **Risk Score**: High risk: Savings, MSME Medium. Low risk: Others.

### B. Occupation (`F3891`)
- **Findings**:
  - **Students** have the highest rate of mule accounts: **1.94%** (23 out of 1,185 student accounts).
  - **Agriculture** has the second-highest rate: **1.26%** (14 out of 1,112 accounts).
  - **Salaried** accounts have a rate of **0.73%** (14 out of 1,909).
  - **Self-employed** accounts have a rate of **0.66%** (26 out of 3,951).
  - **Housewives** have a rate of **0.45%** (3 out of 660).
- **Risk Score**: High risk: Students, Agriculture. Medium risk: Salaried, Self-employed.

### C. Category / Area (`F3890`)
- **Findings**:
  - **Category R** (Rural/Regional) has the highest mule rate: **1.44%** (29 out of 2,015 accounts).
  - **Category SU** (Semi-Urban) has a rate of **0.88%** (21 out of 2,390).
  - **Category U** (Urban) has a rate of **0.73%** (13 out of 1,777).
  - **Category M** (Metropolitan) has a rate of **0.62%** (18 out of 2,900).
- **Risk Score**: Rural areas have the highest risk, decreasing as urbanization increases.

### D. Customer Segment (`F3893`)
- **Findings**:
  - **RETAIL** accounts contain **93.8% of all mule accounts** (76 out of 81) with a rate of **1.18%**.
  - **CORPORATE** accounts contain only 5 mules (rate **0.19%**).
- **Risk Score**: Retail is high risk; Corporate is low risk.

### E. Gender / Status (`F3892`)
- **Findings**:
  - **Males (M)** have a higher mule rate: **1.26%** (63 out of 5,007).
  - **Females (F)** have a rate of **0.92%** (13 out of 1,416).
  - **Missing status** has a rate of **0.19%** (5 out of 2,598).
- **Risk Score**: Males show a slightly higher risk than females.

### F. Historical Code (`F3889`)
- **Findings**:
  - **G365D** (General 365 Days) has **88.9% of all mule accounts** (72 out of 81) with a rate of **0.95%**.
  - **L365D** has a rate of **1.26%** (5 out of 397).
  - **L180D** has a rate of **0.96%** (3 out of 313).
  - Others (`L7D`, `L90D`, `L14D`) have **0.00%** mules.

---

## 4. Datetime Columns Analysis

We analyzed the two datetime columns: `F2230` (observation date) and `F3888` (account opening date).

### A. Observation Date (`F2230`) — Confirmed Leakage
- **Findings**:
  - Normal accounts are only observed in **October 2025** (9,001 rows).
  - Mule accounts are only observed in **September 2025** (48), **November 2025** (23), and **December 2025** (10).
- **Verdict**: Perfect target leakage. Dropped from the training set.

### B. Account Age (`F3888`)
- **Findings**:
  - Account opening year ranges from 1900 to 2025.
  - **5,504 accounts** have no year (parsed as missing), containing 53 mules.
  - **Average Account Age (in Days)** relative to observation date:
    - **Normal Accounts (`0`)**: mean = 3,205 days (~8.8 years); median = 2,856 days (~7.8 years).
    - **Mule Accounts (`1`)**: mean = 3,410 days (~9.3 years); median = 2,914 days (~8.0 years).
- **Key Business Insight**: Mule accounts are **not newly opened accounts**. They are mature accounts with a median age of 8 years. This strongly suggests that fraudsters are taking over active/dormant accounts, renting them, or purchasing mature accounts rather than establishing fresh accounts.

---

## 5. Pre-Processing Recommendations

To prepare the dataset for Phase 3 (Modeling), we must execute the following pipeline:

### 1. Row Shuffling
- **Action**: Shuffle all rows in the dataset using a set random seed (e.g. `random_state=42`) before performing train-test splits. This is required because the raw dataset is grouped sequentially by target class.

### 2. Feature Engineering
- **Account Age**: Create a numeric feature `account_age_days` using `F2230_parsed - F3888_parsed`. Set any missing values to a negative constant or median, and handle outliers.

### 3. Categorical Encoding
- **One-Hot Encoding**: Apply One-Hot Encoding to categorical columns:
  - `F3886` (Account Type)
  - `F3889` (Historical Code)
  - `F3890` (Category)
  - `F3891` (Occupation)
  - `F3892` (Gender/Status)
  - `F3893` (Customer Segment)

### 4. Robust Scaling
- **Action**: Apply `RobustScaler` to all continuous features because several features have extreme outliers (e.g. max values > 1,000,000). Robust scaling is resistant to outliers by scaling based on the IQR (Interquartile Range).

### 5. Imputation
- **Action**: For columns with < 10% missing values, use median imputation. For columns with > 10% missing values (if kept), use indicator features (`is_missing`) paired with median imputation.

### 6. Modeling Strategy for Imbalance
- **Loss Function Adjustment**: Use class-weighted algorithms (e.g., `class_weight='balanced'` in XGBoost, LightGBM, Random Forest).
- **Evaluation**: Report and optimize Precision, Recall, F1-Score, and PR-AUC. Accuracy must not be used as the primary metric.
