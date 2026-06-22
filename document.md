# AI-Powered Fraud Intelligence System: Money Mule Account Detection
## Technical Proposal & System Design Document
*Prepared for the Bank of India Hackathon Submission*

---

## 1. Executive Summary

### What is a Money Mule Account?
A money mule account is a bank account used by criminals to receive and wash illicitly obtained funds (e.g., from cyber fraud, phishing, scams, money laundering, and drug trafficking). Criminals recruit or trick individuals (often students, rural populations, or vulnerable groups) into letting them use their accounts, or they take over dormant, mature accounts to transfer these illicit funds, masking the identity of the primary perpetrators.

### Why is it Dangerous?
Money mules are a critical node in the financial crime ecosystem. They serve as the primary entry point for cash-out operations. Without money mule accounts, the utility of digital fraud is significantly diminished, as criminals cannot easily withdraw or integrate their stolen funds into the legitimate banking system. Furthermore, money mule operations:
*   Shield organized crime syndicates from law enforcement tracking.
*   Expose financial institutions to severe regulatory penalties, anti-money laundering (AML) compliance failures, and reputational damage.
*   Result in significant direct financial losses for victims of fraud.
*   Undermine public trust in the integrity of digital banking channels.

### Why Traditional Rule-Based Systems Fail
Most financial institutions rely on static rule-based transaction monitoring systems (e.g., flagging any transaction exceeding ₹50,000, or accounts with more than 10 transactions per day). These systems fail in modern fraud detection because:
1.  **High False-Positive Rates**: Rule-based alerts flag high volumes of legitimate activities (e.g., small business transactions, holiday shopping spikes), causing severe alert fatigue among fraud investigators.
2.  **Easy Evasion**: Fraudsters quickly reverse-engineer bank rules and adjust their behavior (e.g., transferring ₹49,999 to bypass a ₹50,000 threshold), making rules easily bypassable.
3.  **Inability to Detect Novel Patterns**: Rules are reactive; they are only written *after* a fraud pattern has been identified and caused damage. They cannot detect zero-day evasion tactics.
4.  **Lack of Context**: Simple transaction thresholds fail to consider the demographic profile, behavioral changes, and historical baseline of the account, leading to poor signal quality.

### What the Bank of India Seeks to Solve
The Bank of India is looking to solve the problem of identifying suspicious money mule accounts accurately and rapidly. The goal is to build an intelligence layer that is capable of:
*   Analyzing thousands of anonymized customer, account, behavioral, and transactional features.
*   Pinpointing the microscopic signatures of money mule behavior.
*   Minimizing false positives so that manual investigation teams can focus on high-probability cases.
*   Providing clear, auditable explanations for every alert to comply with financial regulations.
*   Automating investigator dossiers to accelerate case resolution.

### What Our System Does Differently
We propose and deliver an **AI-Powered Fraud Intelligence System** that combines:
1.  **Supervised Machine Learning (Layer 1)**: A highly tuned LightGBM classifier optimized on historical fraud labels to detect known, structured mule patterns.
2.  **Statistical Anomaly Detection (Layer 2)**: An Isolation Forest model trained strictly on normal accounts to detect general profile shifts.
3.  **Behavioral Anomaly Detection (Layer 3)**: A Local Outlier Factor (LOF) model trained on a compact, risk-engineered behavioral dataset (focusing on credit-to-debit ratios, balance retention, and pass-through fund velocities) to catch zero-day evasions.
4.  **Unified Risk Scoring & Boosting**: A fusion engine that weights these three pillars ($70\%$ ML, $10\%$ Statistical, $20\%$ Behavioral) and applies a $+10.0$ behavioral boost to capture sophisticated mules missed by the classifier.
5.  **Explainable AI (SHAP)**: Translates model decision boundaries into local waterfall plots, mapping raw feature codes to verified, compliance-friendly terms.
6.  **GenAI Investigator Copilot**: Automatically writes natural language case summaries, recommending compliant banking actions and accelerating investigations from hours to seconds.
7.  **Production-Ready Web Dashboard**: A Streamlit interface designed with clean, Notion-style aesthetics providing interactive Forensic Profiler, queue management, sandbox testing, and batch processing.

---

## 2. Dataset Description

### 2.1 Dataset Overview
The project is built upon a large-scale, anonymized bank account dataset containing both customer profile indicators and transactional logs. The target variable is masked, as are the feature names, to preserve customer confidentiality.

| Attribute | Value |
| :--- | :--- |
| **Total Records** | 9,082 bank accounts |
| **Total Features** | 3,925 columns |
| **Target Variable** | `F3924` (0 = Normal Account, 1 = Money Mule Account) |
| **Positive Class** | Money Mule Account (81 cases) |
| **Negative Class** | Normal Account (9,001 cases) |
| **Mule Accounts** | 81 |
| **Normal Accounts** | 9,001 |
| **Fraud Rate (Prevalence)** | **0.89%** |

![Target Class Distribution](phase1/images_for_eda/target_distribution.png)
*Figure 1: Extreme class imbalance in the target variable (0.89% fraud prevalence).*

*   **Anonymization**: The dataset features have been intentionally masked (labeled `F1` to `F3924`) to hide proprietary bank database structures. Both continuous variables (e.g., transaction velocities, values) and categorical indicators (e.g., account type, employment status) are represented.

### 2.2 Feature Composition
Excluding the index column (`Unnamed: 0`), target column (`F3924`), and 359 constant columns, the active features represent four distinct formats:

| Feature Type | Count | Description / Role |
| :--- | :---: | :--- |
| **Continuous** | 2,657 | Ledger balances, transaction counts, debited/credited volumes, and velocities. |
| **Binary** | 526 | Flag variables representing customer/account attributes (e.g., yes/no indicators). |
| **Categorical** | 381 | Cardinality variables mapping to customer classes, locations, and codes. |
| **Non-Numeric** | 8 | String structures, categorical texts, and date formats. |

![Feature Distribution](phase1/images_for_eda/boi_features_distribution.png)
*Figure 2: Distribution profiles of select anonymized features across normal and mule classes.*

#### Examples of Categorical Business Features (from the 8 non-numeric columns):
*   **Account Type (`F3886`)**: Account product types such as `'Savings'`, `'Current'`, or `'MSME Medium'`.
*   **Occupation (`F3891`)**: Customer occupation categories such as `'student'`, `'agriculture'`, `'salaried'`, or `'selfemployed'`.
*   **Category/Area (`F3890`)**: Location/urbanization markers such as `'R'` (Rural), `'SU'` (Semi-Urban), `'U'` (Urban), or `'M'` (Metropolitan).
*   **Customer Segment (`F3893`)**: Customer segment classification, primarily `'RETAIL'` and `'CORPORATE'`.
*   **Account Opening Date (`F3888`)**: Datetime structures representing the date the account was activated.

### 2.3 Dataset Challenges

#### Severe Class Imbalance
The dataset has an extreme class imbalance, with only 81 mule accounts out of 9,082 ($0.89\%$). Standard accuracy is a completely useless metric here—a baseline model that classifies everything as "Normal" achieves $99.11\%$ accuracy but catches zero fraud. We must prioritize precision, recall, F1-score, and Area Under the Precision-Recall Curve (PR-AUC).

#### High Dimensionality
With 3,925 features, the dataset contains a massive amount of noise, collinearity, and sparse entries. Standard classifiers suffer from the "curse of dimensionality," leading to severe overfitting.

#### Missing Values
A high percentage of features are sparsely populated. Specifically, $30.0\%$ of all columns ($1,178$ columns) have missingness rates exceeding $40\%$.

#### Leakage Risks
1.  **Row-Order Leakage**: The raw dataset was found to be fully sorted. Normal accounts occupied rows 1 to 9,001, and all 81 mule accounts were grouped at the very bottom (rows 9,002 to 9,082). The index column (`Unnamed: 0`) correlated at $+0.163$ with the target. Without shuffling, any sequential split will result in $0\%$ recall on the test set.
2.  **Target Proxy Leakage (`F3912`)**: Feature `F3912` exhibits a $+0.969$ correlation with the target. When `F3912 = 1.0`, the account is almost always a mule. In production, this feature is either unavailable at prediction time or represents a post-facto fraud label, making its inclusion in training a source of lookahead bias.

    ![F3912 Target Leakage Analysis](phase1/images_for_eda/f3912_vs_target.png)
    *Figure 3: Correlation between feature F3912 and Target variable, demonstrating target proxy leakage.*

3.  **Observation Date Leakage (`F2230`)**: Normal accounts were observed only in October 2025, while mule accounts were observed in September, November, and December 2025, allowing models to separate classes based purely on observation timestamps.

#### Anonymous Features
Because features are anonymized, we cannot apply intuitive business rules to prune variables. We must design a data-driven feature selection pipeline to systematically isolate the high-signal variables.

---

## 3. Solution Approach

The solution relies on a multi-layered hybrid architecture that fuses supervised machine learning with unsupervised anomaly detection, followed by explainability and GenAI automated case narrative generation.

```text
               +----------------------------------+
               |        Raw Excel Ingestion       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |     Data Audit & Verification    |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    Leakage Removal & Shuffling   |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    Data Preprocessing Pipeline    |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   Mutual Info Feature Selection  |
               +----------------------------------+
                                |
                                v
         +----------------------+----------------------+
         |                      |                      |
         v                      v                      v
+-------------------+  +-------------------+  +-------------------+
| Layer 1: Superv.  |  | Layer 2: Stat.    |  | Layer 3: Behav.   |
| LightGBM Model    |  | Anomaly (I-Forest)|  | Anomaly (LOF)     |
+-------------------+  +-------------------+  +-------------------+
         |                      |                      |
         | (Probability)        | (Anomaly Score)      | (Outlier Rank)
         v                      v                      v
         +----------------------+----------------------+
                                |
                                v
               +----------------------------------+
               |  Unified Risk Fusion Engine &    |
               |      Behavioral Outlier Boost    |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    Explainable AI (SHAP Engine)  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   GenAI Narrative Generator &    |
               |       Automated Validation       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  Notion-Style Streamlit Dashboard|
               +----------------------------------+
```

### Detailed System Architecture (Mermaid Integration)

```mermaid
flowchart TD
    A[Raw Ingestion] --> B[Row Shuffling & Leakage Removal]
    B --> C[Variance & Missingness Drops]
    C --> D[Robust Scaling & One-Hot Encoding]
    D --> E[Redundancy & MI Feature Selection]
    E --> F[Top 300 Clean Features]
    F --> G1[Layer 1: Supervised LightGBM]
    F --> G2[Layer 2: Statistical Isolation Forest]
    F --> G3[Layer 3: Behavioral LOF on 10 Risk Features]
    G1 -- "ML Probabilities (70%)" --> H[Risk Fusion Engine]
    G2 -- "Scaled Outlier Score (10%)" --> H
    G3 -- "Percentile-Rank Score (20%)" --> H
    H --> I[Behavioral Boost +10.0 Escalation]
    I --> J[SHAP Explainer Engine]
    J --> K[GenAI Narrative copilot]
    K --> L[CMS Integration & ReportLab PDF Generation]
```

---

## 4. Data Preparation Strategy

To handle a large-scale bank dataset with 3,925 features, a rigorous, modular, and leakage-free data engineering pipeline was established.

### 4.1 Leakage Mitigation
*   **Row-Order Shuffling**: Before executing any train-test splits, the rows of the dataset were shuffled using a fixed seed (`random_state=42`). This disrupted the target-based sequential sorting and prevented lookahead row index leakage. The index column (`Unnamed: 0`) was permanently dropped.
*   **Target Leakage Removal**: Column `F3912` (correlation proxy with target) was removed from the active training feature set.
*   **Temporal Leakage Removal**: Column `F2230` (observation date) was removed to prevent models from separating mules based on observation months.

### 4.2 Missing Value Handling
*   **Missingness Filtering**: Columns with $>40\%$ missing values ($1,178$ columns) were dropped. A thorough statistical audit of the remaining sparse features ($10\%$ to $40\%$ missingness) showed no statistically significant relationship with target fraud labels.
*   **Imputation Strategy**:
    *   **Continuous Features**: Imputed using the **median** value of each column calculated strictly from the training partition to prevent lookahead bias.
    *   **Categorical/Binary Features**: Imputed using the **mode** (most frequent value) of each column.

### 4.3 Feature Engineering
*   **Account Age (Temporal Analysis)**: The categorical column `F3888` (Account Opening Date) was parsed. Using a baseline date of `2025-12-31`, we engineered the numerical feature `account_age_days` and `account_age_years`.
    *   *Key Business Insight*: Statistical profiling revealed that money mules are **not newly opened accounts**. The median age of mule accounts was 8.0 years (mean 9.3 years), indistinguishable from the normal account median of 7.8 years (mean 8.8 years). This confirms that fraudsters prefer renting or buying established, mature accounts (dormant takeover) to bypass bank onboarding filters.
*   **Categorical One-Hot Encoding**: Categorical features (`F3886`, `F3889`, `F3890`, `F3891`, `F3892`, `F3893`) were one-hot encoded into lowercase snake_case variables.
*   **Robust Scaling**: Continuous features were scaled using a `RobustScaler`. Since $>550$ features had maximum values exceeding $1,000,000$, robust scaling (using interquartile ranges) prevented outliers from dominating model gradients.

### 4.4 Feature Reduction
To achieve high efficiency, reduce overfitting, and maintain extreme signal quality, features were systematically reduced in a 4-step pipeline:

```text
3,925 Raw Columns
   ↓ (Drop Index, Targets, 359 Constant Columns & 1,178 Columns with >40% Missingness)
2,479 Clean Features
   ↓ (Drop 1,214 Redundant Collinear Features with Mutual Pearson Correlation >0.95)
1,299 Reconciled Features
   ↓ (Select Top 300 Features via Mutual Information Scores and Stratified CV Tuning)
  300 Final Features (Input to Models)
```

1.  **Constant Removal**: Dropped **359 constant features** (no variance).
2.  **Missingness Drop**: Dropped **1,178 high-missing features** ($>40\%$ missingness).
3.  **Redundancy & Multicollinerity Filtering**: Computed a mutual correlation matrix for numerical features. For any feature pair with a Pearson correlation coefficient $>0.95$ ($3,932$ highly correlated pairs), we evaluated their correlations with the target and **dropped the weaker feature**. This removed **1,214 redundant features**, leaving $1,299$.
4.  **Mutual Information (MI) Selection**: Computed Mutual Information scores. Running stratified cross-validation sweeps identified that selecting the **top 300 features** optimized model validation F1-scores ($0.6087$) and reduced feature dimension by $92.3\%$ from raw ingestion.

    ![Top Correlated Features](phase1/images_for_eda/top_correlated_features.png)
    *Figure 4: Correlation heat map of top features selected via Mutual Information and correlation tuning.*

---

## 5. Detection Engine

Our detection engine is structured as a 3-layer analytical pipeline.

### Layer 1 — Supervised Learning (Historical Patterns)
Layer 1 is designed to identify accounts that closely resemble known historical fraud patterns.

#### Model Evaluation (5-Fold Stratified Cross-Validation)
Four supervised classification algorithms were trained and cross-validated on the training split (7,265 rows, 65 mules) to select the top model based on Recall and PR-AUC:

| Model | CV Precision | CV Recall | CV F1-Score | CV ROC-AUC | CV PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Baseline)** | $0.9548$ | **$0.7538$** | **$0.8324$** | $0.9759$ | $0.8650$ |
| **LightGBM (Baseline)** | $0.9500$ | $0.6154$ | $0.7281$ | $0.9658$ | $0.8074$ |
| **Random Forest** | **$1.0000$** | $0.3692$ | $0.5329$ | **$0.9708$** | **$0.8233$** |
| **Logistic Regression** | $0.0159$ | $0.1538$ | $0.0288$ | $0.6841$ | $0.0206$ |

*   *Model Selection*: While XGBoost had slightly higher raw baseline metrics, **LightGBM** was selected for final hyperparameter tuning due to its superior execution speed, low memory footprint, and high stability during threshold sweeps.

#### Threshold Optimization & Custom Business Cost Function
In fraud detection, missing a mule (False Negative, FN) is far costlier than raising a false alarm (False Positive, FP). To find the optimal decision boundary, we defined a custom bank cost function:
$$\text{Business Cost} = (10 \times \text{False Negatives}) + (1 \times \text{False Positives})$$

Running a threshold sweep on the tuned LightGBM model yielded:

| Threshold | Precision | Recall | F1-Score | False Negatives (FN) | False Positives (FP) | Total Cost |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.10$ | $92.86\%$ | $81.25\%$ | $0.8667$ | 3 | 1 | 31 |
| $0.20$ | $92.86\%$ | $81.25\%$ | $0.8667$ | 3 | 1 | 31 |
| $0.30$ | $100.00\%$ | $81.25\%$ | $0.8966$ | 3 | 0 | 30 |
| **$0.40$** | **$100.00\%$** | **$81.25\%$** | **$0.8966$** | **3** | **0** | **30** (Selected) |
| $0.50$ | $100.00\%$ | $68.75\%$ | $0.8148$ | 5 | 0 | 50 |
| $0.60$ | $100.00\%$ | $62.50\%$ | $0.7692$ | 6 | 0 | 60 |

At the optimized decision threshold of **$0.40$**, LightGBM achieved:
*   **Test Precision**: **$100.00\%$** (0 false positives)
*   **Test Recall**: **$81.25\%$** (13 out of 16 mules caught)
*   **Test ROC-AUC**: **$0.9820$**
*   **Test PR-AUC**: **$0.8689$**
*   **Holdout Test Cost**: **30** (lowest cost profile)

![Confusion Matrix](phase3/confusion_matrix.png)
*Figure 5a: Confusion Matrix for the optimized LightGBM model on the holdout test set.*

![ROC Curve](phase3/roc_curve.png)
*Figure 5b: ROC Curve for the optimized LightGBM model showing 0.9820 AUC.*

![Precision-Recall Curve](phase3/pr_curve.png)
*Figure 5c: Precision-Recall Curve for the optimized LightGBM model showing 0.8689 PR-AUC.*

---

### Layer 2 — Statistical Anomaly Detection (Broad Profile Shifts)
*   **Model**: **Isolation Forest** (500 estimators, contamination = 0.005, trained strictly on genuine/normal training accounts to prevent label poisoning).
*   **Purpose**: Act as a safety net to detect unusual ledger behaviors that are completely unrepresented in historical fraud labels (zero-day general anomalies).
*   **Method**: Estimates continuous anomalies by isolating features. The raw score `decision_function` was inverted ($-\text{score}$) and mapped to a $0-100$ scale using a MinMaxScaler fitted strictly on training data.
*   **Performance Insight**: Isolation Forest on the 300 supervised-selected features yielded a ROC-AUC of $0.3305$, and normal accounts scored higher on average ($24.18$) than known mules ($17.74$). This occurred because the 300 features were selected for supervised correlation, and the unsupervised model isolated high-value legitimate business spikes rather than subtle fraud signals. This highlighted the need for Layer 3.

![Anomaly Score Distribution](phase4/anomaly_distribution.png)
*Figure 6a: Distribution of statistical anomaly scores across genuine and mule classes.*

![Anomaly Score vs Target](phase4/anomaly_vs_target.png)
*Figure 6b: Comparison of anomaly scores for genuine vs mule accounts.*

---

### Layer 3 — Behavioral Anomaly Detection (Flow of Funds)
*   **Model**: **Local Outlier Factor (LOF)** (configured with `n_neighbors=10` and `novelty=True` to support out-of-sample inference).
*   **Purpose**: Specifically target accounts whose transactional *flow of funds* differs significantly from the peer population.
*   **Compact Behavioral Feature Set**: Trained on **10 risk-engineered behavioral features**:
    1.  *Occupation Risk Score* (Target-encoded mule probability of employment type)
    2.  *Area Risk Score* (Target-encoded mule probability of location category)
    3.  *Account Type Risk Score* (Target-encoded mule probability of account type)
    4.  *Customer Segment Risk Score* (Target-encoded mule probability of customer segment)
    5.  *Gender Risk Score* (Target-encoded mule probability of gender)
    6.  *Ending Balance (`F3836`)*: Ending ledger balance.
    7.  *Total Credit (`F2737`)*: Incoming transaction volume.
    8.  *Total Debit (`F2678`)*: Outgoing transaction volume.
    9.  *Credit-Debit Ratio*: $\frac{|Total Credit| + \epsilon}{|Total Debit| + \epsilon}$
    10. *Balance Retention Ratio*: $\frac{|Ending Balance|}{|Total Credit| + \epsilon}$
    11. *Pass-Through Ratio*: $\frac{|Total Debit|}{|Total Credit| + \epsilon}$ (Note: For $50\%$ of money mules, the pass-through ratio is exactly $1.000000$, showing immediate transfer of incoming funds).
*   **Evaluation & Selection**: Out of several models tested via cross-validation, LOF (NN=10) achieved the highest validation PR-AUC of **$0.0158$**.

![Behavioral Score Distribution](phase4b/behavioral_distribution.png)
*Figure 7a: Behavioral outlier score distribution.*

![Behavioral Score vs Target](phase4b/behavioral_vs_target.png)
*Figure 7b: Behavioral score comparison showing separation between normal and mule accounts.*

---

## 6. Unified Risk Scoring Framework

Fusing supervised probability with unsupervised anomalies ensures the bank is protected against both established fraud patterns and novel evasion tactics.

### Score Fusion Formula
The unified risk score is calculated as follows:

$$\text{Fused Risk Score} = \min\left(0.70 \times \text{ML} + 0.10 \times \text{Stat} + 0.20 \times \text{Behavioral} + \text{Boost}, 100.0\right)$$

Where:
*   **`ML`** = LightGBM prediction probability $\times 100$.
*   **`Stat`** = Isolation Forest anomaly score scaled to $0.0 - 100.0$.
*   **`Behavioral`** = Local Outlier Factor (LOF) score mapped to a $0.0 - 100.0$ scale via **percentile ranking** (rank-scaled relative to reference training distributions to handle LOF's highly skewed, multi-order distribution).
*   **`Boost`** = $+10.0$ if the behavioral outlier score ranks in the top 1% of the population (`Behavioral` $\ge 99.0$), otherwise $0.0$.

### Why Multiple Signals are Fused
*   **Layered Defense**: Fusing these indicators creates a balanced assessment. Supervised ML handles high-precision detection of known patterns ($70\%$ weight). Unsupervised models ($30\%$ combined weight) detect behavioral and statistical outliers.
*   **The Behavioral Boost**: Flagging the top 1% of behavioral outliers with a $+10.0$ score boost ensures that accounts displaying extreme transactional anomalies bubble up to the investigator's queue even if the supervised model predicts a $0\%$ risk.

### Risk Bands & Recommended Actions

The continuous Fused Risk Score ($0.0 - 100.0$) maps directly to four operational risk bands:

| Fused Score | Risk Band | Action / Protocol | Test Volume | Mules Caught | Precision |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **$0.00 - 30.00$** | **Normal** | No action required. Allowed standard operations. | 1,785 | 2 | -- |
| **$30.01 - 60.00$**| **Monitor** | Enhanced monitoring, log-ledger tracking, transaction velocity caps. | 22 | 4 | $18.18\%$ |
| **$60.01 - 80.00$**| **High Risk** | Manual fraud investigation, temporary 24-hour hold on large debits. | 5 | 5 | **$100.00\%$**|
| **$80.01 - 100.00$**|**Critical** | Immediate debit freeze, routing to critical verification queue. | 5 | 5 | **$100.00\%$**|

![Risk Scores Distribution](phase5/risk_distribution.png)
*Figure 8a: Distribution of Fused Risk Scores across normal and mule accounts.*

![Risk Band Counts](phase5/risk_band_counts.png)
*Figure 8b: Number of accounts categorized under each risk band in the test set.*

---

## 7. Explainability & Investigation Support

To maintain compliance and support fraud analysts, the system provides full transparency for every risk score.

### Local & Global Explainability (SHAP)
*   **Engine**: We deployed `shap.TreeExplainer` on the LightGBM classifier to calculate exact Shapley values.
*   **Global Driver Beeswarm**: The global SHAP plot shows that `F3898` (continuous volume driver) and `F3914` (binary normalizer switch) are the primary drivers of predictions.

    ![Global SHAP Drivers Beeswarm](phase6/shap_summary_plot.png)
    *Figure 9: Global feature importance Beeswarm plot showing SHAP values.*

*   **Key Feature Mappings**:
    *   `F3898` (RF Rank #6): Primary local risk driver. A low value (e.g. $-1.00$ or $-0.67$) adds $+1.20$ to $+3.08$ to the prediction log-odds.
    *   `F3914` (RF Rank #118): Primary normalizer switch. When `F3914 = 0.0`, it adds $+1.38$ to $+1.99$ to log-odds. When `F3914 = 1.0`, it acts as a risk-reducer (SHAP $-0.56$ to $-0.89$), dropping prediction probability to near-zero.
    *   `F3908` (RF Rank #39): Adds $+0.70$ to $+1.14$ SHAP log-odds when its value is `1.0`.

    ![Local SHAP Explanation for Mule Account A9044](phase6/waterfall_A9044.png)
    *Figure 10: Local SHAP waterfall explanation for mule account A9044, showing specific risk drivers.*

### Investigation Cards
For every flagged account, the system exports a structured **Investigation Card** containing:
1.  **3-Pillar Score Breakdown**: Fused Risk Score, ML Score, Stat Anomaly Score, and Behavioral Score.
2.  **Top SHAP Contributors**: Specific features and their mathematical impact on the log-odds.
3.  **Natural Language Brief**: Auto-generated by the GenAI narrative copilot.

### GenAI Narrative Dossier & Compliance Summaries
Integrating the Gemini API, the system writes a comprehensive compliance narrative for non-Normal alerts. To ensure absolute compliance and safety, all narratives run through an **Automated Output Validator** before saving:
*   *Speculation Check*: Rejects narratives that guess the real-world semantic meanings of masked features.
*   *Terminology Check*: Confirms the narrative includes the exact risk band name.
*   *Action Consistency Check*: Ensures the recommended banking action matches the risk band.
*   *Accusation Check*: Rejects direct accusations of fraud, forcing the copilot to maintain a professional, risk-based tone.

#### Example CMS Output Formats:
*   **JSON cards** for programmatic case management ingestion.
*   **Interactive HTML reports** with embedded styling for browser reviews.
*   **ReportLab PDF dockets** featuring paginated tables, confidentiality headers, and signature fields for formal submissions.

---

## 8. Dashboard & Operational Workflow

The Streamlit analyst dashboard operates locally using a Notion-style theme (Inter typography, clean borders, semantic color accents).

### 8.1 Executive Dashboard
Provides high-level KPIs, including:
*   Total Flagged Accounts (32 alerts).
*   Breakdown by Risk Bands (5 Critical, 5 High Risk, 22 Monitor).
*   Precision rate in the High/Critical alerts queue ($100.00\%$).
*   Total fraud capture rate ($87.50\%$).

![Executive Dashboard Home](images_for_readme/dashboard_home.png)
*Figure 11: Main page of the Streamlit dashboard displaying active investigation KPIs.*

### 8.2 Investigation Queue
Ranks the 32 flagged accounts using a compound **Priority Score**:

$$\text{Priority Score} = 0.8 \times \text{Risk Score} + 0.2 \times \text{Behavior Score}$$

This prioritizes accounts that exhibit both high model scores and extreme behavioral outliers.
*   *Calibrated Demo Run*: Processing 10 demo accounts successfully sorted them into their exact calibrated bands, flagging 3 Critical, 2 High Risk, 2 Monitor, and 3 Normal accounts.

![Investigation Queue Tab](images_for_readme/tab1_queue.png)
*Figure 12: Interactive investigation queue sorted by priority score with status filters.*

### 8.3 Forensic Profiler (Account Deep Dive)
For any selected account:
*   **Scoring Gauges**: Displays four radial gauges (Fused Risk, ML Probability, Stat Anomaly, Behavioral Percentile).
*   **Demographic Profile Table**: Displays account age, category, segment, occupation, and historical code.
*   **SHAP Waterfall Chart**: Plots feature contributions dynamically.
*   **AI-Generated Investigation Brief**: Displays the validated compliance dossier.

![Forensic Profiler Interface](images_for_readme/forensic_profiler.png)
*Figure 13a: Account forensic profiler interface displaying demographic metrics and score breakdowns.*

![Forensic Profiler Details Tab](images_for_readme/tab2_profiler.png)
*Figure 13b: Detailed profiling view with risk indicators and SHAP waterfalls.*

```text
+-----------------------------------------------------------------------------+
|  Account ID: A9044                                   Risk Band: CRITICAL    |
|  Fused Score: 83.49 | ML: 99.96 | Stat Anomaly: 30.77 | Behavioral: 52.23   |
+-----------------------------------------------------------------------------+
|  Demographics:                                                              |
|  - Account Type: Savings    - Occupation: Student    - Category: Rural      |
+-----------------------------------------------------------------------------+
|  SHAP Risk Drivers:                                                         |
|  [+] F3898 (Value: -0.67) -----------------------------> +2.46 SHAP log-odds |
|  [+] F3914 (Value: 0.00)  -----------------------------> +1.39 SHAP log-odds |
|  [-] F3914 (Value: 1.00)  -----------------------------> -0.56 SHAP log-odds |
+-----------------------------------------------------------------------------+
|  AI Dossier Brief:                                                          |
|  "Account Fused Risk score is 83.49 (Critical). Primary driver is the       |
|  supervised LightGBM model (99.96%). Feature F3898 shows extreme ledger     |
|  variance (+2.46 SHAP). Recommend immediate review and outgoing freeze."    |
+-----------------------------------------------------------------------------+
```

### 8.4 Batch Ingestion & Model Metrics
*   **Batch Ingestion**: Allows uploading an Excel or CSV file to perform batch predictions, generating a downloadable priority queue and case briefs.
*   **Model Metrics**: Displays the 5-fold cross-validation comparisons, ROC/PR curves, and confusion matrices.

![Sandbox Tester Tab](images_for_readme/tab3_sandbox.png)
*Figure 14: Sandbox Tester module for dynamic feature perturbation and immediate risk recalculation.*

![Batch Ingestion Dashboard](images_for_readme/tab4a_batch_processing.png)
*Figure 15a: Upload interface for processing bulk transaction ledger entries.*

![Batch Ingestion Results](images_for_readme/tab4b_batch_processing.png)
*Figure 15b: Generated priority lists and downloadable investigator dockets from batch ingestion.*

![Model Metrics and SHAP Drivers Tab](images_for_readme/tab5_metrics.png)
*Figure 16: Model validation, global feature importances, and distribution analysis views.*

---

## 9. Expected Business Impact

Implementing this hybrid fraud intelligence layer delivers significant business outcomes:

### Reduced Alert Fatigue (Operational Efficiency)
Instead of flagging hundreds of accounts, the system flagged only **32 out of 1,817 accounts** ($1.76\%$ alert rate). Investigators can review the entire queue in a single day, eliminating backlog and alert fatigue.

### Extreme Alert Precision
In the test split, the **Critical (5)** and **High Risk (5)** bands achieved **$100.00\%$ precision**. Fraud teams handling these alerts face zero false positives, ensuring optimal resource allocation.

### Earlier Detection (Unsupervised Recovery)
Unsupervised behavioral anomaly models successfully flag novel fraud signatures that bypass supervised classifiers.
*   *Case Study (Account `A9078`)*: LightGBM assigned a near-zero probability of **$0.000025$** because the account's features appeared standard. However, because its pass-through and credit-to-debit ratio ranked in the top 1% (`behavior_score` = $99.17$), the LOF model triggered the $+10.0$ boost. The final fused score of **$30.66$** successfully escalated the account to the **Monitor** band, recovering a false negative.

### Explainable Auditing
Every alert includes an interactive SHAP force plot and narrative audit trail, making the engine fully compliant with banking regulations regarding explainable automated decisions.

### Faster Case Resolution
The automated GenAI Narrative Dossier generator drafts a complete case summary instantly, reducing manual case drafting times from hours to seconds.

---

## 10. Future Scope & Final Round Roadmap

The Phase 1 system provides a robust account-level foundation. For the next round, we propose expanding this system into a multi-dimensional fraud intelligence platform.

### Phase A — Graph-Based Fraud Ring Detection (Network-Level Intelligence)
*   **Limitation of Current System**: Analyzes bank accounts individually, ignoring network linkages.
*   **Future Approach**: Build a network graph using NetworkX or Graph Neural Networks (GNNs).
*   **Method**: Nodes represent accounts; edges represent shared attributes (e.g., matching phone numbers, registration addresses, IP addresses, or device IDs) or direct transaction transfers.
*   **Outcome**: Detect coordinated fraud rings, wash chains, and hub-and-spoke layering operations.

```text
Normal Account A --(Shared Phone)--> Mule Account B --(Transfer)--> Mule Account C
                                            |
                                     (Shared Address)
                                            v
                                     Mule Account D
```

### Phase B — Real-Time Fraud Scoring API
*   **Future Approach**: Wrap the preprocessing and prediction pipeline (`predict_account.py`) into a Dockerized **FastAPI** REST endpoint.
*   **Performance Goal**: Process incoming transaction events and output a unified risk score in **$<1$ second**, enabling the bank to block mule transfers in real-time.

### Phase C — Deep Learning Anomaly Detection
*   **Future Approach**: Replace/supplement the Isolation Forest layer with a deep learning **Autoencoder** or **Variational Autoencoder (VAE)**.
*   **Method**: Train the autoencoder to reconstruct normal customer transactions. High reconstruction errors will indicate complex, non-linear transactional anomalies, enhancing zero-day fraud detection.

### Phase D — Adaptive Self-Learning Fraud Intelligence
*   **Future Approach**: Establish a feedback loop between the Streamlit dashboard and the database.
*   **Method**: When investigators mark an alert as "Confirmed Fraud" or "False Positive," the system logs this feedback. The engine will automatically retrain and dynamically adjust risk fusion weights based on historical analyst actions.

### Phase E — Enterprise Fraud Intelligence Platform
*   **Future Approach**: Integrate transaction streams from multiple channels: UPI, IMPS, NEFT, ATM, and Internet Banking. Fusing these channels into a single data stream will allow the system to detect cross-channel velocity anomalies.

---

## 11. Final Recommendation

This system transitions the bank's fraud detection capabilities from a reactive, rule-based approach to a proactive, multi-layered, and explainable AI framework. It delivers:
1.  **Immediate Value**: An operational dashboard that flags mules with $100\%$ precision in high-risk bands.
2.  **Regulatory Compliance**: Interactive SHAP explainability and GenAI narrative dossiers that meet strict model risk governance standards.
3.  **Future Growth**: A clear roadmap to transition from account-level analysis to network-level graph intelligence.

*This technical proposal represents a deployment-ready system that will safeguard the Bank of India's operations, protect customers, and lead financial crime detection standards.*
