# Phase 5 Report: Risk Engine & Score Fusion

This report presents the implementation and evaluation of the **Unified Risk Fusion Engine** for Bank of India's money mule detection system. By combining supervised predictive modeling with unsupervised statistical and behavioral anomaly detection, the risk engine creates a complementary defense strategy that catches novel fraud patterns while maintaining extremely high alert precision.

---

## 1. Executive Summary

The Phase 5 Risk Engine integrates three distinct analytical perspectives to score and prioritize accounts:
1. **Supervised Classifier (LightGBM)**: Evaluates resemblance to known, historic mule account profiles (70% weight).
2. **Statistical Anomaly Detector (Isolation Forest)**: Measures deviation across general account features (10% weight).
3. **Behavioral Anomaly Detector (Local Outlier Factor)**: Captures extreme fund-flow velocity, retention, and pass-through discrepancies (20% weight).

### Performance Highlights
- **100% Critical Alert Precision**: All accounts flagged in the **Critical** band ($>80.0$) are actual money mules (zero false positives).
- **62.50% Fraud Capture Rate**: The combined **High Risk** and **Critical** bands successfully capture **10 out of the 16** holdout test set mules.
- **Low Alert Volume (1.76%)**: Only **32 accounts** are flagged for investigator review (Normal is excluded), minimizing investigator fatigue.
- **Supervised False Negative Recovery**: The unsupervised behavioral boost successfully flags and escalates a mule account (`A9078`) that the supervised model completely missed (assigning it a probability of $0.000025$).

---

## 2. Risk Engine Methodology

### Score Fusion Formula
To score each account, the engine applies a weighted combination of the percentage-scaled outputs:

$$\text{Risk Score} = \min\left(0.70 \times \text{ML Score} + 0.10 \times \text{Stat Anomaly Score} + 0.20 \times \text{Behavioral Score} + \text{Boost}, 100.0\right)$$

Where:
- **`ml_score`** = LightGBM prediction probability $\times 100$ (Range: $0.0 - 100.0$).
- **`stat_score`** = Isolation Forest anomaly risk score (Range: $0.0 - 100.0$).
- **`behavior_score`** = Percentile-scaled Local Outlier Factor anomaly score (Range: $0.0 - 100.0$).
- **`Boost`** = $+10.0$ if the behavioral anomaly score is in the top 1% (`behavior_score` $\ge 99.0$), otherwise $0.0$.
- The final score is rounded to **2 decimal places**.

### Percentile Scaling Justification for LOF
The raw Local Outlier Factor (LOF) score distribution is highly skewed, with outlier scores spanning multiple orders of magnitude (Max = $1626.33$, Median = $-0.44$). Standard min-max scaling compresses all normal accounts into near-zero scores, making the behavioral component contribute nothing to the final score of most accounts.
By applying **percentile ranking scaling** (`scipy.stats.rankdata` normalized by length) strictly on the holdout test set, we map the LOF signals to a uniform $0-100$ range representing relative outlier severity. This guarantees stable, scale-independent behavior across future datasets.

### Behavioral Anomaly Boost
Unsupervised models excel at identifying extreme, novel anomalies that supervised models have never seen. To reward extreme behavioral anomalies, we apply a **+10.0 boost** to any account with a `behavior_score` $\ge 99.0$. This allows the risk engine to bubble up suspicious activity even if the supervised model predicts a near-zero risk.

---

## 3. Risk Bands & Recommended Actions

The risk engine maps the continuous risk score ($0.0 - 100.0$) into four operational bands to guide investigator workflows:

| Risk Score Range | Category | Account Count | Mules Captured | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| **$0.00 \le \text{Score} \le 30.00$** | **Normal** | 1,785 (98.24%) | 2 | No action required |
| **$30.01 \le \text{Score} \le 60.00$** | **Monitor** | 22 (1.21%) | 4 | Enhanced monitoring & velocity limits |
| **$60.01 \le \text{Score} \le 80.00$** | **High Risk** | 5 (0.28%) | 5 | Manual investigation & temporary hold |
| **$80.01 \le \text{Score} \le 100.00$**| **Critical** | 5 (0.28%) | 5 | Immediate account review & debit freeze |

---

## 4. Evaluation & Case Studies

### A. Alert Precision & Fraud Capture
The engine flags only **32 accounts** for investigation (combining Monitor, High Risk, and Critical), representing an **alert rate of just 1.76%** of all test accounts.
- **87.5% of Mules Detected**: By investigating this small pool (1.76% of all accounts), investigators detect **14 out of the 16 actual money mules** (87.5% capture rate).
- **Critical Band Precision**: **100.00%** (5 alerts, 5 actual mules). Investigators handling Critical alerts will have a 0% false alarm rate.
- **High Risk + Critical Capture Rate**: **62.50%** (10 of 16 mules). By focusing on just 10 accounts ($0.55\%$ of the dataset), investigators capture nearly two-thirds of all active money mules.

```
Risk Band Counts:
Normal       1785
Monitor        22
High Risk       5
Critical        5
```

### B. Detailed Breakdown of Mules in the Monitor Band (Total: 4)
Out of the 22 accounts placed in the **Monitor** band ($30.01 \le \text{Score} \le 60.00$), **4 are actual money mules**. Their scores and component details are:

| Account ID | Fused Risk Score | ML Score (70% wt) | Stat Score (10% wt) | Behavior Score (20% wt) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`A9078`** | **`30.66`** | `0.00` | `8.21` | `99.17` | Recovered False Negative (+10.0 boost applied) |
| **`A9043`** | **`44.82`** | `52.68` | `8.22` | `35.61` | Borderline ML detection |
| **`A9068`** | **`47.53`** | `48.19` | `3.83` | `67.09` | Borderline ML detection |
| **`A9051`** | **`35.89`** | `43.01` | `18.75` | `19.54` | Borderline ML detection |

### C. Detailed Breakdown of Mules in the Normal Band (Total: 2)
Only **2 actual money mule accounts** were missed by the alert system and classified in the **Normal** risk band ($\text{Score} \le 30.00$):

* **Account `A9047`** (Risk Score: **`13.37`**)
  * *Component Scores*: ML = `0.01` | Stat = `22.51` | Behavior = `55.59`
  * *Reason for Miss*: The supervised model assigned a near-zero probability, and neither the general statistical nor the behavioral models detected extreme anomaly signatures.
* **Account `A9076`** (Risk Score: **`2.71`**)
  * *Component Scores*: ML = `1.40` | Stat = `0.00` | Behavior = `8.64`
  * *Reason for Miss*: This account exhibited transaction volume patterns that look completely standard and indistinguishable from normal customer accounts across all features.

### D. False Negative Recovery Case Study: Account `A9078`
Supervised models suffer from blind spots when fraudsters adopt transaction behaviors that resemble normal customer activity. Account `A9078` is an actual mule that the LightGBM classifier assigned a probability of **$0.000025$** (`ml_score` = $0.0025$), making it completely invisible to the primary supervised filter.

However:
- The account exhibited extreme outlier behavior under the LOF model, scoring in the top 1% (`behavior_score` = $99.17$).
- This triggered the $+10.0$ Behavioral Boost.
- Fused Score: $0.70 \times 0.00 + 0.10 \times 8.21 + 0.20 \times 99.17 + 10.0 = 30.66$.
- By scoring $30.66$, the account was successfully escalated from the **Normal** band into the **Monitor** band. This proves the value of the unsupervised backup system in recovering hidden false negatives (33.3% of the classifier's missed mules).

---

## 5. Visualizations

The generated visualizations are stored in the output directory:
1. **`risk_distribution.png`**: Highlights the distribution of the fused risk scores across the test set, showing how normal accounts are heavily concentrated near zero, while risk thresholds successfully isolate anomalous accounts.
2. **`risk_band_counts.png`**: Visualizes the volume of accounts in each operational band, validating the manageable workload for the bank's fraud teams.

---

## 6. Business Recommendations for Bank of India

1. **Tiered Investigation Workflows**:
   - **Critical Alerts**: Automate a temporary freeze on outgoing debits and queue for immediate verification.
   - **High Risk Alerts**: Route to manual investigator queues for resolution within 24 hours.
   - **Monitor Alerts**: Implement automated transaction velocity limits and enhanced ledger logging.
2. **Periodic Calibration**:
   - The `risk_engine.pkl` contains the weights and thresholds. If investigator capacity increases or fraud patterns shift, the weights can be adjusted (e.g. increasing the behavioral weight to capture more anomalies).
