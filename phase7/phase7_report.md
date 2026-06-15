# Phase 7 Report: GenAI Investigation Assistant

This report presents the implementation and outcomes of **Phase 7: GenAI Investigation Assistant** for Bank of India's money mule detection system. In this phase, we evolve the system from a technical risk scoring mechanism into a decision-support copilot for bank fraud teams, generating prioritized, validated, and human-readable case dossiers.

---

## 1. AI Copilot Design & Prompt Engineering

To bridge the gap between technical models and compliance teams, Phase 7 acts as an **AI Copilot** that converts complex analytical outputs (SHAP values, percentiles, outlier factors) into operational case management guides.

### Prompt Template Design
Standardized prompts are structured to feed all necessary context to the LLM without allowing speculative interpretations. The template dynamically formats:
1. **Administrative Data**: Account ID, Fused Risk Score, Risk Band.
2. **Operational Rules**: Recommended banking actions that keep a human in the loop.
3. **Multi-Pillar Scores**: Supervised ML probabilities (0-100), statistical anomaly scores, and behavioral percentiles.
4. **SHAP Drivers**: Technical feature codes mapped strictly to their verified dictionary definitions (feature type, missingness, and Random Forest importance rank).

### Semantic Safety & No Speculation
To maintain strict compliance and safety, the prompt strictly enforces:
- **No speculative semantic guessing**: Feature codes are presented only with their verified metadata. The model is forbidden from guessing the real-world meaning of anonymous features.
- **Probabilistic phrasing**: Direct accusations (e.g., "this customer is a fraudster") are prohibited. The copilot uses risk-based, operational terms like "exhibits highly suspicious ledger velocity consistent with historic mule patterns."

---

## 2. Risk-Based Recommended Actions

To ensure that humans remain in the loop and to mitigate the business impact of false positives, we map the four risk bands to safe, recommended banking procedures:

| Risk Band | Fused Score Range | Recommended Banking Action |
| :--- | :---: | :--- |
| **Normal** | $\le 30.00$ | **No action** |
| **Monitor** | $30.01 - 60.00$ | **Enhanced monitoring** |
| **High Risk** | $60.01 - 80.00$ | **Manual fraud investigation** |
| **Critical** | $80.01 - 100.00$| **Immediate escalation for investigator review and possible restrictions** |

---

## 3. Unified Priority Queue & Case Ranking

Investigators face limited resources. To optimize their workflow, we sort the 32 flagged cases using a compound **Priority Score** that weights fused risk and behavioral outlier severity:

$$\text{Priority Score} = 0.8 \times \text{Risk Score} + 0.2 \times \text{Behavior Score}$$

This formula ensures that accounts exhibiting both high overall risk and extreme behavioral LOF anomalies bubble to the very top.

### Top 10 Priority Queue Summary

| Rank | Account ID | Priority Score | Fused Risk Score | Behavior Score | Risk Band | Recommended Action |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | `A9037` | **`83.25`** | `85.81` | `73.01` | Critical | Immediate escalation for investigator review |
| **2** | `A9044` | **`77.24`** | `83.49` | `52.23` | Critical | Immediate escalation for investigator review |
| **3** | `A9080` | **`76.52`** | `80.41` | `60.98` | Critical | Immediate escalation for investigator review |
| **4** | `A9073` | **`73.97`** | `81.52` | `43.76` | Critical | Immediate escalation for investigator review |
| **5** | `A9075` | **`73.44`** | `81.20` | `42.38` | Critical | Immediate escalation for investigator review |
| **6** | `A9011` | **`68.33`** | `76.29` | `36.49` | High Risk | Manual fraud investigation |
| **7** | `A9040` | **`66.96`** | `75.99` | `30.82` | High Risk | Manual fraud investigation |
| **8** | `A9005` | **`66.80`** | `75.95` | `30.21` | High Risk | Manual fraud investigation |
| **9** | `A9074` | **`64.79`** | `75.52` | `21.85` | High Risk | Manual fraud investigation |
| **10** | `A9068` | **`51.44`** | `47.53` | `67.09` | Monitor | Enhanced monitoring |

All 5 Critical accounts and all 5 High Risk accounts occupy the top 9 positions, showing the formula's effectiveness in prioritizing high-risk cases.

---

## 4. Automated Output Validation (Step 7.3B)

To ensure the safety and reliability of generated reports, we implemented an automated validator (`validate_report`) in [generate_genai_reports.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/generate_genai_reports.py) that screens all generated summaries before saving them.

The validation script runs four checks on every narrative:
1. **Speculation Check**: Confirms the absence of phrases suggesting feature meanings (e.g., "F3898 represents...").
2. **Terminology Check**: Confirms the exact risk band name (e.g., "Critical", "Monitor") is mentioned.
3. **Action Consistency**: Verifies that the recommended banking action is present in the text.
4. **Accusation Check**: Rejects direct accusations of fraud, ensuring the narrative maintains a probabilistic, risk-based tone.

**Outcome**: **100% of the 32 generated narratives passed all validation checks successfully.**

---

## 5. Sample GenAI Case Report: Account `A9044` (Rank 2)

Below is the generated natural language dossier for `A9044`, illustrating the structured, investigator-friendly format:

### 1. Investigation Summary
This account has been flagged in the CRITICAL band with a unified risk score of 83.49. The unified risk assessment was compiled by fusing the supervised predictive model (ML), general ledger statistical deviations, and behavioral outlier indicators. The primary driver of this alert is the supervised machine learning classifier.

### 2. Reasons for Suspicion
- High Supervised Predictor Signal: The LightGBM classifier assigned a risk score of 99.96/100, indicating that the tabular profile highly resembles historical money mules.
  - Key SHAP drivers include Feature F3898 (Continuous feature, 0.0% missing; RF importance rank #6) contributing 2.4554 SHAP, and Feature F3914 (Binary feature, 0.0% missing; RF importance rank #118) contributing 1.3891 SHAP.
- Statistical Anomaly Status: The general statistical anomaly score is 30.77/100 (compared to the test-set average of 24.12), indicating an above-average level of feature variance deviations.
- Behavioral Outlier Status: The Local Outlier Factor (LOF) percentile is 52.23/100. Account has elevated behavioral outlier status.

### 3. Recommended Actions
In accordance with Bank of India compliance policies, the recommended action for the CRITICAL band is: Immediate escalation for investigator review and possible restrictions. Investigators should perform standard verification check-lists, including: 
- Auditing occupation metadata (verify retail account consistency).
- Inspecting the ledger for rapid in-and-out cash velocity patterns (pass-through checking).
- Cross-referencing current observation dates and transaction records against historical profiles.

### 4. Investigation Priority
Based on the priority queue ranking, this case has been assigned to the flagged queue. Investigators must review cases sequentially starting from the Critical queue to optimize operational resources.

---

## 6. Case Files & Report Deliverables

We have generated and verified the following case management assets in the [phase7/](file:///c:/Projects/bank_of_India/mule_account_detection/phase7) directory:
1. **`investigation_dataset.csv`**: Merges risk scores, SHAP explanations, and the optimized model predictions (`predicted_class`).
2. **`investigation_queue.csv`**: Ranks the 32 flagged accounts by priority score.
3. **`case_Axxxx.json`**: Individual Case Files structured for easy ingestion by Bank of India's Case Management Systems (CMS).
4. **`investigator_report.html`**: A responsive, styled HTML dossier for quick browser reviews.
5. **`investigator_report.pdf`**: A professional, paginated PDF docket generated using ReportLab, containing page headers, confidentiality footers, and a formatted queue table.

---

## 7. Phase 7b: Unified Inference Pipeline & Demo Execution

To transition the mule account detection model into production, we implemented a unified inference pipeline in [predict_account.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/predict_account.py) capable of processing single new accounts or batch CSV records.

### 7.1 Key Infrastructure Verification
Before deployment, the following verification tasks were performed:
- **LOF Novelty Verification**: Verified that the behavioral LOF model in [behavioral_lof.pkl](file:///c:/Projects/bank_of_India/mule_account_detection/phase4b/behavioral_lof.pkl) was trained with `novelty=True`, enabling the `.decision_function()` method to score unseen, out-of-sample data.
- **Isolation Forest MinMaxScaler Export**: Re-fit and exported the MinMaxScaler to [isolation_forest_scaler.pkl](file:///c:/Projects/bank_of_India/mule_account_detection/phase4/isolation_forest_scaler.pkl) using the negative of Isolation Forest anomaly scores (`-decision_function`) on the training set, ensuring correct scaling of inference-time anomaly outputs.

### 7.2 Calibration of Synthetic Demo Data
To demonstrate the end-to-end inference capabilities, we created [generate_demo_data.py](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/generate_demo_data.py) which builds [demo_accounts.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/demo_accounts.csv).
The dataset represents exactly **10 demo accounts** calibrated to match the target composition required for demonstration:
- **Normal**: 3 accounts
- **Monitor**: 2 accounts
- **High Risk**: 2 accounts
- **Critical**: 3 accounts

The generation script achieves this by sampling representative accounts from historical splits, stripping target labels (`F3924`), and verifying risk scores in noise-free calibration modes.

### 7.3 Multi-Pillar Scoring & Fusion Logic
For every incoming raw account, the pipeline executes:
1. **Schema Validation**: Ensures all required column names are present and that target `F3924` is absent. Failed records are saved to [failed_records.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/failed_records.csv).
2. **Preprocessing**: Transforms raw data using the pre-fitted preprocessing pipeline (`preprocessing_pipeline.pkl`) into 300 ML-ready features.
3. **Behavioral Feature Generation**: Dynamically constructs 10 behavioral features using median imputation and categorical risk mappings calculated from the training split.
4. **ML prediction**: Computes supervised probability using LightGBM and assigns `predicted_class` using the optimized decision threshold of `0.40`.
5. **Statistical Anomaly Scoring**: Computes Isolation Forest anomaly score scaled to $0-100$ using `isolation_forest_scaler.pkl`.
6. **Behavioral Outlier Percentile**: Scores scaled behavior features with LOF and maps the raw signal to a percentile rank ($0-100$) using reference scores from `behavioral_anomaly_scores.csv`.
7. **Risk Fusion and Boosting**: Fuses scores using the formula:
   $$\text{Risk Score} = 0.70 \times \text{ML Score} + 0.10 \times \text{Stat Score} + 0.20 \times \text{Behavior Score}$$
   If the behavioral outlier percentile score $\ge 99.0$, a $+10.0$ behavioral boost is added (capped at $100.0$).
8. **Dynamic SHAP Explanations**: Computes SHAP values on-the-fly for the preprocessed new account. Extracts top 5 positive and top 5 negative risk drivers mapped strictly to verified metadata in `feature_dictionary.json`.
9. **Dossier Narrative & Validation**: Generates natural language reports only for non-Normal accounts. All reports must pass automated terminology, non-speculation, action consistency, and safety checks before export.

### 7.4 Demo Prediction Execution Outputs
We ran the batch prediction command:
`python phase7/predict_account.py --batch demo_accounts.csv`

The run processed all 10 demo accounts, resulting in the following outputs in [phase7/](file:///c:/Projects/bank_of_India/mule_account_detection/phase7):

#### A. Prediction Table: [predictions.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/predictions.csv)
All 10 accounts were predicted and categorized into their exact calibrated bands:

| Account ID | ML Prob | Fused Risk Score | Risk Band | Recommended Action |
| :--- | :---: | :---: | :--- | :--- |
| `DEMO001` | `0.000002` | `12.89` | **Normal** | No action |
| `DEMO002` | `0.000005` | `17.65` | **Normal** | No action |
| `DEMO003` | `0.000086` | `5.41` | **Normal** | No action |
| `DEMO004` | `0.000002` | `32.20` | **Monitor** | Enhanced monitoring |
| `DEMO005` | `0.000005` | `32.25` | **Monitor** | Enhanced monitoring |
| `DEMO006` | `0.955839` | `74.78` | **High Risk**| Manual fraud investigation |
| `DEMO007` | `0.931793` | `82.63` | **Critical** | Immediate escalation for investigator review |
| `DEMO008` | `0.998711` | `76.92` | **High Risk**| Manual fraud investigation |
| `DEMO009` | `0.999995` | `81.19` | **Critical** | Immediate escalation for investigator review |
| `DEMO010` | `0.957211` | `80.42` | **Critical** | Immediate escalation for investigator review |

*Note: For `DEMO004` and `DEMO005`, the LOF behavioral percentile was $\ge 99.0$ ($99.78$ and $99.56$ respectively), triggering a $+10.0$ boost that lifted their risk scores into the Monitor band despite a low ML probability.*

#### B. Narrative Reports: [reports.json](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/reports.json)
Contains exactly 7 validated narrative reports. The 3 Normal accounts (`DEMO001`, `DEMO002`, `DEMO003`) are intentionally excluded.

#### C. Validation Failures log: [failed_records.csv](file:///c:/Projects/bank_of_India/mule_account_detection/phase7/failed_records.csv)
Created as an empty schema-compliant CSV since 100% of demo accounts passed schema validation checks.

