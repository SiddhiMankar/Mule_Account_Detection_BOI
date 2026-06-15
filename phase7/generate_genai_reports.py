"""
generate_genai_reports.py
-------------------------
Phase 7: GenAI Investigation Assistant
Bank of India -- Mule Account Detection

This script:
1. Merges Phase 5 risk scores, Phase 6 top features, and feature dictionary.
2. Builds 'investigation_dataset.csv' with predicted_class = int(ml_probability >= 0.40).
3. Connects to the Gemini API using google-generativeai and a dynamic model environment variable.
4. Generates human-readable investigator reports with a fallback to structured rule-based templates.
5. Implements Step 7.3B LLM Output Validation (checks for hallucinations, direct accusations, formatting).
6. Maps risk bands to safe recommended actions.
7. Calculates compound Priority Score (0.8 * Risk Score + 0.2 * Behavior Score) and exports 'investigation_queue.csv'.
8. Generates case JSON files for all 32 flagged accounts.
9. Exports Markdown, HTML, and ReportLab PDF investigator dossiers.
10. Validates consistency end-to-end.
"""

import os
import sys
import json
import csv
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE5_DIR = os.path.join(BASE_DIR, "phase5")
PHASE6_DIR = os.path.join(BASE_DIR, "phase6")
PHASE7_DIR = os.path.join(BASE_DIR, "phase7")
os.makedirs(PHASE7_DIR, exist_ok=True)

print("=" * 60)
print("Phase 7: GenAI Investigation Assistant")
print("=" * 60)

# =====================================================================
# STEP 7.1 — Build Unified Investigation Dataset
# =====================================================================
print("\n--- Step 7.1: Building Unified Investigation Dataset ---")

# Load Phase 5 Risk Scores
risk_scores_path = os.path.join(PHASE5_DIR, "risk_scores.csv")
if not os.path.exists(risk_scores_path):
    print(f"Error: {risk_scores_path} not found.")
    sys.exit(1)
df_risk = pd.read_csv(risk_scores_path)

# Load Phase 6 Top Features
top_feat_path = os.path.join(PHASE6_DIR, "top_features_per_account.csv")
if not os.path.exists(top_feat_path):
    print(f"Error: {top_feat_path} not found.")
    sys.exit(1)
df_top_feat = pd.read_csv(top_feat_path)

# Load Phase 6 Feature Dictionary
feat_dict_path = os.path.join(PHASE6_DIR, "feature_dictionary.json")
if not os.path.exists(feat_dict_path):
    print(f"Error: {feat_dict_path} not found.")
    sys.exit(1)
with open(feat_dict_path, "r", encoding="utf-8") as f:
    feature_dictionary = json.load(f)

# Pivot top features to wide format (we need top 3 features per account)
top_pivot = {}
for name, group in df_top_feat.groupby("account_id"):
    sorted_group = group.sort_values("rank")
    feats = sorted_group["feature"].tolist()
    shaps = sorted_group["shap_value"].tolist()
    vals = sorted_group["feature_value"].tolist()
    dirs = sorted_group["direction"].tolist()
    
    # Pad to make sure we have at least 3 features
    while len(feats) < 3:
        feats.append("None")
        shaps.append(0.0)
        vals.append(0.0)
        dirs.append("+")
        
    explanation_parts = []
    for i in range(3):
        if feats[i] != "None":
            explanation_parts.append(f"{feats[i]} ({dirs[i]}{abs(shaps[i]):.4f} SHAP)")
            
    top_pivot[name] = {
        "top_feature_1": feats[0],
        "top_feature_2": feats[1],
        "top_feature_3": feats[2],
        "explanation": ", ".join(explanation_parts)
    }

# Build unified dataset
inv_dataset_rows = []
for idx, row in df_risk.iterrows():
    acct = row["account_id"]
    pivot_data = top_pivot.get(acct, {
        "top_feature_1": "None",
        "top_feature_2": "None",
        "top_feature_3": "None",
        "explanation": "None"
    })
    
    # ml_probability is [0,1], decision threshold is 0.40
    # ml_score is [0,100] (ml_probability * 100)
    # predicted_class = 1 if ml_probability >= 0.40 else 0 (which is identical to ml_score >= 40.0)
    predicted_class = int(row["ml_probability"] >= 0.40)
    
    inv_dataset_rows.append({
        "account_id": acct,
        "ml_score": round(row["ml_score"], 2),
        "stat_score": round(row["stat_score"], 2),
        "behavior_score": round(row["behavior_score"], 2),
        "final_risk_score": round(row["risk_score"], 2),
        "risk_band": row["risk_band"],
        "top_feature_1": pivot_data["top_feature_1"],
        "top_feature_2": pivot_data["top_feature_2"],
        "top_feature_3": pivot_data["top_feature_3"],
        "explanation": pivot_data["explanation"],
        "predicted_class": predicted_class
    })

df_inv = pd.DataFrame(inv_dataset_rows)
inv_csv_path = os.path.join(PHASE7_DIR, "investigation_dataset.csv")
df_inv.to_csv(inv_csv_path, index=False)
print(f"  Unified investigation dataset saved to: {inv_csv_path}")
print(f"  Total rows: {len(df_inv)} accounts. Columns: {list(df_inv.columns)}")

# =====================================================================
# STEP 7.2 & 7.4 — Prompt Templates & Recommended Actions
# =====================================================================
print("\n--- Step 7.2: Designing Prompts & Step 7.4: Banking Action Rules ---")

# Define safe, keeps-human-in-the-loop actions
RECOMMENDED_ACTIONS = {
    "Normal": "No action",
    "Monitor": "Enhanced monitoring",
    "High Risk": "Manual fraud investigation",
    "Critical": "Immediate escalation for investigator review and possible restrictions"
}

PROMPT_TEMPLATE = """You are a senior bank fraud analyst generating an operational case management dossier for a fraud investigator.

Account Information:
- Account ID: {account_id}
- Fused Risk Score: {risk_score}
- Risk Band: {risk_band}
- Recommended Banking Action: {recommended_action}

3-Pillar Score Breakdown:
- Supervised ML Score (70% weight): {ml_score} / 100
- Statistical Anomaly Score (10% weight): {stat_score} / 100 (Test set average: 24.12)
- Behavioral Anomaly Score (20% weight): {behavior_score} / 100 {boost_notes}

Key Technical Indicators (SHAP Drivers):
{top_features_desc}

Based strictly on this data, write a professional case investigation narrative with these four sections:
1. Investigation Summary
2. Reasons for Suspicion
3. Recommended Actions
4. Investigation Priority

CRITICAL INSTRUCTIONS:
- You MUST present F-codes and metrics using metadata from the verified feature dictionary only.
- DO NOT introduce any semantic guesses or speculation about what anonymous F-codes mean (e.g. do not say "F3898 represents withdrawal amount"). Explain them strictly by their feature types, missingness, and importance ranks.
- DO NOT make direct, absolute accusations of fraud. Always use professional, probabilistic language like "exhibits highly suspicious patterns resembling known historical money mules", "consistent with suspicious pass-through activity", or "anomalous ledger profile".
- Ensure the recommended action matches: "{recommended_action}"
- The output must be clear, structured, and operational for banking teams.
"""

# =====================================================================
# STEP 7.3 & 7.3B — Generate Natural Language Reports & Validation
# =====================================================================
print("\n--- Step 7.3 & 7.3B: Generating Narratives & Automated Validation ---")

# Try to initialize google-generativeai SDK
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

has_gemini = False
if api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        print(f"  Successfully imported and configured google-generativeai with model '{GENAI_MODEL}'")
        has_gemini = True
    except Exception as e:
        print(f"  Warning: Could not configure google-generativeai ({e}). Using rule-based fallback.")
else:
    print("  No GEMINI_API_KEY or GOOGLE_API_KEY found in environment. Using rule-based fallback.")

# High-quality, verified rule-based generator for fallback and validation assurance
def generate_local_copilot_report(acct, risk_score, band, ml_score, stat_score, behavior_score, boost_applied, top_features):
    action = RECOMMENDED_ACTIONS.get(band, "No action")
    
    # Feature dictionary text mapping
    feat_lines = []
    for tf in top_features:
        feat_lines.append(f"- Feature {tf['feature']} ({tf['label']}): value={tf['feature_value']}, SHAP={tf['shap_value']}")
    
    # 3-pillar breakdown details for the text
    boost_desc = "Account is in the top 1% of behavioral outliers on the ledger (with +10.0 boost applied)." if boost_applied else "Account has elevated behavioral outlier status."
    
    # Construct Narrative Sections
    summary_section = (
        f"This account has been flagged in the {band.upper()} band with a unified risk score of {risk_score:.2f}. "
        f"The unified risk assessment was compiled by fusing the supervised predictive model (ML), general ledger statistical deviations, and behavioral outlier indicators. "
        f"The primary driver of this alert is the { 'supervised machine learning classifier' if ml_score > 40 else 'unsupervised behavioral anomaly detector' }."
    )
    
    suspicion_parts = []
    if ml_score >= 40.0:
        suspicion_parts.append(
            f"- High Supervised Predictor Signal: The LightGBM classifier assigned a risk score of {ml_score:.2f}/100, indicating that the tabular profile highly resembles historical money mules."
        )
        suspicion_parts.append(
            f"  - Key SHAP drivers include {top_features[0]['feature']} (RF importance rank #{top_features[0]['label'].split('RF importance rank #')[-1]}) contributing {top_features[0]['shap_value']:.4f} SHAP, "
            f"and {top_features[1]['feature']} (RF importance rank #{top_features[1]['label'].split('RF importance rank #')[-1]}) contributing {top_features[1]['shap_value']:.4f} SHAP."
        )
    else:
        suspicion_parts.append(
            f"- Unsupervised Anomaly Highlight: The supervised classifier assigned a low probability ({ml_score:.2f}/100), but the unsupervised detectors flagged extreme outliers. "
            f"This profile represents a potential novel or sophisticated evasion pattern not covered by historical training data."
        )
        
    suspicion_parts.append(
        f"- Statistical Anomaly Status: The general statistical anomaly score is {stat_score:.2f}/100 (compared to the test-set average of 24.12), "
        f"indicating {'an above-average' if stat_score > 24.12 else 'a below-average'} level of feature variance deviations."
    )
    
    suspicion_parts.append(
        f"- Behavioral Outlier Status: The Local Outlier Factor (LOF) percentile is {behavior_score:.2f}/100. {boost_desc}"
    )
    
    suspicion_section = "\n".join(suspicion_parts)
    
    action_section = (
        f"In accordance with Bank of India compliance policies, the recommended action for the {band.upper()} band is: "
        f"{action}. Investigators should perform standard verification check-lists, including: \n"
        f"- Auditing occupation metadata ({'verify student or agricultural status' if 'student' in str(top_features) or 'savings' in str(top_features) else 'verify retail account consistency'}).\n"
        f"- Inspecting the ledger for rapid in-and-out cash velocity patterns (pass-through checking).\n"
        f"- Cross-referencing current observation dates and transaction records against historical profiles."
    )
    
    priority_section = (
        f"Based on the priority queue ranking, this case has been assigned to the flagged queue. "
        f"Investigators must review cases sequentially starting from the Critical queue to optimize operational resources."
    )
    
    narrative = f"""### 1. Investigation Summary
{summary_section}

### 2. Reasons for Suspicion
{suspicion_section}

### 3. Recommended Actions
{action_section}

### 4. Investigation Priority
{priority_section}"""
    return narrative

# LLM Output Validator to ensure safety and reproducibility
def validate_report(report_text, band, action):
    # Rule 1: No semantic guessing of F-codes (e.g. Fxxx means...)
    import re
    guesses = re.findall(r"F\d{3,4}\s+means|F\d{3,4}\s+represents|F\d{3,4}\s+refers\s+to", report_text, re.IGNORECASE)
    if guesses:
        print(f"    [Validation Failed]: Speculative F-code definition found: {guesses}")
        return False
        
    # Rule 2: Risk band name must be present in text
    if band.lower() not in report_text.lower():
        print(f"    [Validation Failed]: Risk band '{band}' not found in the text.")
        return False
        
    # Rule 3: Recommended action must be present in text
    # Clean up punctuation and spacing for robust checking
    clean_action = re.sub(r'[^\w\s]', '', action.lower())
    clean_report = re.sub(r'[^\w\s]', '', report_text.lower())
    
    # Check if a substantial part of the action is in the report
    action_words = clean_action.split()
    matched_words = sum(1 for w in action_words if w in clean_report)
    if matched_words < min(3, len(action_words)):
        print(f"    [Validation Failed]: Recommended action '{action}' not found in the text.")
        return False
        
    # Rule 4: No direct fraud accusations (must use probabilistic language)
    accusations = ["is a fraudster", "is a criminal", "guilty of fraud", "committed fraud", "definitely a mule"]
    for acc in accusations:
        if acc in report_text.lower():
            print(f"    [Validation Failed]: Direct accusation found: '{acc}'")
            return False
            
    return True

# Load investigation cards (to extract the same account selections and structures)
with open(os.path.join(PHASE6_DIR, "investigation_cards.json"), "r", encoding="utf-8") as f:
    cards_data = json.load(f)

flagged_accounts = cards_data["flagged_accounts"]
print(f"  Loaded {len(flagged_accounts)} flagged accounts to process.")

genai_reports = {}

for acct_id, card in flagged_accounts.items():
    risk_score = card["risk_score"]
    band = card["risk_band"]
    ml_score = card["ml_score"]
    stat_score = card["stat_score"]
    behavior_score = card["behavior_score"]
    boost_applied = card["boost_applied"]
    top_features = card["top_shap_contributors"]
    action = RECOMMENDED_ACTIONS.get(band, "No action")
    
    # 1. Format the prompt template
    feat_desc_parts = []
    for tf in top_features[:3]:
        feat_desc_parts.append(
            f"- Feature {tf['feature']} ({tf['label']}): value={tf['feature_value']}, SHAP={tf['shap_value']}"
        )
    top_features_desc = "\n".join(feat_desc_parts)
    
    boost_notes = "(LOF percentile >= 99.0, +10.0 boost applied)" if boost_applied else "(No boost applied)"
    
    prompt = PROMPT_TEMPLATE.format(
        account_id=acct_id,
        risk_score=risk_score,
        risk_band=band,
        recommended_action=action,
        ml_score=ml_score,
        stat_score=stat_score,
        behavior_score=behavior_score,
        boost_notes=boost_notes,
        top_features_desc=top_features_desc
    )
    
    report_text = ""
    # 2. Try to call Gemini API if configured
    if has_gemini:
        try:
            model = genai.GenerativeModel(GENAI_MODEL)
            response = model.generate_content(prompt)
            candidate_text = response.text
            
            # 3. Validate GenAI Output (Step 7.3B)
            if validate_report(candidate_text, band, action):
                report_text = candidate_text
                print(f"  ✅ Account {acct_id}: Generated via Gemini API & successfully validated.")
            else:
                print(f"  ⚠️ Account {acct_id}: Gemini API output failed validation. Falling back to local template.")
        except Exception as api_err:
            print(f"  ⚠️ Account {acct_id}: Gemini API call failed ({api_err}). Falling back to local template.")
            
    # 4. Fallback to local high-quality rule-based Copilot report if needed
    if not report_text:
        report_text = generate_local_copilot_report(
            acct=acct_id,
            risk_score=risk_score,
            band=band,
            ml_score=ml_score,
            stat_score=stat_score,
            behavior_score=behavior_score,
            boost_applied=boost_applied,
            top_features=top_features
        )
        # Verify fallback itself is valid
        assert validate_report(report_text, band, action), f"Fallback report validation failed for {acct_id}"
        print(f"  ✅ Account {acct_id}: Generated via local rule-based template.")
        
    genai_reports[acct_id] = {
        "account_id": acct_id,
        "prompt": prompt,
        "report": report_text
    }

# Save GenAI Reports JSON
reports_json_path = os.path.join(PHASE7_DIR, "genai_reports.json")
with open(reports_json_path, "w", encoding="utf-8") as f:
    json.dump(genai_reports, f, indent=2, ensure_ascii=False)
print(f"  Saved all generated Copilot reports to: {reports_json_path}")

# =====================================================================
# STEP 7.5 — Generate Priority Queue
# =====================================================================
print("\n--- Step 7.5: Calculating Priority Scores & Ranking Queue ---")

# Formula: Priority Score = 0.8 * Risk Score + 0.2 * Behavior Score
priority_rows = []
for acct_id, card in flagged_accounts.items():
    risk_score = card["risk_score"]
    behavior_score = card["behavior_score"]
    priority_score = round(0.8 * risk_score + 0.2 * behavior_score, 2)
    
    priority_rows.append({
        "account_id": acct_id,
        "priority_score": priority_score,
        "risk_score": risk_score,
        "risk_band": card["risk_band"],
        "recommended_action": RECOMMENDED_ACTIONS[card["risk_band"]]
    })

df_priority = pd.DataFrame(priority_rows)
# Sort descending by priority score, then by risk score
df_priority = df_priority.sort_values(by=["priority_score", "risk_score"], ascending=False).reset_index(drop=True)
df_priority.index += 1  # 1-based ranking index
df_priority.index.name = "rank"
df_priority = df_priority.reset_index()

queue_csv_path = os.path.join(PHASE7_DIR, "investigation_queue.csv")
df_priority.to_csv(queue_csv_path, index=False)
print(f"  Priority queue ranked and saved to: {queue_csv_path}")

# Print Top 10 Priority Queue Accounts
print("  Top 10 Queue Rank:")
print(df_priority.head(10)[["rank", "account_id", "priority_score", "risk_score", "risk_band"]])

# Create a lookup mapping for rank and priority score
priority_lookup = {}
for _, r in df_priority.iterrows():
    priority_lookup[r["account_id"]] = {
        "rank": int(r["rank"]),
        "priority_score": float(r["priority_score"])
    }

# =====================================================================
# STEP 7.6 — Generate Case Files
# =====================================================================
print("\n--- Step 7.6: Exporting Standardized JSON Case Files ---")

for acct_id, card in flagged_accounts.items():
    p_info = priority_lookup[acct_id]
    copilot_info = genai_reports[acct_id]
    
    case_file = {
        "account_id": acct_id,
        "risk_score": card["risk_score"],
        "risk_band": card["risk_band"],
        "top_features": [
            {
                "feature": tf["feature"],
                "label": tf["label"],
                "value": tf["feature_value"],
                "shap_value": tf["shap_value"]
            } for tf in card["top_shap_contributors"]
        ],
        "shap_explanation": card["narrative"].split("\n\nStatistical Anomaly")[0], # technical segment
        "genai_summary": copilot_info["report"], # human understandable copilot text
        "recommended_action": RECOMMENDED_ACTIONS[card["risk_band"]],
        "priority_score": p_info["priority_score"],
        "priority_rank": p_info["rank"]
    }
    
    case_path = os.path.join(PHASE7_DIR, f"case_{acct_id}.json")
    with open(case_path, "w", encoding="utf-8") as f:
        json.dump(case_file, f, indent=2, ensure_ascii=False)
        
print(f"  Successfully exported {len(flagged_accounts)} individual JSON case files.")

# =====================================================================
# STEP 7.7 — Export Human-Readable Reports (HTML, MD, PDF)
# =====================================================================
print("\n--- Step 7.7: Exporting Markdown, HTML, and PDF Dossiers ---")

# 1. Markdown Report
md_report_path = os.path.join(PHASE7_DIR, "investigator_report.md")
with open(md_report_path, "w", encoding="utf-8") as f:
    f.write("# Bank of India — Unified Money Mule Investigation Dossier\n")
    f.write(f"Generated: 2026-06-14 | Total Flagged Cases: {len(df_priority)}\n\n")
    f.write("## 1. Unified Priority Investigation Queue\n\n")
    f.write("| Rank | Account ID | Priority Score | Risk Score | Risk Band | Recommended Action |\n")
    f.write("|---|---|---|---|---|---|\n")
    for _, r in df_priority.iterrows():
        f.write(f"| {r['rank']} | **`{r['account_id']}`** | {r['priority_score']} | {r['risk_score']} | **{r['risk_band']}** | {r['recommended_action']} |\n")
        
    f.write("\n---\n\n## 2. Detailed Case Files\n\n")
    for _, r in df_priority.iterrows():
        acct_id = r["account_id"]
        copilot_info = genai_reports[acct_id]
        card = flagged_accounts[acct_id]
        
        f.write(f"### Rank {r['rank']}: Account {acct_id} ({r['risk_band']} Band)\n\n")
        f.write(f"- **Priority Score**: {r['priority_score']} / 100\n")
        f.write(f"- **Fused Risk Score**: {r['risk_score']} / 100\n")
        f.write(f"- **ML Score**: {card['ml_score']} | **Statistical Score**: {card['stat_score']} | **Behavioral Score**: {card['behavior_score']}\n")
        f.write(f"- **Recommended Action**: **{r['recommended_action']}**\n\n")
        f.write("#### AI Copilot Case Report:\n")
        # Format the text with indents for clean md rendering
        indented_report = "\n".join("  " + line for line in copilot_info["report"].split("\n"))
        f.write(f"{indented_report}\n\n")
        f.write("---\n\n")
        
print(f"  Markdown investigator report saved to: {md_report_path}")

# 2. HTML Dossier
html_report_path = os.path.join(PHASE7_DIR, "investigator_report.html")
html_content = ["""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bank of India — Investigation Dossier</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #F8FAFC; color: #1E293B; margin: 0; padding: 40px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
        h1 { color: #1E3A8A; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; margin-top: 0; }
        h2 { color: #1E3A8A; margin-top: 30px; border-bottom: 1px solid #E2E8F0; padding-bottom: 5px; }
        h3 { color: #2563EB; margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 25px; }
        th { background-color: #1E3A8A; color: white; text-align: left; padding: 12px; font-weight: 600; }
        td { padding: 12px; border-bottom: 1px solid #E2E8F0; }
        tr:hover { background-color: #F1F5F9; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .badge-Critical { background-color: #FEE2E2; color: #991B1B; }
        .badge-High { background-color: #FFEDD5; color: #9A3412; }
        .badge-Monitor { background-color: #FEF9C3; color: #854D0E; }
        .card { border: 1px solid #E2E8F0; border-radius: 8px; padding: 25px; margin-bottom: 30px; background: #FFFFFF; }
        .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F5F9; padding-bottom: 10px; margin-bottom: 15px; }
        .meta-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; background: #F8FAFC; padding: 15px; border-radius: 6px; }
        .meta-item { font-size: 14px; }
        .meta-label { font-weight: bold; color: #64748B; display: block; margin-bottom: 2px; }
        .meta-val { font-size: 16px; font-weight: 600; color: #0F172A; }
        .report-text { white-space: pre-wrap; font-size: 15px; line-height: 1.6; color: #334155; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bank of India — Unified Money Mule Investigation Dossier</h1>
        <p><strong>Generated Date:</strong> 2026-06-14 | <strong>Total Cases for Review:</strong> """ + str(len(df_priority)) + """</p>
        
        <h2>1. Priority Queue Overview</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Account ID</th>
                    <th>Priority Score</th>
                    <th>Risk Score</th>
                    <th>Risk Band</th>
                    <th>Recommended Action</th>
                </tr>
            </thead>
            <tbody>"""]

for _, r in df_priority.iterrows():
    acct_id = r["account_id"]
    badge_band = r["risk_band"].replace(" Risk", "") # High Risk -> High
    html_content.append(f"""
                <tr>
                    <td><strong>{r['rank']}</strong></td>
                    <td><code>{acct_id}</code></td>
                    <td>{r['priority_score']:.2f}</td>
                    <td>{r['risk_score']:.2f}</td>
                    <td><span class="badge badge-{badge_band}">{r['risk_band']}</span></td>
                    <td>{r['recommended_action']}</td>
                </tr>""")

html_content.append("""
            </tbody>
        </table>
        
        <h2>2. Detailed Case Reports</h2>""")

for _, r in df_priority.iterrows():
    acct_id = r["account_id"]
    copilot_info = genai_reports[acct_id]
    card = flagged_accounts[acct_id]
    badge_band = r["risk_band"].replace(" Risk", "")
    
    # Format narratives for neat display
    formatted_report = copilot_info["report"].replace("\n", "<br>").replace("### ", "<h4>").replace("</h4><br>", "</h4>")
    
    html_content.append(f"""
        <div class="card">
            <div class="card-header">
                <h3>Rank {r['rank']}: Account {acct_id}</h3>
                <span class="badge badge-{badge_band}">{r['risk_band']}</span>
            </div>
            <div class="meta-grid">
                <div class="meta-item">
                    <span class="meta-label">Priority Score</span>
                    <span class="meta-val">{r['priority_score']:.2f} / 100</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Fused Risk Score</span>
                    <span class="meta-val">{r['risk_score']:.2f} / 100</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">ML / Stat / Behavior</span>
                    <span class="meta-val">{card['ml_score']:.1f} / {card['stat_score']:.1f} / {card['behavior_score']:.1f}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Recommended Action</span>
                    <span class="meta-val" style="font-size:13px; color:#1E3A8A;">{r['recommended_action']}</span>
                </div>
            </div>
            <div class="report-text">
                {formatted_report}
            </div>
        </div>""")

html_content.append("""
    </div>
</body>
</html>""")

with open(html_report_path, "w", encoding="utf-8") as f:
    f.writelines(html_content)
print(f"  HTML investigator report saved to: {html_report_path}")

# 3. PDF Docket (using ReportLab)
pdf_report_path = os.path.join(PHASE7_DIR, "investigator_report.pdf")

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    
    # Custom Canvas to handle professional header/footer & dynamic page counts
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []
            
        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()
            
        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_number(num_pages)
                super().showPage()
            super().save()
            
        def draw_page_number(self, page_count):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            
            # Header (pages after the first page)
            if self._pageNumber > 1:
                self.drawString(54, 750, "Bank of India — Money Mule Investigation Docket")
                self.setStrokeColor(colors.HexColor("#E2E8F0"))
                self.setLineWidth(0.5)
                self.line(54, 742, 558, 742)
                
            # Footer (all pages)
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 55, 558, 55)
            
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 40, page_text)
            self.drawString(54, 40, "CONFIDENTIAL — FOR INTERNAL AUDIT & LAW ENFORCEMENT USE ONLY")
            self.restoreState()

    # Setup document
    doc = SimpleDocTemplate(
        pdf_report_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles to avoid collisions and format text nicely
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )
    
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )
    
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#64748B')
    )
    
    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#0F172A')
    )

    story = []
    
    # Cover / Header Title
    story.append(Paragraph("Bank of India — Unified Money Mule Investigation Docket", title_style))
    story.append(Paragraph(f"<b>Generated Date:</b> 2026-06-14 &nbsp;|&nbsp; <b>Total Cases Flagged:</b> {len(df_priority)}", body_style))
    story.append(Spacer(1, 15))
    
    # 1. Table of Priority Queue
    story.append(Paragraph("1. Unified Priority Investigation Queue", h1_style))
    
    table_data = [[
        Paragraph("Rank", table_header),
        Paragraph("Account ID", table_header),
        Paragraph("Priority Score", table_header),
        Paragraph("Risk Score", table_header),
        Paragraph("Risk Band", table_header),
        Paragraph("Recommended Action", table_header)
    ]]
    
    for _, r in df_priority.iterrows():
        table_data.append([
            Paragraph(f"<b>{r['rank']}</b>", table_text),
            Paragraph(f"<code>{r['account_id']}</code>", table_text),
            Paragraph(f"{r['priority_score']:.2f}", table_text),
            Paragraph(f"{r['risk_score']:.2f}", table_text),
            Paragraph(f"<b>{r['risk_band']}</b>", table_text),
            Paragraph(r["recommended_action"], table_text)
        ])
        
    queue_table = Table(table_data, colWidths=[35, 65, 75, 60, 70, 199])
    queue_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ('TOPPADDING', (0,1), (-1,-1), 6),
    ]))
    
    story.append(queue_table)
    story.append(Spacer(1, 20))
    story.append(PageBreak())
    
    # 2. Detailed Case Reports
    story.append(Paragraph("2. Detailed Case Reports", h1_style))
    
    for _, r in df_priority.iterrows():
        acct_id = r["account_id"]
        copilot_info = genai_reports[acct_id]
        card = flagged_accounts[acct_id]
        
        # Meta Grid Table for the case header
        meta_data = [
            [
                Paragraph("PRIORITY SCORE", meta_label_style),
                Paragraph("FUSED RISK SCORE", meta_label_style),
                Paragraph("ML / STAT / BEHAVIOR", meta_label_style),
                Paragraph("RECOMMENDED ACTION", meta_label_style)
            ],
            [
                Paragraph(f"{r['priority_score']:.2f} / 100", meta_val_style),
                Paragraph(f"{r['risk_score']:.2f} / 100", meta_val_style),
                Paragraph(f"{card['ml_score']:.1f} / {card['stat_score']:.1f} / {card['behavior_score']:.1f}", meta_val_style),
                Paragraph(r["recommended_action"], meta_val_style)
            ]
        ]
        
        meta_table = Table(meta_data, colWidths=[90, 100, 110, 204])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        
        case_flowables = [
            Paragraph(f"Rank {r['rank']}: Account {acct_id} ({r['risk_band']})", h2_style),
            meta_table,
            Spacer(1, 8)
        ]
        
        # Parse the narrative report text
        raw_text = copilot_info["report"]
        paragraphs = raw_text.split("\n\n")
        for p in paragraphs:
            if p.startswith("### "):
                header_text = p.replace("### ", "")
                case_flowables.append(Paragraph(header_text, h2_style))
            else:
                case_flowables.append(Paragraph(p, body_style))
                
        case_flowables.append(Spacer(1, 15))
        
        # Group each case file so it is kept together where possible or split nicely
        story.append(KeepTogether(case_flowables))
        story.append(Spacer(1, 10))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  PDF investigator report saved to: {pdf_report_path}")
    
except Exception as pdf_err:
    print(f"  Error generating PDF report ({pdf_err}). Make sure reportlab is fully configured.")

# =====================================================================
# STEP 7.8 — Evaluate GenAI Consistency
# =====================================================================
print("\n--- Step 7.8: Evaluating GenAI Output Consistency ---")

consistency_passed = True
total_audited = 0

for acct_id, copilot_info in genai_reports.items():
    card = flagged_accounts[acct_id]
    action = RECOMMENDED_ACTIONS[card["risk_band"]]
    
    total_audited += 1
    # Run the validation check
    valid = validate_report(copilot_info["report"], card["risk_band"], action)
    if not valid:
        consistency_passed = False
        print(f"  ❌ Account {acct_id}: Consistency evaluation failed!")
        
if consistency_passed:
    print(f"  ✅ All {total_audited} generated summaries passed consistency checks successfully.")
else:
    print("  ⚠️ Some generated summaries failed consistency checks. Please review log warnings.")

print("\n" + "=" * 60)
print("Phase 7 — Deliverables Summary:")
print("=" * 60)
print(f"  investigation_dataset.csv  : Unified dataset ({len(df_inv)} rows)")
print(f"  investigation_queue.csv    : Ranked priority queue ({len(df_priority)} rows)")
print(f"  genai_reports.json         : {len(genai_reports)} case narrative reports")
print(f"  case_Axxxx.json files      : {len(flagged_accounts)} standardized case files")
print(f"  investigator_report.md     : Markdown report")
print(f"  investigator_report.html   : Sleek HTML report")
print(f"  investigator_report.pdf    : Professional PDF docket (via ReportLab)")
print("=" * 60)
print("Phase 7 generate_genai_reports.py completed successfully.")
print("=" * 60)
