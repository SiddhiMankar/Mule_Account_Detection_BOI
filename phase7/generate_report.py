"""
generate_report.py
------------------
Step 7B.10: Report Generation with Gemini API and local fallback.
"""

import os
import sys

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
### 1. Investigation Summary
### 2. Reasons for Suspicion
### 3. Recommended Actions
### 4. Investigation Priority

CRITICAL INSTRUCTIONS:
- You MUST present F-codes and metrics using metadata from the verified feature dictionary only.
- DO NOT introduce any semantic guesses or speculation about what anonymous F-codes mean (e.g. do not say "F3898 represents withdrawal amount"). Explain them strictly by their feature types, missingness, and importance ranks.
- DO NOT make direct, absolute accusations of fraud. Always use professional, probabilistic language like "exhibits highly suspicious patterns resembling known historical money mules", "consistent with suspicious pass-through activity", or "anomalous ledger profile".
- Ensure the recommended action matches: "{recommended_action}"
- The output must be clear, structured, and operational for banking teams.
"""

def generate_local_copilot_report(acct_id, risk_score, band, ml_score, stat_score, behavior_score, boost_applied, top_features):
    action = RECOMMENDED_ACTIONS.get(band, "No action")
    
    # Construct Narrative Sections
    boost_desc = "Account is in the top 1% of behavioral outliers on the ledger (with +10.0 boost applied)." if boost_applied else "Account has elevated behavioral outlier status."
    
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
        if len(top_features) >= 2:
            suspicion_parts.append(
                f"  - Key SHAP drivers include Feature {top_features[0]['feature']} ({top_features[0]['label']}) contributing {top_features[0]['shap_value']:.4f} SHAP, "
                f"and Feature {top_features[1]['feature']} ({top_features[1]['label']}) contributing {top_features[1]['shap_value']:.4f} SHAP."
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
        f"- Auditing occupation metadata (verify retail account consistency).\n"
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

def generate_report(acct_id, risk_score, band, ml_score, stat_score, behavior_score, boost_applied, top_features, api_key=None):
    """
    Generates a natural language report using Gemini API or local fallback.
    """
    action = RECOMMENDED_ACTIONS.get(band, "No action")
    
    # If Normal, no narrative report is generated
    if band == "Normal":
        return "No significant mule-account risk indicators detected."
        
    # Format the top features description
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
    GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
    
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(GENAI_MODEL)
            response = model.generate_content(prompt)
            candidate = response.text
            
            # Import validator
            from phase7.validate_report import validate_report
            is_valid, _ = validate_report(candidate, band, action)
            if is_valid:
                report_text = candidate
        except Exception:
            pass
            
    # Fallback to local rule-based generator
    if not report_text:
        report_text = generate_local_copilot_report(
            acct_id=acct_id,
            risk_score=risk_score,
            band=band,
            ml_score=ml_score,
            stat_score=stat_score,
            behavior_score=behavior_score,
            boost_applied=boost_applied,
            top_features=top_features
        )
        
    return report_text
