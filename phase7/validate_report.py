"""
validate_report.py
------------------
Step 7B.11: GenAI Output Validation
"""

import re

def validate_report(report_text, band, action):
    """
    Validation checks:
    - Speculation Check: Reject any semantic guessing of F-codes (e.g. Fxxx means...)
    - Terminology Check: Reject if correct risk band name is missing
    - Action Consistency: Reject if recommended banking action is missing
    - Accusation Check: Reject if there are direct, absolute fraud accusations (keeps tone probabilistic)
    """
    # 1. Speculation Check
    guesses = re.findall(r"F\d{3,4}\s+means|F\d{3,4}\s+represents|F\d{3,4}\s+refers\s+to|F\d{3,4}\s+is\s+the|meaning\s+of\s+F\d{3,4}", report_text, re.IGNORECASE)
    if guesses:
        return False, f"Speculative F-code definition found: {guesses}"
        
    # 2. Terminology Check (Risk Band must be present)
    if band.lower() not in report_text.lower():
        return False, f"Risk band '{band}' not mentioned in report text."
        
    # 3. Recommended Action Consistency
    clean_action = re.sub(r'[^\w\s]', '', action.lower())
    clean_report = re.sub(r'[^\w\s]', '', report_text.lower())
    action_words = clean_action.split()
    matched_words = sum(1 for w in action_words if w in clean_report)
    # Check if a substantial part of the action is in the report
    if matched_words < min(3, len(action_words)):
        return False, f"Recommended action '{action}' not referenced in report text."
        
    # 4. Accusation Check
    accusations = ["is a fraudster", "is a criminal", "guilty of fraud", "committed fraud", "definitely a mule", "this is a mule account"]
    for acc in accusations:
        if acc in report_text.lower():
            return False, f"Direct absolute accusation found: '{acc}'"
            
    # Check that basic section headers exist
    headers = ["Investigation Summary", "Reasons for Suspicion", "Recommended Actions", "Investigation Priority"]
    for h in headers:
        if h.lower() not in report_text.lower():
            return False, f"Required section header '{h}' is missing."
            
    return True, "Validation passed successfully."
