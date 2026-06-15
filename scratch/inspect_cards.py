import json

with open("phase6/investigation_cards.json") as f:
    d = json.load(f)

print("--- CRITICAL ACCOUNTS ---")
crit = {k: v for k, v in d["flagged_accounts"].items() if v["risk_band"] == "Critical"}
for a, c in list(crit.items())[:3]:
    print(f"\nAccount: {a}")
    print(f"Risk Score: {c['risk_score']}")
    print(f"ML Score: {c['ml_score']} (Weighted: {c['ml_weighted']})")
    print(f"Stat Score: {c['stat_score']} (Weighted: {c['stat_weighted']})")
    print(f"Behavior Score: {c['behavior_score']} (Weighted: {c['behavior_weighted']})")
    print(f"Recommended Action: {c['recommended_action']}")
    print("Top SHAP Contributors:")
    for f_info in c["top_shap_contributors"]:
        print(f"  - Rank {f_info['rank']}: {f_info['feature']} ({f_info['label']}): value={f_info['feature_value']}, SHAP={f_info['shap_value']}")
    print("Narrative Summary:")
    print(c["narrative"])

print("\n--- HIGH RISK ACCOUNTS ---")
hr = {k: v for k, v in d["flagged_accounts"].items() if v["risk_band"] == "High Risk"}
for a, c in list(hr.items())[:2]:
    print(f"\nAccount: {a}")
    print(f"Risk Score: {c['risk_score']}")
    print(f"ML Score: {c['ml_score']} (Weighted: {c['ml_weighted']})")
    print(f"Stat Score: {c['stat_score']} (Weighted: {c['stat_weighted']})")
    print(f"Behavior Score: {c['behavior_score']} (Weighted: {c['behavior_weighted']})")
    print(f"Recommended Action: {c['recommended_action']}")

print("\n--- FALSE NEGATIVES ---")
for a, c in d["false_negatives"].items():
    print(f"\nAccount: {a}")
    print(f"Risk Score: {c['risk_score']}")
    print(f"ML Score: {c['ml_score']}, Stat Score: {c['stat_score']}, Behavior Score: {c['behavior_score']}")
    print("Top Risk Increasing:")
    for f_info in c["top_risk_increasing_features"][:3]:
        print(f"  {f_info['feature']} ({f_info['label']}): value={f_info['feature_value']}, SHAP={f_info['shap_value']}")
    print("Top Risk Reducing:")
    for f_info in c["top_risk_reducing_features"][:3]:
        print(f"  {f_info['feature']} ({f_info['label']}): value={f_info['feature_value']}, SHAP={f_info['shap_value']}")

print("\n--- FALSE POSITIVES OVERVIEW ---")
fps = d["false_positives"]
print(f"Total FPs: {len(fps)}")
print("Sample FPs:")
for a, c in list(fps.items())[:3]:
    print(f"  {a}: risk_score={c['risk_score']}, ml={c['ml_score']}, stat={c['stat_score']}, behavior={c['behavior_score']}")
