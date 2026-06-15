"""
generate_explanations.py
------------------------
Phase 6: Explainability & Investigation Reports
Bank of India -- Mule Account Detection

Steps 6.1 – 6.11:
1. Load all phase outputs and reconstruct test set alignment.
2. Compute SHAP values using TreeExplainer.
3. Extract top-5 contributing features per account.
4. Build a verified feature dictionary (NO speculation).
5. Generate investigation cards for all 32 flagged accounts.
6. Create SHAP summary (PNG), waterfall (PNG), and force (HTML/JS) plots.
7. Generate narrative explanations covering all 3 scoring pillars.
8. Investigate false positives (18) and false negatives (2).
"""

import os
import sys
import json

# Bootstrap project root
def bootstrap_root():
    path = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(path, "phase1")):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    if path not in sys.path:
        sys.path.insert(0, path)
    return path

PROJECT_ROOT = bootstrap_root()
from config.paths import DATA_PHASE2, PHASE3_DIR, PHASE5_DIR, PHASE6_DIR
import joblib
import numpy as np
import pandas as pd
import shap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = str(PROJECT_ROOT)
# PHASE3_DIR, PHASE5_DIR, and PHASE6_DIR are imported from config.paths
os.makedirs(PHASE6_DIR, exist_ok=True)

print("=" * 60)
print("Phase 6: Explainability & Investigation Reports")
print("=" * 60)

# =====================================================================
# STEP 6.1 — Load Phase Outputs
# =====================================================================
print("\n--- Step 6.1: Loading Phase Outputs ---")

# LightGBM model
model = joblib.load(os.path.join(PHASE3_DIR, "best_model.pkl"))
print(f"  Loaded LightGBM model: {type(model).__name__}")

# Features and target
X_final = pd.read_csv(os.path.join(DATA_PHASE2, "X_final.csv"))
y_final = pd.read_csv(os.path.join(DATA_PHASE2, "y_final.csv")).values.ravel()
feature_names = list(X_final.columns)
print(f"  Loaded X_final: {X_final.shape}, y_final: {y_final.shape}")

# Test indices
test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))
print(f"  Loaded test_indices: {len(test_idx)} accounts")

# Risk scores from Phase 5
df_risk = pd.read_csv(os.path.join(PHASE5_DIR, "risk_scores.csv"))
print(f"  Loaded risk_scores.csv: {df_risk.shape}")

# Reconstruct test set using same alignment as Phase 5
X_test = X_final.iloc[test_idx].reset_index(drop=True)
y_test = y_final[test_idx]
print(f"  Reconstructed X_test: {X_test.shape}, y_test sum: {y_test.sum()}")

# Verify alignment with Phase 5 risk scores
assert len(X_test) == len(df_risk), f"Length mismatch: X_test={len(X_test)}, risk_scores={len(df_risk)}"
assert y_test.sum() == df_risk["target"].sum(), "Target sum mismatch between test set and risk_scores!"
print("  ✅ Alignment verified.")

# =====================================================================
# STEP 6.2 — Compute SHAP Values
# =====================================================================
print("\n--- Step 6.2: Computing SHAP Values ---")

explainer = shap.TreeExplainer(model)
print("  TreeExplainer initialized.")

shap_values_raw = explainer.shap_values(X_test)

# LightGBM binary: shap_values returns [class_0, class_1] — take class 1
if isinstance(shap_values_raw, list):
    shap_matrix = shap_values_raw[1]
    print(f"  SHAP values extracted for class 1 (mule). Shape: {shap_matrix.shape}")
else:
    shap_matrix = shap_values_raw
    print(f"  SHAP values shape: {shap_matrix.shape}")

# Save raw SHAP values
shap_path = os.path.join(PHASE6_DIR, "shap_values.npy")
np.save(shap_path, shap_matrix)
print(f"  Saved SHAP values to: {shap_path}")

# =====================================================================
# STEP 6.3 — Extract Top-5 Contributors Per Account
# =====================================================================
print("\n--- Step 6.3: Extracting Top-5 Contributors ---")

top_n = 5
rows = []
for i in range(len(X_test)):
    account_id = df_risk.iloc[i]["account_id"]
    sv = shap_matrix[i]
    abs_sv = np.abs(sv)
    top_indices = np.argsort(abs_sv)[::-1][:top_n]

    for rank, idx in enumerate(top_indices, 1):
        rows.append({
            "account_id": account_id,
            "rank": rank,
            "feature": feature_names[idx],
            "shap_value": round(float(sv[idx]), 6),
            "direction": "+" if sv[idx] > 0 else "-",
            "feature_value": round(float(X_test.iloc[i, idx]), 6)
        })

df_top = pd.DataFrame(rows)
top_csv_path = os.path.join(PHASE6_DIR, "top_features_per_account.csv")
df_top.to_csv(top_csv_path, index=False)
print(f"  Saved top-{top_n} features per account to: {top_csv_path}")
print(f"  Total rows: {len(df_top)} ({len(X_test)} accounts × {top_n})")

# =====================================================================
# STEP 6.4 — Build Feature Dictionary (Verified, No Speculation)
# =====================================================================
print("\n--- Step 6.4: Building Feature Dictionary ---")

# Source 1: Categorical encoding map from mule_preprocessor.py (verified)
categorical_cols_original = ['F3886', 'F3889', 'F3890', 'F3891', 'F3892', 'F3893']
feature_prefix_map = {
    'F3886': 'account_type',
    'F3889': 'historical_code',
    'F3890': 'area_category',
    'F3891': 'occupation',
    'F3892': 'gender',
    'F3893': 'customer_segment'
}

# Source 2: Feature inventory (type, missing %)
feat_inv_path = os.path.join(BASE_DIR, "feature_inventory.csv")
df_inv = pd.read_csv(feat_inv_path)
inv_lookup = {}
for _, row in df_inv.iterrows():
    inv_lookup[row["Feature"]] = {
        "type": row["Type"],
        "missing_pct": round(float(row["Missing %"]), 2)
    }

# Source 3: Feature importance (RF rank)
feat_imp_path = os.path.join(BASE_DIR, "feature_importance.csv")
df_imp = pd.read_csv(feat_imp_path)
rf_rank_lookup = {}
for rank, (_, row) in enumerate(df_imp.iterrows(), 1):
    rf_rank_lookup[row["Feature"]] = rank

# Source 4: BOI-highlighted features
boi_highlighted = [
    'F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122',
    'F2582', 'F2678', 'F2737', 'F2956', 'F3043', 'F3836', 'F3887',
    'F3889', 'F3891', 'F3894'
]

# Build dictionary
feature_dict = {}
for feat in feature_names:
    parts = []

    # Check if this is an encoded categorical column
    encoded_parent = None
    for orig_col, prefix in feature_prefix_map.items():
        if feat.startswith(prefix + "_"):
            encoded_parent = orig_col
            category_value = feat[len(prefix) + 1:]
            parts.append(f"One-hot encoded: {orig_col} ({prefix}) = {category_value}")
            break

    if encoded_parent is None:
        # Regular F-code feature
        if feat in inv_lookup:
            info = inv_lookup[feat]
            parts.append(f"{info['type']} feature, {info['missing_pct']}% missing")
        else:
            parts.append("Feature")

        if feat in rf_rank_lookup:
            parts.append(f"RF importance rank #{rf_rank_lookup[feat]}")

        if feat in boi_highlighted:
            parts.append("BOI-highlighted")

    feature_dict[feat] = "; ".join(parts)

dict_path = os.path.join(PHASE6_DIR, "feature_dictionary.json")
with open(dict_path, "w", encoding="utf-8") as f:
    json.dump(feature_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

# Verify: count encoded vs F-code
n_encoded = sum(1 for f in feature_names if any(f.startswith(p + "_") for p in feature_prefix_map.values()))
print(f"  Feature dictionary: {len(feature_dict)} entries")
print(f"    Encoded categorical columns: {n_encoded}")
print(f"    F-code features: {len(feature_dict) - n_encoded}")
print(f"    BOI-highlighted features in model: {sum(1 for f in feature_names if f in boi_highlighted)}")
print(f"  Saved to: {dict_path}")

# =====================================================================
# STEP 6.5 & 6.9 — Generate Investigation Cards with Narratives
# =====================================================================
print("\n--- Step 6.5 & 6.9: Generating Investigation Cards ---")

# Compute test-set average stat score for narrative context
avg_stat_score = df_risk["stat_score"].mean()

flagged = df_risk[df_risk["risk_band"] != "Normal"].copy()
flagged_indices = flagged.index.tolist()
print(f"  Flagged accounts: {len(flagged)}")

investigation_cards = {}

for idx in flagged_indices:
    row = df_risk.iloc[idx]
    acct = row["account_id"]
    sv = shap_matrix[idx]
    abs_sv = np.abs(sv)
    top_indices = np.argsort(abs_sv)[::-1][:5]

    # Top SHAP contributors
    top_contributors = []
    for rank, fi in enumerate(top_indices, 1):
        feat = feature_names[fi]
        top_contributors.append({
            "rank": rank,
            "feature": feat,
            "label": feature_dict.get(feat, feat),
            "shap_value": round(float(sv[fi]), 6),
            "direction": "+" if sv[fi] > 0 else "-",
            "feature_value": round(float(X_test.iloc[idx, fi]), 6)
        })

    # Determine boost
    boost_applied = row["behavior_score"] >= 99.0
    boost_amount = 10.0 if boost_applied else 0.0

    # Build narrative (all 3 pillars)
    narrative_lines = []
    narrative_lines.append(
        f"Account {acct} — {row['risk_band'].upper()} (risk score: {row['risk_score']})"
    )

    # ML pillar
    ml_weighted = round(row["ml_score"] * 0.70, 2)
    narrative_lines.append(
        f"\nML Component (weight: 70%): ML score = {round(row['ml_score'], 2)}. "
        f"Weighted contribution = {ml_weighted}."
    )
    if row["ml_score"] > 1.0:
        shap_desc_parts = []
        for tc in top_contributors[:3]:
            shap_desc_parts.append(
                f"  {tc['feature']} ({tc['label']}): "
                f"{tc['direction']}{abs(tc['shap_value']):.4f}"
            )
        narrative_lines.append("  Top SHAP drivers:")
        narrative_lines.extend(shap_desc_parts)
    else:
        narrative_lines.append(
            "  ML model assigned near-zero probability. "
            "SHAP contributions are minimal."
        )

    # Statistical anomaly pillar
    stat_weighted = round(row["stat_score"] * 0.10, 2)
    stat_comparison = "above" if row["stat_score"] > avg_stat_score else "below"
    narrative_lines.append(
        f"\nStatistical Anomaly (weight: 10%): Score = {round(row['stat_score'], 2)}. "
        f"Weighted contribution = {stat_weighted}. "
        f"This is {stat_comparison} the test-set average ({round(avg_stat_score, 2)})."
    )

    # Behavioral anomaly pillar
    beh_weighted = round(row["behavior_score"] * 0.20, 2)
    narrative_lines.append(
        f"\nBehavioral Anomaly (weight: 20%): LOF percentile = {round(row['behavior_score'], 2)}. "
        f"Weighted contribution = {beh_weighted}."
    )
    if boost_applied:
        narrative_lines.append(
            f"  Behavioral boost of +{boost_amount} applied (LOF percentile >= 99). "
            f"Account is in the top 1% of behavioral outliers."
        )

    # Assessment
    pillars_driving = []
    if ml_weighted > 20:
        pillars_driving.append("ML")
    if stat_weighted > 5:
        pillars_driving.append("Statistical Anomaly")
    if beh_weighted > 10 or boost_applied:
        pillars_driving.append("Behavioral Anomaly")

    if pillars_driving:
        narrative_lines.append(
            f"\nAssessment: Alert primarily driven by {', '.join(pillars_driving)}."
        )
    else:
        narrative_lines.append(
            "\nAssessment: Marginal contributions from multiple pillars."
        )

    card = {
        "account_id": acct,
        "target": int(row["target"]),
        "risk_score": float(row["risk_score"]),
        "risk_band": row["risk_band"],
        "recommended_action": row["recommended_action"],
        "ml_score": round(float(row["ml_score"]), 2),
        "ml_weighted": ml_weighted,
        "stat_score": round(float(row["stat_score"]), 2),
        "stat_weighted": stat_weighted,
        "behavior_score": round(float(row["behavior_score"]), 2),
        "behavior_weighted": beh_weighted,
        "boost_applied": boost_applied,
        "boost_amount": boost_amount,
        "top_shap_contributors": top_contributors,
        "narrative": "\n".join(narrative_lines)
    }
    investigation_cards[acct] = card

print(f"  Generated {len(investigation_cards)} investigation cards.")

# =====================================================================
# STEP 6.10 — False Positive Investigation
# =====================================================================
print("\n--- Step 6.10: False Positive Investigation ---")

fp_df = df_risk[(df_risk["risk_band"] != "Normal") & (df_risk["target"] == 0)]
fp_cards = {}

for idx in fp_df.index:
    row = df_risk.iloc[idx]
    acct = row["account_id"]
    sv = shap_matrix[idx]
    abs_sv = np.abs(sv)
    top_indices = np.argsort(abs_sv)[::-1][:5]

    top_contributors = []
    for rank, fi in enumerate(top_indices, 1):
        feat = feature_names[fi]
        top_contributors.append({
            "rank": rank,
            "feature": feat,
            "label": feature_dict.get(feat, feat),
            "shap_value": round(float(sv[fi]), 6),
            "direction": "+" if sv[fi] > 0 else "-",
            "feature_value": round(float(X_test.iloc[idx, fi]), 6)
        })

    boost_applied = row["behavior_score"] >= 99.0
    primary_driver = "Behavioral Anomaly (LOF boost)" if boost_applied else "Unknown"

    fp_cards[acct] = {
        "account_id": acct,
        "risk_score": float(row["risk_score"]),
        "risk_band": row["risk_band"],
        "ml_score": round(float(row["ml_score"]), 2),
        "stat_score": round(float(row["stat_score"]), 2),
        "behavior_score": round(float(row["behavior_score"]), 2),
        "boost_applied": boost_applied,
        "primary_false_alert_driver": primary_driver,
        "top_shap_contributors": top_contributors,
        "analysis": (
            f"Account {acct} is a false positive (actual: normal, predicted band: {row['risk_band']}). "
            f"ML score is {round(row['ml_score'], 2)} (near zero), indicating the ML model "
            f"correctly identified this as non-mule. The alert was triggered by the behavioral "
            f"anomaly LOF percentile of {round(row['behavior_score'], 2)} "
            f"{'(with +10 boost applied)' if boost_applied else ''}, which pushed the fused "
            f"score above the Normal threshold."
        )
    }

print(f"  False Positives analyzed: {len(fp_cards)}")
fp_boost_count = sum(1 for c in fp_cards.values() if c["boost_applied"])
print(f"  FPs driven by LOF boost: {fp_boost_count}/{len(fp_cards)}")

# =====================================================================
# STEP 6.11 — False Negative Investigation
# =====================================================================
print("\n--- Step 6.11: False Negative Investigation ---")

fn_df = df_risk[(df_risk["risk_band"] == "Normal") & (df_risk["target"] == 1)]
fn_cards = {}

for idx in fn_df.index:
    row = df_risk.iloc[idx]
    acct = row["account_id"]
    sv = shap_matrix[idx]
    abs_sv = np.abs(sv)

    # Get ALL features sorted by SHAP magnitude for deeper analysis
    all_sorted = np.argsort(abs_sv)[::-1]

    # Top 5 positive (risk-increasing) contributors
    pos_indices = [i for i in all_sorted if sv[i] > 0][:5]
    # Top 5 negative (risk-reducing) contributors
    neg_indices = [i for i in all_sorted if sv[i] < 0][:5]

    top_positive = []
    for rank, fi in enumerate(pos_indices, 1):
        feat = feature_names[fi]
        top_positive.append({
            "rank": rank,
            "feature": feat,
            "label": feature_dict.get(feat, feat),
            "shap_value": round(float(sv[fi]), 6),
            "feature_value": round(float(X_test.iloc[idx, fi]), 6)
        })

    top_negative = []
    for rank, fi in enumerate(neg_indices, 1):
        feat = feature_names[fi]
        top_negative.append({
            "rank": rank,
            "feature": feat,
            "label": feature_dict.get(feat, feat),
            "shap_value": round(float(sv[fi]), 6),
            "feature_value": round(float(X_test.iloc[idx, fi]), 6)
        })

    fn_cards[acct] = {
        "account_id": acct,
        "risk_score": float(row["risk_score"]),
        "risk_band": row["risk_band"],
        "ml_score": round(float(row["ml_score"]), 2),
        "stat_score": round(float(row["stat_score"]), 2),
        "behavior_score": round(float(row["behavior_score"]), 2),
        "top_risk_increasing_features": top_positive,
        "top_risk_reducing_features": top_negative,
        "analysis": (
            f"Account {acct} is a FALSE NEGATIVE — an actual money mule that was classified "
            f"in the Normal risk band (score: {row['risk_score']}). "
            f"ML score: {round(row['ml_score'], 2)}, Stat score: {round(row['stat_score'], 2)}, "
            f"Behavior score: {round(row['behavior_score'], 2)}. "
            f"All three scoring pillars assigned low scores. The ML model was unable to detect "
            f"this mule because the top risk-reducing SHAP features dominated the prediction. "
            f"This account may represent a sophisticated mule pattern not captured by the "
            f"current feature set."
        )
    }

print(f"  False Negatives analyzed: {len(fn_cards)}")
for acct, card in fn_cards.items():
    print(f"    {acct}: risk_score={card['risk_score']}, ml={card['ml_score']}, "
          f"stat={card['stat_score']}, behavior={card['behavior_score']}")

# =====================================================================
# Combine and Save All Investigation Cards
# =====================================================================
all_cards = {
    "flagged_accounts": investigation_cards,
    "false_positives": fp_cards,
    "false_negatives": fn_cards,
    "metadata": {
        "total_flagged": len(investigation_cards),
        "total_false_positives": len(fp_cards),
        "total_false_negatives": len(fn_cards),
        "fp_lof_boost_driven": fp_boost_count,
        "test_set_avg_stat_score": round(avg_stat_score, 2)
    }
}

cards_path = os.path.join(PHASE6_DIR, "investigation_cards.json")
with open(cards_path, "w", encoding="utf-8") as f:
    json.dump(all_cards, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print(f"\n  Saved all investigation cards to: {cards_path}")

# =====================================================================
# STEP 6.6 — SHAP Summary Plot (Global, PNG)
# =====================================================================
print("\n--- Step 6.6: Generating SHAP Summary Plot ---")

plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_matrix,
    X_test,
    feature_names=feature_names,
    max_display=20,
    show=False,
    plot_size=None
)
plt.title("SHAP Feature Importance — Top 20 Features (Test Set)", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
summary_path = os.path.join(PHASE6_DIR, "shap_summary_plot.png")
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved SHAP summary plot to: {summary_path}")

# =====================================================================
# STEP 6.7 — SHAP Waterfall Plots (Critical + High Risk, PNG)
# =====================================================================
print("\n--- Step 6.7: Generating Waterfall Plots ---")

critical_high = df_risk[df_risk["risk_band"].isin(["Critical", "High Risk"])]
print(f"  Generating waterfall plots for {len(critical_high)} accounts (Critical + High Risk)")

for idx in critical_high.index:
    acct = df_risk.iloc[idx]["account_id"]
    band = df_risk.iloc[idx]["risk_band"]

    # Create a shap.Explanation object for this account
    explanation = shap.Explanation(
        values=shap_matrix[idx],
        base_values=float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
        data=X_test.iloc[idx].values,
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.title(f"SHAP Waterfall — {acct} ({band}, score: {df_risk.iloc[idx]['risk_score']})",
              fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    wf_path = os.path.join(PHASE6_DIR, f"waterfall_{acct}.png")
    plt.savefig(wf_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {wf_path}")

# =====================================================================
# STEP 6.8 — SHAP Force Plots (Critical, HTML/JS Only)
# =====================================================================
print("\n--- Step 6.8: Generating Force Plots (HTML/JS) ---")

critical_df = df_risk[df_risk["risk_band"] == "Critical"]
print(f"  Generating force plots for {len(critical_df)} Critical accounts")

expected_val = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)

for idx in critical_df.index:
    acct = df_risk.iloc[idx]["account_id"]

    force_plot = shap.force_plot(
        expected_val,
        shap_matrix[idx],
        X_test.iloc[idx],
        feature_names=feature_names,
        matplotlib=False  # Use JS visualization
    )

    force_html_path = os.path.join(PHASE6_DIR, f"force_{acct}.html")
    shap.save_html(force_html_path, force_plot)
    print(f"    Saved: {force_html_path}")

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("Phase 6 — Deliverables Summary:")
print("=" * 60)
print(f"  shap_values.npy          : {shap_matrix.shape}")
print(f"  top_features_per_account : {len(df_top)} rows")
print(f"  feature_dictionary.json  : {len(feature_dict)} entries")
print(f"  investigation_cards.json : {len(investigation_cards)} flagged + {len(fp_cards)} FP + {len(fn_cards)} FN")
print(f"  shap_summary_plot.png    : Global SHAP beeswarm")
print(f"  Waterfall plots          : {len(critical_high)} PNG files")
print(f"  Force plots              : {len(critical_df)} HTML files")
print("=" * 60)
print("Phase 6 generate_explanations.py completed successfully.")
print("=" * 60)
