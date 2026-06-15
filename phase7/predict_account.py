"""
predict_account.py
------------------
Master inference script for new accounts.
Step 7B.1 to Step 7B.11
"""

import os
import sys
import json
import argparse
import joblib
import warnings
import pandas as pd
import numpy as np
import shap

warnings.filterwarnings("ignore")

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
from config.paths import DATA_PHASE1, DATA_PHASE2, PHASE3_DIR, PHASE4_DIR, PHASE4B_DIR, PHASE5_DIR, PHASE6_DIR, PHASE7_DIR

# Import the custom preprocessor (using package absolute import)
from phase2.mule_preprocessor import MuleAccountPreprocessor
from phase7.generate_report import generate_report, RECOMMENDED_ACTIONS
from phase7.validate_report import validate_report

BASE_DIR = str(PROJECT_ROOT)

# Global variables for models and parameters
PREPROCESSOR = None
ML_MODEL = None
ISO_MODEL = None
ISO_SCALER = None
LOF_MODEL = None
LOF_SCALER = None
REF_LOF_SCORES = None
FEATURE_DICTIONARY = None

BEHAVIOR_MEDIANS = {}
BEHAVIOR_MODES = {}
BEHAVIOR_ENCODINGS = {}

# =====================================================================
# LOAD ARTIFACTS AND FIT BEHAVIORAL MAPPINGS
# =====================================================================
def load_all_artifacts():
    global PREPROCESSOR, ML_MODEL, ISO_MODEL, ISO_SCALER, LOF_MODEL, LOF_SCALER, REF_LOF_SCORES, FEATURE_DICTIONARY
    global BEHAVIOR_MEDIANS, BEHAVIOR_MODES, BEHAVIOR_ENCODINGS
    
    print("\n--- Loading Model Artifacts ---")
    
    # 1. Preprocessor
    PREPROCESSOR = joblib.load(os.path.join(DATA_PHASE2, "preprocessing_pipeline.pkl"))
    print("  Loaded Preprocessing Pipeline successfully.")
    
    # 2. LightGBM Supervised Model
    ML_MODEL = joblib.load(os.path.join(PHASE3_DIR, "best_model.pkl"))
    print("  Loaded LightGBM Classifier successfully.")
    
    # 3. Isolation Forest Anomaly Model & Scaler
    ISO_MODEL = joblib.load(os.path.join(PHASE4_DIR, "isolation_forest.pkl"))
    ISO_SCALER = joblib.load(os.path.join(PHASE4_DIR, "isolation_forest_scaler.pkl"))
    print("  Loaded Isolation Forest and MinMaxScaler successfully.")
    
    # 4. LOF Behavioral Anomaly Model & Scaler
    LOF_MODEL = joblib.load(os.path.join(PHASE4B_DIR, "behavioral_lof.pkl"))
    LOF_SCALER = joblib.load(os.path.join(PHASE4B_DIR, "behavior_scaler.pkl"))
    
    # 5. Reference LOF Scores for Percentile Mapping
    ref_scores_df = pd.read_csv(os.path.join(PHASE4B_DIR, "behavioral_anomaly_scores.csv"))
    REF_LOF_SCORES = ref_scores_df["anomaly_signal"].values
    print(f"  Loaded LOF Model, Scaler, and {len(REF_LOF_SCORES)} reference scores successfully.")
    
    # 6. Feature Dictionary (for verified metadata in SHAP explanations)
    with open(os.path.join(PHASE6_DIR, "feature_dictionary.json"), "r", encoding="utf-8") as f:
        FEATURE_DICTIONARY = json.load(f)
    print("  Loaded Feature Dictionary successfully.")
    
    # Reconstruct training splits to fit behavioral mappings (leakage-free)
    print("  Constructing behavioral mapping parameters from training split...")
    df_clean = pd.read_csv(os.path.join(DATA_PHASE2, "dataset_cleaned.csv"))
    df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    train_idx = np.load(os.path.join(PHASE3_DIR, "train_indices.npy"))
    df_train = df_shuffled.iloc[train_idx].copy()
    
    # Parse F3888 to engineer age features on df_train before median calculation
    ref_date = pd.to_datetime('2025-12-31')
    opening_date_train = pd.to_datetime(df_train['F3888'], format='mixed', errors='coerce')
    df_train['account_age_days'] = (ref_date - opening_date_train).dt.days
    df_train['account_age_years'] = df_train['account_age_days'] / 365.25
    
    # 1. Medians for raw continuous columns
    raw_continuous_cols = ['F2737', 'F2678', 'F3836', 'account_age_days', 'account_age_years']
    for col in raw_continuous_cols:
        val = df_train[col].median()
        BEHAVIOR_MEDIANS[col] = 0.0 if pd.isna(val) else val
        
    # Modes for raw categorical columns
    categorical_cols = ['F3891', 'F3890', 'F3886', 'F3893', 'F3892']
    for col in categorical_cols:
        val = df_train[col].mode()[0] if not df_train[col].mode().empty else 'Missing'
        BEHAVIOR_MODES[col] = val
        
    # Target encodings: P(mule | category) on train split
    global_mule_rate = df_train['F3924'].mean()
    categorical_risk_mapping = [
        ('F3891', 'occupation_risk_score'),
        ('F3890', 'area_risk_score'),
        ('F3886', 'account_type_risk_score'),
        ('F3893', 'customer_segment_risk_score'),
        ('F3892', 'gender_risk_score')
    ]
    for orig_col, encoded_col in categorical_risk_mapping:
        train_counts = df_train.groupby(orig_col)['F3924'].count()
        train_mules = df_train.groupby(orig_col)['F3924'].sum()
        train_encoding = (train_mules / train_counts).to_dict()
        BEHAVIOR_ENCODINGS[orig_col] = {
            "mapping": train_encoding,
            "global_rate": global_mule_rate
        }
    print("  ✅ Mappings initialized.")


# =====================================================================
# STEP 7B.3 — Behavioral Feature Builder
# =====================================================================
def build_behavioral_features(raw_df):
    """
    Constructs the 10 behavioral features from the raw DataFrame,
    applying median/mode imputation and risk encoding from training data.
    """
    df_beh = pd.DataFrame(index=raw_df.index)
    
    # 1. Parse account age relative to reference date 2025-12-31
    ref_date = pd.to_datetime('2025-12-31')
    opening_date = pd.to_datetime(raw_df['F3888'], format='mixed', errors='coerce')
    df_beh['account_age_days'] = (ref_date - opening_date).dt.days
    df_beh['account_age_years'] = df_beh['account_age_days'] / 365.25
    
    # 2. Impute raw continuous columns
    df_beh['F2737'] = raw_df['F2737'].fillna(BEHAVIOR_MEDIANS['F2737'])
    df_beh['F2678'] = raw_df['F2678'].fillna(BEHAVIOR_MEDIANS['F2678'])
    df_beh['F3836'] = raw_df['F3836'].fillna(BEHAVIOR_MEDIANS['F3836'])
    
    df_beh['account_age_days'] = df_beh['account_age_days'].fillna(BEHAVIOR_MEDIANS['account_age_days'])
    df_beh['account_age_years'] = df_beh['account_age_years'].fillna(BEHAVIOR_MEDIANS['account_age_years'])
    
    # 3. Apply target-based risk encodings (categorical columns)
    categorical_risk_mapping = [
        ('F3891', 'occupation_risk_score'),
        ('F3890', 'area_risk_score'),
        ('F3886', 'account_type_risk_score'),
        ('F3893', 'customer_segment_risk_score'),
        ('F3892', 'gender_risk_score')
    ]
    
    for orig_col, encoded_col in categorical_risk_mapping:
        # Impute missing categories with train mode
        clean_cat_col = raw_df[orig_col].fillna(BEHAVIOR_MODES[orig_col])
        # Map values to risk encoding
        mapping_meta = BEHAVIOR_ENCODINGS[orig_col]
        df_beh[encoded_col] = clean_cat_col.map(mapping_meta["mapping"]).fillna(mapping_meta["global_rate"])
        
    # 4. Engineer Ratio Features
    eps = 1e-6
    df_beh['credit_debit_ratio'] = (df_beh['F2737'].abs() + eps) / (df_beh['F2678'].abs() + eps)
    df_beh['balance_retention_ratio'] = df_beh['F3836'].abs() / (df_beh['F2737'].abs() + eps)
    df_beh['pass_through_ratio'] = df_beh['F2678'].abs() / (df_beh['F2737'].abs() + eps)
    
    # Selected columns in exact order
    behavioral_cols = [
        'account_age_days', 'account_age_years',
        'occupation_risk_score', 'area_risk_score', 'account_type_risk_score',
        'customer_segment_risk_score', 'gender_risk_score',
        'credit_debit_ratio', 'balance_retention_ratio', 'pass_through_ratio'
    ]
    
    return df_beh[behavioral_cols]


# =====================================================================
# INDIVIDUAL ACCOUNT PREDICTION PIPELINE
# =====================================================================
def predict_account(account_df, api_key=None):
    """
    Runs prediction for a single account (single row DataFrame).
    """
    acct_id = account_df.iloc[0]["account_id"]
    
    # 1. Preprocess raw account data into model-ready features (300 columns)
    X_processed = PREPROCESSOR.transform(account_df)
    
    # 2. ML Prediction (Step 7B.4)
    ml_probability = float(ML_MODEL.predict_proba(X_processed)[0, 1])
    ml_score = ml_probability * 100
    predicted_class = int(ml_probability >= 0.40)
    
    # 3. Statistical Anomaly Score (Step 7B.5)
    raw_if_score = float(ISO_MODEL.decision_function(X_processed)[0])
    stat_score_raw = -raw_if_score
    stat_score = float(ISO_SCALER.transform([[stat_score_raw]])[0, 0])
    stat_score = np.clip(stat_score, 0.0, 100.0)
    
    # 4. Behavioral Anomaly Score (Step 7B.6)
    behavior_features = build_behavioral_features(account_df)
    behavior_features_scaled = LOF_SCALER.transform(behavior_features)
    # raw_lof is -decision_function
    raw_lof = -float(LOF_MODEL.decision_function(behavior_features_scaled)[0])
    
    # Map raw LOF signal to percentile rank score relative to test set reference scores
    percentile = (REF_LOF_SCORES < raw_lof).mean()
    behavior_score = percentile * 100.0
    
    # 5. Risk Score Fusion & Boosting (Step 7B.7)
    risk_score = 0.70 * ml_score + 0.10 * stat_score + 0.20 * behavior_score
    boost_applied = False
    if behavior_score >= 99.0:
        risk_score += 10.0
        boost_applied = True
    risk_score = min(risk_score, 100.0)
    risk_score_rounded = round(risk_score, 2)
    
    # 6. Assign Risk Band (Step 7B.8)
    if risk_score_rounded <= 30.0:
        risk_band = "Normal"
    elif risk_score_rounded <= 60.0:
        risk_band = "Monitor"
    elif risk_score_rounded <= 80.0:
        risk_band = "High Risk"
    else:
        risk_band = "Critical"
        
    action = RECOMMENDED_ACTIONS[risk_band]
    
    # 7. Dynamic SHAP Computation (Step 7B.9)
    explainer = shap.TreeExplainer(ML_MODEL)
    shap_values_raw = explainer.shap_values(X_processed)
    
    # Handle both newer and older shap structures safely
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1][0]
    else:
        shap_vals = shap_values_raw[0] if len(shap_values_raw.shape) == 2 else shap_values_raw
        
    # Get all features sorted by SHAP value
    feature_names = list(X_processed.columns)
    shap_indices_sorted = np.argsort(shap_vals)[::-1]
    
    pos_contrib = []
    neg_contrib = []
    
    for idx in shap_indices_sorted:
        feat = feature_names[idx]
        val = round(float(X_processed.iloc[0, idx]), 6)
        sv = round(float(shap_vals[idx]), 6)
        label = FEATURE_DICTIONARY.get(feat, feat)
        
        contrib = {
            "feature": feat,
            "label": label,
            "feature_value": val,
            "shap_value": sv
        }
        
        if sv > 0:
            pos_contrib.append(contrib)
        elif sv < 0:
            neg_contrib.append(contrib)
            
    # Positive: top 5 descending
    top_pos = pos_contrib[:5]
    # Negative: top 5 ascending (most negative first)
    top_neg = sorted(neg_contrib, key=lambda x: x["shap_value"])[:5]
    
    top_shap_features = [c["feature"] for c in top_pos + top_neg]
    
    # 8. Report Generation & Validation (Step 7B.10 & 7B.11)
    report_text = generate_report(
        acct_id=acct_id,
        risk_score=risk_score_rounded,
        band=risk_band,
        ml_score=round(ml_score, 2),
        stat_score=round(stat_score, 2),
        behavior_score=round(behavior_score, 2),
        boost_applied=boost_applied,
        top_features=top_pos + top_neg,
        api_key=api_key
    )
    
    # Output schema (returns rich dictionary covering all user requested keys)
    response = {
        "account_id": acct_id,
        "predicted_class": predicted_class,
        "ml_probability": round(ml_probability, 6),
        "risk_score": risk_score_rounded,
        "prediction": risk_band,  # Matches Step 7B.11 Output
        "risk_band": risk_band,   # Matches Step 7.10 Output
        "recommended_action": action,
        "top_features": top_shap_features[:5], # Keep top 5 features overall
        "top_positive_contributors": top_pos,
        "top_negative_contributors": top_neg,
        "report": report_text
    }
    
    return response


# =====================================================================
# SCHEMA VALIDATION (Step 7B.1)
# =====================================================================
def validate_schema(batch_df, expected_cols):
    """
    Validates that:
    - All required columns exist.
    - Column names match training schema.
    - Target F3924 is absent.
    Returns:
    - valid_df: DataFrame of valid records.
    - invalid_df: DataFrame of invalid records.
    """
    missing_cols = [col for col in expected_cols if col not in batch_df.columns]
    
    if missing_cols or "F3924" in batch_df.columns:
        valid_rows = []
        invalid_rows = []
        
        for idx, row in batch_df.iterrows():
            row_dict = row.to_dict()
            is_valid = True
            reason = ""
            
            # Check target
            if "F3924" in row_dict:
                is_valid = False
                reason = "Target column 'F3924' must be absent during inference."
            # Check missing
            else:
                row_missing = [c for c in expected_cols if c not in row_dict]
                if row_missing:
                    is_valid = False
                    reason = f"Missing required columns: {row_missing}"
                    
            if is_valid:
                valid_rows.append(row_dict)
            else:
                row_dict["failed_reason"] = reason
                invalid_rows.append(row_dict)
                
        valid_df = pd.DataFrame(valid_rows)
        invalid_df = pd.DataFrame(invalid_rows)
        return valid_df, invalid_df
    else:
        return batch_df, pd.DataFrame()


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Inference Pipeline")
    parser.add_argument("--account_idx", type=int, help="Index of single test account in dataset.csv to predict")
    parser.add_argument("--batch", type=str, help="Path to batch input CSV file of new accounts")
    args = parser.parse_args()
    
    load_all_artifacts()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    # 1. Single Account Index Prediction (Useful for testing/demo)
    if args.account_idx is not None:
        raw_dataset_path = os.path.join(DATA_PHASE1, "dataset.csv")
        df_raw = pd.read_csv(raw_dataset_path)
        
        if args.account_idx < 0 or args.account_idx >= len(df_raw):
            print(f"Error: Account index must be between 0 and {len(df_raw)-1}")
            sys.exit(1)
            
        row = df_raw.iloc[[args.account_idx]].copy()
        
        # Format input row
        if "Unnamed: 0" in row.columns:
            row = row.drop(columns=["Unnamed: 0"])
        if "F3924" in row.columns:
            row = row.drop(columns=["F3924"])
        row.insert(0, "account_id", f"TEST{args.account_idx:04d}")
        
        print(f"\nRunning prediction for single account at index {args.account_idx}...")
        res = predict_account(row, api_key=api_key)
        
        # Save prediction.json
        output_json_path = os.path.join(PHASE7_DIR, "prediction.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
            
        print("\nStructured JSON Output:")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n[DONE] Saved prediction JSON to: {output_json_path}")
        
    # 2. Batch CSV Prediction (Step 7B.11)
    elif args.batch is not None:
        print(f"\nRunning batch prediction for CSV: {args.batch}...")
        
        if not os.path.exists(args.batch):
            # Check if it's located in phase7/
            alt_path = os.path.join(PHASE7_DIR, args.batch)
            if os.path.exists(alt_path):
                args.batch = alt_path
            else:
                print(f"Error: Batch file {args.batch} not found.")
                sys.exit(1)
                
        df_batch = pd.read_csv(args.batch)
        print(f"  Loaded {len(df_batch)} accounts.")
        
        # Reconstruct expected raw columns from preprocessor features lists
        # Raw columns needed for transform: categorical cols, continuous and binary cols, and F3888 (date)
        # We can extract it by reading columns of raw dataset minus Unnamed: 0 and target F3924
        raw_dataset_path = os.path.join(DATA_PHASE1, "dataset.csv")
        df_example = pd.read_csv(raw_dataset_path, nrows=5)
        if "Unnamed: 0" in df_example.columns:
            df_example = df_example.drop(columns=["Unnamed: 0"])
        if "F3924" in df_example.columns:
            df_example = df_example.drop(columns=["F3924"])
            
        expected_cols = list(df_example.columns)
        
        # Run schema validation
        valid_df, invalid_df = validate_schema(df_batch, expected_cols)
        print(f"  Schema validation: {len(valid_df)} valid, {len(invalid_df)} invalid.")
        
        # Save failed records (create empty file with schema headers if no failures)
        failed_path = os.path.join(PHASE7_DIR, "failed_records.csv")
        if not invalid_df.empty:
            invalid_df.to_csv(failed_path, index=False)
            print(f"  Saved {len(invalid_df)} invalid records to: {failed_path}")
        else:
            empty_df = pd.DataFrame(columns=expected_cols + ["failed_reason"])
            empty_df.to_csv(failed_path, index=False)
            print(f"  Saved empty failed records log to: {failed_path}")
            
        predictions = []
        reports = {}
        
        for idx, row in valid_df.iterrows():
            row_df = pd.DataFrame([row])
            res = predict_account(row_df, api_key=api_key)
            
            predictions.append({
                "account_id": res["account_id"],
                "prediction": res["prediction"],
                "probability": res["ml_probability"],
                "risk_score": res["risk_score"],
                "band": res["risk_band"]
            })
            
            # Step 7B.11: Save report only if account is not Normal (Monitor, High Risk, Critical)
            if res["risk_band"] != "Normal":
                reports[res["account_id"]] = res["report"]
                
        # Export predictions.csv
        predictions_df = pd.DataFrame(predictions)
        pred_csv_path = os.path.join(PHASE7_DIR, "predictions.csv")
        predictions_df.to_csv(pred_csv_path, index=False)
        
        # Export reports.json (only non-Normal accounts are included)
        reports_json_path = os.path.join(PHASE7_DIR, "reports.json")
        with open(reports_json_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
            
        print(f"\n[DONE] Saved predictions table to: {pred_csv_path} ({len(predictions)} rows)")
        print(f"[DONE] Saved reports JSON to: {reports_json_path} ({len(reports)} narrative reports)")
        
    else:
        parser.print_help()
