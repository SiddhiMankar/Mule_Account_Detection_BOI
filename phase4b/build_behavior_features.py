"""
build_behavior_features.py
--------------------------
Phase 4B: Behavioral Feature Dataset Generation
Bank of India -- Mule Account Detection

This script loads dataset_cleaned.csv, aligns the rows using the single random_state=42 shuffle,
splits into train and test portions using Phase 3 indices, imputes raw columns, computes target-based
risk encodings strictly on train, engineers ratio features, fits RobustScaler on train, and outputs
behavioral_features.csv and behavior_scaler.pkl.
"""

import os
import sys
import joblib

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
from config.paths import DATA_PHASE2, PHASE3_DIR, PHASE4B_DIR
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Setup directories
BASE_DIR   = str(PROJECT_ROOT)
# PHASE3_DIR and PHASE4B_DIR are imported from config.paths
os.makedirs(PHASE4B_DIR, exist_ok=True)

print("=" * 60)
print("Step 4B.1 -- Generating Behavioral Feature Dataset")
print("=" * 60)

print("\nLoading dataset_cleaned.csv...")
df = pd.read_csv(os.path.join(DATA_PHASE2, "dataset_cleaned.csv"))
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\nRestoring Phase 3 row alignment...")
# Match the exact random state and row order from preprocess_pipeline.py
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("  Alignment complete.")

# 2. Split Using Phase 3 Indices
print("\nLoading train/test indices...")
train_idx = np.load(os.path.join(PHASE3_DIR, "train_indices.npy"))
test_idx  = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))

print(f"  Train set size: {len(train_idx)} rows")
print(f"  Test set size : {len(test_idx)} rows")

df_train = df_shuffled.iloc[train_idx].copy()
df_test  = df_shuffled.iloc[test_idx].copy()

# 3. Parse Account Opening Date (F3888) to Account Age
print("\nEngineering account age features from F3888...")
ref_date = pd.to_datetime('2025-12-31')

# Train set
opening_date_train = pd.to_datetime(df_train['F3888'], format='mixed', errors='coerce')
df_train['account_age_days'] = (ref_date - opening_date_train).dt.days
df_train['account_age_years'] = df_train['account_age_days'] / 365.25

# Test set
opening_date_test = pd.to_datetime(df_test['F3888'], format='mixed', errors='coerce')
df_test['account_age_days'] = (ref_date - opening_date_test).dt.days
df_test['account_age_years'] = df_test['account_age_days'] / 365.25

# 4. Impute Raw Continuous Columns
print("\nImputing continuous columns with training median...")
raw_continuous_cols = ['F2737', 'F2678', 'F3836', 'account_age_days', 'account_age_years']
for col in raw_continuous_cols:
    train_median = df_train[col].median()
    if pd.isna(train_median):
        train_median = 0.0
    print(f"  Feature '{col}' train median: {train_median:.4f}")
    df_train[col] = df_train[col].fillna(train_median)
    df_test[col] = df_test[col].fillna(train_median)

# 5. Target-based Risk Encodings (Leakage-free)
print("\nGenerating target-based risk encodings...")
categorical_risk_mapping = [
    ('F3891', 'occupation_risk_score'),
    ('F3890', 'area_risk_score'),
    ('F3886', 'account_type_risk_score'),
    ('F3893', 'customer_segment_risk_score'),
    ('F3892', 'gender_risk_score')
]

for orig_col, encoded_col in categorical_risk_mapping:
    # Impute missing categories with train mode
    train_mode = df_train[orig_col].mode()[0] if not df_train[orig_col].mode().empty else 'Missing'
    df_train[orig_col] = df_train[orig_col].fillna(train_mode)
    df_test[orig_col]  = df_test[orig_col].fillna(train_mode)
    
    # Compute encoding: P(mule | category) on train only
    train_counts = df_train.groupby(orig_col)['F3924'].count()
    train_mules  = df_train.groupby(orig_col)['F3924'].sum()
    
    # Target mean
    train_encoding = train_mules / train_counts
    global_train_mule_rate = df_train['F3924'].mean()
    
    print(f"  Encoding '{orig_col}' -> '{encoded_col}' (Global baseline: {global_train_mule_rate*100:.4f}%)")
    for val, rate in train_encoding.items():
        print(f"    Category '{val}': {rate*100:.4f}% ({train_mules[val]} mules in {train_counts[val]} accounts)")
        
    # Apply encoding to train/test, fallback to global baseline for unseen
    df_train[encoded_col] = df_train[orig_col].map(train_encoding).fillna(global_train_mule_rate)
    df_test[encoded_col]  = df_test[orig_col].map(train_encoding).fillna(global_train_mule_rate)

# 6. Engineer Ratio Features
print("\nEngineering ratio features...")
eps = 1e-6

# Train ratios
df_train['credit_debit_ratio'] = (df_train['F2737'].abs() + eps) / (df_train['F2678'].abs() + eps)
df_train['balance_retention_ratio'] = df_train['F3836'].abs() / (df_train['F2737'].abs() + eps)
df_train['pass_through_ratio'] = df_train['F2678'].abs() / (df_train['F2737'].abs() + eps)

# Test ratios
df_test['credit_debit_ratio'] = (df_test['F2737'].abs() + eps) / (df_test['F2678'].abs() + eps)
df_test['balance_retention_ratio'] = df_test['F3836'].abs() / (df_test['F2737'].abs() + eps)
df_test['pass_through_ratio'] = df_test['F2678'].abs() / (df_test['F2737'].abs() + eps)

print("  Ratio features generated successfully.")

# 7. Robust Scaling and Export
behavioral_cols = [
    'account_age_days', 'account_age_years',
    'occupation_risk_score', 'area_risk_score', 'account_type_risk_score',
    'customer_segment_risk_score', 'gender_risk_score',
    'credit_debit_ratio', 'balance_retention_ratio', 'pass_through_ratio'
]

print(f"\nScaling behavioral features (count={len(behavioral_cols)})...")
scaler = RobustScaler()

# Fit only on train and transform
train_scaled = scaler.fit_transform(df_train[behavioral_cols])
test_scaled  = scaler.transform(df_test[behavioral_cols])

# Reassemble full behavioral dataframe, maintaining split order and indices
df_behavioral = pd.DataFrame(index=df_shuffled.index)
df_behavioral['target'] = df_shuffled['F3924']

# Populate scaled values
df_behavioral_train = pd.DataFrame(train_scaled, columns=behavioral_cols, index=train_idx)
df_behavioral_test  = pd.DataFrame(test_scaled, columns=behavioral_cols, index=test_idx)

df_behavioral = pd.concat([df_behavioral['target'], pd.concat([df_behavioral_train, df_behavioral_test]).sort_index()], axis=1)

print(f"\nFinal Behavioral Feature Dataset Shape: {df_behavioral.shape}")
print(f"Checking null values in exported dataset:\n{df_behavioral.isnull().sum()}")

# Save scaler and dataset
scaler_path = os.path.join(PHASE4B_DIR, "behavior_scaler.pkl")
dataset_path = os.path.join(PHASE4B_DIR, "behavioral_features.csv")

joblib.dump(scaler, scaler_path)
df_behavioral.to_csv(dataset_path, index=False)

print(f"\n[DONE] Saved behavior scaler to: {scaler_path}")
print(f"[DONE] Saved behavioral features to: {dataset_path}")
print("=" * 60)
