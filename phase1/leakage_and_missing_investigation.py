import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

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
from config.paths import DATA_PHASE1

csv_path = os.path.join(DATA_PHASE1, "dataset.csv")
df = pd.read_csv(csv_path)

target_col = "F3924"
y = df[target_col]

# ----------------- Step 1: Leakage Audit -----------------
print("--- Step 1: Leakage Audit ---")
potential_leak_cols = ["F3912", "F2507", "F2506", "F2409", "F2408", "F515", "F518"]

for col in potential_leak_cols:
    if col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        nunique = df[col].nunique(dropna=False)
        corr = df[col].corr(y) if df[col].dtype != object else np.nan
        print(f"Feature: {col}")
        print(f"  Missing %: {missing_pct:.2f}%")
        print(f"  Unique Values: {df[col].unique()[:10]}")
        print(f"  Correlation with target: {corr:.5f}")
        # Crosstab
        print("  Crosstab with target:")
        print(pd.crosstab(df[col].fillna("Missing"), y, margins=True))
        print("-" * 30)

# Check if F2506 & F2507 are identical
if "F2506" in df.columns and "F2507" in df.columns:
    are_identical = df["F2506"].equals(df["F2507"])
    print(f"F2506 and F2507 are identical: {are_identical}")

# Check if F2408 & F2409 are identical
if "F2408" in df.columns and "F2409" in df.columns:
    are_identical = df["F2408"].equals(df["F2409"])
    print(f"F2408 and F2409 are identical: {are_identical}")


# ----------------- Step 3: Investigate High-Missing Columns -----------------
print("\n--- Step 3: Investigate High-Missing Columns ---")
missing_percent = (df.isnull().sum() / len(df)) * 100

cols_gt_90 = missing_percent[missing_percent > 90].index.tolist()
cols_40_90 = missing_percent[(missing_percent >= 40) & (missing_percent <= 90)].index.tolist()

print(f"Number of columns with >90% missing: {len(cols_gt_90)}")
print(f"Number of columns with 40%-90% missing: {len(cols_40_90)}")

# Check if any column in 40%-90% is highly predictive of mule accounts
# Let's compute the proportion of mule accounts in non-nulls vs nulls
predictive_sparse_cols = []
for col in cols_40_90:
    if col == target_col or col == "Unnamed: 0":
        continue
    # Let's check non-nulls
    non_null_mask = df[col].notnull()
    non_null_count = non_null_mask.sum()
    if non_null_count < 5:
        continue # too few samples
    
    # Mule rate in non-nulls
    mule_in_non_null = df.loc[non_null_mask, target_col].sum()
    mule_rate_non_null = mule_in_non_null / non_null_count
    
    # Let's also check if it's numeric and can compute correlation
    if df[col].dtype != object:
        corr_val = df.loc[non_null_mask, col].corr(df.loc[non_null_mask, target_col])
    else:
        corr_val = np.nan
        
    # Standard mule rate in overall dataset is 0.89% (81/9082)
    # If the mule rate in non-nulls is significantly higher, e.g., > 5% or has a high correlation
    if mule_rate_non_null > 0.05 or abs(corr_val) > 0.2:
        predictive_sparse_cols.append({
            'column': col,
            'missing_pct': missing_percent[col],
            'non_null_count': non_null_count,
            'mules_in_non_null': mule_in_non_null,
            'mule_rate_non_null': mule_rate_non_null * 100,
            'corr_in_non_null': corr_val
        })

print(f"Found {len(predictive_sparse_cols)} predictive sparse columns (40%-90% missing).")
if len(predictive_sparse_cols) > 0:
    print("Top 10 predictive sparse columns:")
    pred_sparse_df = pd.DataFrame(predictive_sparse_cols).sort_values(by='mule_rate_non_null', ascending=False)
    print(pred_sparse_df.head(15))


# ----------------- Step 4: Analyze Business Features -----------------
print("\n--- Step 4: Analyze Business Features ---")
business_features = ["F3886", "F3888", "F3889", "F3890", "F3891", "F3892", "F3893"]

for feat in business_features:
    if feat in df.columns:
        print(f"\nBusiness Feature: {feat}")
        if feat == "F3888":
            print("  F3888 is datetime, showing first 5 unique values:")
            print(df[feat].value_counts().head(5))
        else:
            ct = pd.crosstab(df[feat].fillna("Missing"), y, normalize="index") * 100
            ct_counts = pd.crosstab(df[feat].fillna("Missing"), y)
            ct['Mule Count'] = ct_counts[1]
            ct['Total Count'] = ct_counts[0] + ct_counts[1]
            print(ct)


# ----------------- Step 5: Investigate Datetime Columns -----------------
print("\n--- Step 5: Investigate Datetime Columns ---")
print("DateTime columns details:")
for col in ["F2230", "F3888"]:
    if col in df.columns:
        print(f"{col}: dtype: {df[col].dtype}, missing: {df[col].isnull().sum()}")

# Let's parse them
df['F2230_parsed'] = pd.to_datetime(df['F2230'], errors='coerce')
df['F3888_parsed'] = pd.to_datetime(df['F3888'], errors='coerce')

print(f"F2230 date range: {df['F2230_parsed'].min()} to {df['F2230_parsed'].max()}")
print(f"F3888 date range: {df['F3888_parsed'].min()} to {df['F3888_parsed'].max()}")

# Compute account age in days (F2230_parsed - F3888_parsed)
df['account_age_days'] = (df['F2230_parsed'] - df['F3888_parsed']).dt.days

print("\nAccount Age (Days) by Target Class:")
print(df.groupby(target_col)['account_age_days'].describe())

# Check recency/period clustering of F2230
print("\nF2230 counts by target class:")
print(pd.crosstab(df['F2230'].fillna("Missing"), y))

# Let's check F3888 year
df['F3888_year'] = df['F3888_parsed'].dt.year
print("\nF3888 Account Opening Year by target class (mule rate):")
year_ct = pd.crosstab(df['F3888_year'].fillna(-1), y, normalize="index") * 100
year_ct_counts = pd.crosstab(df['F3888_year'].fillna(-1), y)
year_ct['Mule Count'] = year_ct_counts[1]
year_ct['Total Count'] = year_ct_counts[0] + year_ct_counts[1]
print(year_ct)

# Save results for planning/reporting
import pickle
scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
with open(os.path.join(scratch_dir, "leakage_audit_results.pkl"), "wb") as f:
    pickle.dump({
        'predictive_sparse_cols': predictive_sparse_cols,
        'account_age_stats': df.groupby(target_col)['account_age_days'].describe().to_dict(),
        'f2230_crosstab': pd.crosstab(df['F2230'].fillna("Missing"), y).to_dict(),
        'f3888_year_crosstab': year_ct.to_dict()
    }, f)

print("\nResearch Finished.")
