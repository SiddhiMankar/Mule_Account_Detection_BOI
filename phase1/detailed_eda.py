import os
import sys
import pandas as pd
import numpy as np
import pickle

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
print("Loading dataset from CSV...")
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape}")

# Unnamed: 0 is an index column, let's keep it in mind but exclude from feature analyses.
feature_cols = [c for c in df.columns if c not in ["Unnamed: 0", "F3924"]]
target_col = "F3924"

# Step 1.3 Target Analysis
print("\n--- Target Analysis ---")
target_vals = df[target_col].unique()
target_counts = df[target_col].value_counts()
target_pct = df[target_col].value_counts(normalize=True) * 100

print("Unique values in target:", target_vals)
print("Target value counts:")
print(target_counts)
print("Target percentages:")
print(target_pct)

# Non-numeric columns check
print("\n--- Non-Numeric Columns ---")
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Number of non-numeric columns: {len(non_numeric_cols)}")
for col in non_numeric_cols:
    print(f"Column: {col}, dtype: {df[col].dtype}, nunique: {df[col].nunique()}")
    print("Sample values:", df[col].dropna().head().tolist())

# Constant feature detection
print("\n--- Constant Features ---")
constant_cols = []
for col in feature_cols:
    if df[col].nunique(dropna=False) <= 1:
        constant_cols.append(col)
print(f"Constant features (including NaNs): {len(constant_cols)}")
# Constant features excluding NaNs (i.e. single non-null value but might have NaNs)
constant_non_nan_cols = []
for col in feature_cols:
    if df[col].nunique(dropna=True) <= 1:
        constant_non_nan_cols.append(col)
print(f"Constant features (excluding NaNs): {len(constant_non_nan_cols)}")

# Feature Categorization (Step 1.4)
# Let's categorize features excluding constant features and target
active_features = [c for c in feature_cols if c not in constant_non_nan_cols]

binary_cols = []
categorical_cols = []
continuous_cols = []

for col in active_features:
    n_uni = df[col].nunique()
    # If the column has only 2 unique values and they are numeric (e.g. 0/1)
    if n_uni == 2:
        binary_cols.append(col)
    elif df[col].dtype == object or df[col].dtype == 'str' or n_uni <= 10:
        categorical_cols.append(col)
    else:
        continuous_cols.append(col)

print(f"\nFeature Categorization (Active Features: {len(active_features)}):")
print(f"Binary Features: {len(binary_cols)}")
print(f"Categorical Features (Low Cardinality <=10): {len(categorical_cols)}")
print(f"Continuous Features: {len(continuous_cols)}")

# Basic Statistical Analysis (Step 1.5)
print("\n--- Basic Statistical Analysis ---")
desc = df[active_features].describe()
# Look for extremely large values, negative values, etc.
large_max_cols = desc.columns[desc.loc['max'] > 1e6].tolist()
negative_min_cols = desc.columns[desc.loc['min'] < 0].tolist()

print(f"Features with max > 1,000,000: {len(large_max_cols)}")
if len(large_max_cols) > 0:
    print("Sample of features with large max values:")
    for col in large_max_cols[:5]:
        print(f"  {col}: max = {desc.loc['max', col]}, mean = {desc.loc['mean', col]:.2f}")

print(f"Features with negative min values: {len(negative_min_cols)}")
if len(negative_min_cols) > 0:
    print("Sample of features with negative min values:")
    for col in negative_min_cols[:5]:
        print(f"  {col}: min = {desc.loc['min', col]}, mean = {desc.loc['mean', col]:.2f}")

# Correlation Exploration (Step 1.7)
print("\n--- Correlation Exploration ---")
# Only calculate correlation for numeric columns
numeric_cols = df[active_features].select_dtypes(include=[np.number]).columns.tolist()
corr_series = df[numeric_cols].corrwith(df[target_col])
corr_sorted = corr_series.sort_values(ascending=False)

print("\nTop 15 positively correlated features with F3924:")
print(corr_sorted.head(15))

print("\nTop 15 negatively correlated features with F3924:")
print(corr_sorted.tail(15))

# Save all results
scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
with open(os.path.join(scratch_dir, "detailed_results.pkl"), "wb") as f:
    pickle.dump({
        'target_counts': target_counts.to_dict(),
        'target_pct': target_pct.to_dict(),
        'non_numeric_cols': non_numeric_cols,
        'constant_cols': constant_cols,
        'constant_non_nan_cols': constant_non_nan_cols,
        'binary_cols': binary_cols,
        'categorical_cols': categorical_cols,
        'continuous_cols': continuous_cols,
        'corr_sorted': corr_sorted,
        'large_max_cols': large_max_cols,
        'negative_min_cols': negative_min_cols
    }, f)

print("\nDetailed EDA Completed.")
