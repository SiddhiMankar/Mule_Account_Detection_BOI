import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
artifacts_dir = os.path.join(DATA_PHASE1, "images_for_eda")
os.makedirs(artifacts_dir, exist_ok=True)

print("Loading dataset from CSV...")
df = pd.read_csv(csv_path)

target_col = "F3924"
# Select numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Calculate correlations and drop NaNs
print("Calculating correlations...")
corrs = numeric_df.corrwith(df[target_col]).dropna()
corrs_sorted = corrs.sort_values(ascending=False)

# Exclude target itself
corrs_sorted = corrs_sorted.drop(labels=[target_col], errors='ignore')

print("\n--- Cleaned Top Positively Correlated Features ---")
print(corrs_sorted.head(15))

print("\n--- Cleaned Top Negatively Correlated Features ---")
print(corrs_sorted.tail(15))

# Inspect F3912 which has 0.969 correlation
print("\n--- Inspecting F3912 ---")
print(pd.crosstab(df["F3912"], df[target_col], margins=True))

# Check BOI features: F527, F115, F3889, F2082
boi_features = ["F527", "F115", "F3889", "F2082"]
print("\n--- Inspecting BOI-Highlighted Features ---")
for feat in boi_features:
    if feat in df.columns:
        if df[feat].dtype == object or df[feat].dtype == 'str':
            print(f"\nFeature {feat} (Categorical/String):")
            print(pd.crosstab(df[feat], df[target_col], normalize='index') * 100)
            print("Value counts:")
            print(df[feat].value_counts(dropna=False))
        else:
            feat_corr = corrs.get(feat, np.nan)
            print(f"\nFeature {feat} (Numeric):")
            print(f"  Correlation with target: {feat_corr:.5f}")
            print(f"  Missing %: {df[feat].isnull().mean() * 100:.2f}%")
            print(f"  Mean: {df[feat].mean():.2f}, Std: {df[feat].std():.2f}")
            print(f"  Min: {df[feat].min()}, Max: {df[feat].max()}")
    else:
        print(f"BOI Feature {feat} not found in columns!")

# Generate plots
print("\nGenerating Plots...")

# 1. Target Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x=df[target_col], palette='viridis')
plt.title("Target Distribution (F3924)")
plt.xlabel("Class (0: Normal, 1: Mule)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "target_distribution.png"), dpi=150)
plt.close()
print("Saved target_distribution.png")

# 2. Distribution of F3912
plt.figure(figsize=(6, 4))
sns.countplot(x=df["F3912"].fillna("Missing"), hue=df[target_col], palette='Set2')
plt.title("F3912 vs Target (F3924)")
plt.xlabel("F3912 Value")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "f3912_vs_target.png"), dpi=150)
plt.close()
print("Saved f3912_vs_target.png")

# 3. Distribution of top correlated features
# Let's plot F2506 and F515
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df[target_col], y=df["F2506"])
plt.title("F2506 by Target Class")
plt.xlabel("Target (F3924)")
plt.ylabel("F2506")

plt.subplot(1, 2, 2)
sns.boxplot(x=df[target_col], y=df["F515"])
plt.title("F515 by Target Class")
plt.xlabel("Target (F3924)")
plt.ylabel("F515")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "top_correlated_features.png"), dpi=150)
plt.close()
print("Saved top_correlated_features.png")

# 4. Distribution of BOI features F115, F527, F2082
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
sns.boxplot(x=df[target_col], y=df["F115"])
plt.title("F115 by Target Class")
plt.xlabel("Target (F3924)")

plt.subplot(1, 3, 2)
sns.boxplot(x=df[target_col], y=df["F527"])
plt.title("F527 by Target Class")
plt.xlabel("Target (F3924)")

plt.subplot(1, 3, 3)
sns.boxplot(x=df[target_col], y=df["F2082"])
plt.title("F2082 by Target Class")
plt.xlabel("Target (F3924)")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "boi_features_distribution.png"), dpi=150)
plt.close()
print("Saved boi_features_distribution.png")

# Save detailed stats
scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
with open(os.path.join(scratch_dir, "additional_results.pkl"), "wb") as f:
    pickle.dump({
        'top_pos_corr': corrs_sorted.head(15).to_dict(),
        'top_neg_corr': corrs_sorted.tail(15).to_dict(),
        'f3912_cross_tab': pd.crosstab(df["F3912"], df[target_col]).to_dict(),
        'boi_features_stats': {
            feat: {
                'dtype': str(df[feat].dtype),
                'missing_pct': float(df[feat].isnull().mean() * 100),
                'corr': float(corrs.get(feat, np.nan)) if df[feat].dtype != object and df[feat].dtype != 'str' else np.nan
            } for feat in boi_features if feat in df.columns
        }
    }, f)

print("Additional EDA script finished.")
