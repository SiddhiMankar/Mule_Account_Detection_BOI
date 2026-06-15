"""
preprocess_pipeline.py
-----------------------
Master script to execute the Phase 2 Feature Engineering & Data Preprocessing tasks.
Produces all required deliverables and reports.
"""

import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

import sys

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
from config.paths import DATA_PHASE1, DATA_PHASE2

# Import the custom preprocessor (using package absolute import)
from phase2.mule_preprocessor import MuleAccountPreprocessor

start_time = time.time()

# Define file paths
data_dir = str(DATA_PHASE2)
dataset_path = os.path.join(data_dir, "dataset_cleaned.csv")

print("--- Step 2.12: Loading and Shuffling Dataset ---")
df = pd.read_csv(dataset_path)
print(f"Original shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Shuffle dataset before any split
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Dataset shuffled successfully.")

# Target and features separation
target_col = "F3924"
X_raw = df_shuffled.drop(columns=[target_col])
y_shuffled = df_shuffled[target_col]

# Save target early as y_final.csv
y_shuffled.to_csv(os.path.join(data_dir, "y_final.csv"), index=False)
print(f"Saved target as y_final.csv. Shape: {y_shuffled.shape}")

print("\n--- Step 2.1: Generating Final Feature Inventory ---")
# Temporary date parsing for inventory details
opening_date_parsed = pd.to_datetime(X_raw['F3888'], format='mixed', errors='coerce')
account_age_days_temp = (pd.to_datetime('2025-12-31') - opening_date_parsed).dt.days
account_age_years_temp = account_age_days_temp / 365.25

inventory_records = []
categorical_list = ['F3886', 'F3889', 'F3890', 'F3891', 'F3892', 'F3893']

for col in X_raw.columns:
    missing_pct = (X_raw[col].isnull().sum() / len(X_raw)) * 100
    nunique = X_raw[col].nunique(dropna=True)
    
    # Classify Type
    if col in categorical_list:
        col_type = "Categorical"
    elif col == 'F3888':
        col_type = "Date (Raw)"
    elif nunique <= 2:
        col_type = "Binary"
    else:
        col_type = "Continuous"
        
    inventory_records.append({
        'Feature': col,
        'Type': col_type,
        'Missing %': round(missing_pct, 4),
        'Unique Values': nunique
    })

# Add the engineered features to inventory (since they will be created)
inventory_records.append({
    'Feature': 'account_age_days',
    'Type': 'Continuous',
    'Missing %': round((account_age_days_temp.isnull().sum() / len(X_raw)) * 100, 4),
    'Unique Values': account_age_days_temp.nunique(dropna=True)
})
inventory_records.append({
    'Feature': 'account_age_years',
    'Type': 'Continuous',
    'Missing %': round((account_age_years_temp.isnull().sum() / len(X_raw)) * 100, 4),
    'Unique Values': account_age_years_temp.nunique(dropna=True)
})

inventory_df = pd.DataFrame(inventory_records)
inventory_df.to_csv(os.path.join(data_dir, "feature_inventory.csv"), index=False)
print(f"Generated feature_inventory.csv with {len(inventory_df)} features.")

print("\n--- Step 2.4: Analyzing Remaining Missing Values ---")
total_features = len(X_raw.columns) - 1 + 2 # minus raw F3888, plus 2 age columns
# Categorize based on inventory
missing_0 = len(inventory_df[inventory_df['Missing %'] == 0])
missing_0_5 = len(inventory_df[(inventory_df['Missing %'] > 0) & (inventory_df['Missing %'] <= 5)])
missing_5_20 = len(inventory_df[(inventory_df['Missing %'] > 5) & (inventory_df['Missing %'] <= 20)])
missing_20_40 = len(inventory_df[(inventory_df['Missing %'] > 20) & (inventory_df['Missing %'] <= 40)])

print(f"0% missing: {missing_0}")
print(f"0-5% missing: {missing_0_5}")
print(f"5-20% missing: {missing_5_20}")
print(f"20-40% missing: {missing_20_40}")

report_content = f"""# Missing Value Distribution Report

This report audits the missing values in the remaining features of the dataset (after dropping high-missing features above 40%).

## Missing Value Summary Table

| Missing % Range | Number of Columns | Percentage of Features | Imputation Strategy |
| :--- | :---: | :---: | :--- |
| **0%** (Complete) | {missing_0} | {missing_0 / total_features * 100:.2f}% | None |
| **0% – 5%** | {missing_0_5} | {missing_0_5 / total_features * 100:.2f}% | Median (Continuous) / Most Frequent (Binary & Cat) |
| **5% – 20%** | {missing_5_20} | {missing_5_20 / total_features * 100:.2f}% | Median (Continuous) / Most Frequent (Binary & Cat) |
| **20% – 40%** | {missing_20_40} | {missing_20_40 / total_features * 100:.2f}% | Median (Continuous) / Most Frequent (Binary & Cat) |
| **Total Features** | **{total_features}** | **100.00%** | |

## Key Insights
- **No high missingness remains**: The maximum missing percentage in the remaining features is **{inventory_df['Missing %'].max()}%**, which is well below our 40% threshold.
- **High completeness**: Approximately **{missing_0 / total_features * 100:.2f}%** of features are completely non-missing, minimizing imputation distortion.
- **Imputation selection**: A ColumnTransformer utilizing median imputation for continuous features and most-frequent imputation for binary and categorical features will be used.
"""

with open(os.path.join(DATA_PHASE1, "missing_distribution_report.md"), "w", encoding="utf-8") as f:
    f.write(report_content)
print("Saved missing_distribution_report.md to phase1/")

print("\n--- Fit Preprocessor (Initial Run up to Scaling) ---")
# We initialize our custom preprocessor
preprocessor = MuleAccountPreprocessor(ref_date='2025-12-31')
# Let's fit it on the raw X data
preprocessor.fit(X_raw)

# Extract fitted imputer and scaler for deliverables
joblib.dump(preprocessor.imputer, os.path.join(data_dir, "imputer.pkl"))
joblib.dump(preprocessor.scaler, os.path.join(data_dir, "scaler.pkl"))
print("Saved imputer.pkl and scaler.pkl.")

# Transform the raw dataset to get the imputed, encoded, scaled version (before feature selection)
# Note: Since preprocessor.redundant_cols and preprocessor.selected_features are empty lists,
# it will return the full encoded, scaled matrix.
X_transformed = preprocessor.transform(X_raw)
print(f"Transformed feature matrix shape (pre-redundancy/selection): {X_transformed.shape}")

print("\n--- Step 2.8: Identifying and Removing Redundant Features ---")
# Compute correlation matrix of the transformed numerical feature matrix
print("Computing correlation matrix (this might take a moment)...")
corr_matrix = X_transformed.corr().abs()

# Find upper triangle values with corr > 0.95
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = []
redundancy_details = []

# Rank features using absolute correlation with target as a proxy for feature strength
target_corr = X_transformed.corrwith(y_shuffled).abs()

for col in upper_tri.columns:
    high_corr_cols = upper_tri.index[upper_tri[col] > 0.95].tolist()
    for hc_col in high_corr_cols:
        corr_val = upper_tri.loc[hc_col, col]
        col_target_corr = target_corr[col]
        hc_col_target_corr = target_corr[hc_col]
        
        if col_target_corr >= hc_col_target_corr:
            keep = col
            drop = hc_col
        else:
            keep = hc_col
            drop = col
            
        redundancy_details.append({
            'Feature 1': keep,
            'Feature 2': drop,
            'Correlation': corr_val,
            'Keep': keep,
            'Drop': drop,
            'Keep Target Corr': col_target_corr if keep == col else hc_col_target_corr,
            'Drop Target Corr': hc_col_target_corr if keep == col else col_target_corr
        })
        if drop not in to_drop:
            to_drop.append(drop)

redundancy_df = pd.DataFrame(redundancy_details)
print(f"Found {len(to_drop)} redundant features out of {len(redundancy_df)} pairs.")

# Write redundancy report
redundancy_report = f"""# Redundancy Report

This report documents pairs of numerical features with Pearson correlation coefficients greater than **0.95**. To prevent collinearity and reduce feature dimension, the weaker feature in each pair (the one with the lower correlation to the target `F3924`) is dropped.

## Summary
- **Total Highly Correlated Pairs (>0.95)**: {len(redundancy_df)}
- **Unique Columns Dropped**: {len(to_drop)}
- **Unique Columns Retained**: {X_transformed.shape[1] - len(to_drop)}

## Redundant Pairs Details (Top 25 Pairs)

| Keep Feature | Drop Feature | Correlation | Keep Target Corr | Drop Target Corr |
| :--- | :--- | :---: | :---: | :---: |
"""

for _, row in redundancy_df.sort_values(by='Correlation', ascending=False).head(25).iterrows():
    redundancy_report += f"| `{row['Feature 1']}` | `{row['Feature 2']}` | {row['Correlation']:.4f} | {row['Keep Target Corr']:.4f} | {row['Drop Target Corr']:.4f} |\n"

with open(os.path.join(data_dir, "redundancy_report.md"), "w", encoding="utf-8") as f:
    f.write(redundancy_report)
print("Saved redundancy_report.md")

# Drop redundant features from transformed matrix
X_clean = X_transformed.drop(columns=to_drop)
print(f"Shape after removing redundant features: {X_clean.shape}")

# Update the preprocessor's redundant list
preprocessor.redundant_cols = to_drop

print("\n--- Step 2.6: Feature Importance Screening ---")
# Fit random forest to get feature importance
print("Training quick RandomForestClassifier for feature screening...")
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced', 
    n_jobs=-1
)
rf.fit(X_clean, y_shuffled)
rf_importances = rf.feature_importances_

print("Computing Mutual Information Scores (this might take 1-2 minutes)...")
mi_scores = mutual_info_classif(X_clean, y_shuffled, random_state=42)

feature_importance_df = pd.DataFrame({
    'Feature': X_clean.columns,
    'RF Importance': rf_importances,
    'MI Score': mi_scores
})

# Sort by RF Importance descending
feature_importance_df = feature_importance_df.sort_values(by='RF Importance', ascending=False)
feature_importance_df.to_csv(os.path.join(data_dir, "feature_importance.csv"), index=False)
print("Saved feature_importance.csv.")

print("\n--- Step 2.7: Auditing BOI Highlighted Features ---")
boi_feats = [
    'F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122', 'F2582', 
    'F2678', 'F2737', 'F2956', 'F3043', 'F3836', 'F3887', 'F3889', 'F3891', 'F3894'
]

# We will analyze these features
boi_report = """# Bank of India Highlighted Features Audit

This report details the characteristics and predictive signal of the 18 specific features highlighted by the Bank of India.

## Master Summary Table

| Feature | Type | Missing % | Target Correlation | RF Importance Rank | MI Rank | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
"""

for feat in boi_feats:
    # Check if the feature is in the original raw dataframe
    if feat not in X_raw.columns:
        # Check if it was in the dropped columns log
        boi_report += f"| `{feat}` | - | >40% | - | - | - | ❌ Dropped in Phase 1 (Missingness > 40%) |\n"
        continue
        
    raw_missing = (X_raw[feat].isnull().sum() / len(X_raw)) * 100
    
    # Classify Type
    if feat in categorical_list:
        feat_type = "Categorical"
        # Since it is one-hot encoded, let's look at its encoded versions
        encoded_versions = [c for c in X_clean.columns if c.startswith(preprocessor._get_prefix(feat) if hasattr(preprocessor, '_get_prefix') else feat.lower()) or c.startswith(feat.lower()) or c.startswith('account_type' if feat=='F3886' else 'historical_code' if feat=='F3889' else 'area_category' if feat=='F3890' else 'occupation' if feat=='F3891' else 'gender' if feat=='F3892' else 'customer_segment' if feat=='F3893' else feat.lower())]
        
        # We can report max correlation and rank among its encoded parts
        max_corr = 0
        best_rf_rank = 9999
        best_mi_rank = 9999
        
        for enc in encoded_versions:
            if enc in X_clean.columns:
                corr_val = abs(X_clean[enc].corr(y_shuffled))
                max_corr = max(max_corr, corr_val)
                # Find rank
                rf_rank = feature_importance_df['Feature'].tolist().index(enc) + 1
                best_rf_rank = min(best_rf_rank, rf_rank)
                
                mi_sorted = feature_importance_df.sort_values(by='MI Score', ascending=False)['Feature'].tolist()
                mi_rank = mi_sorted.index(enc) + 1
                best_mi_rank = min(best_mi_rank, mi_rank)
                
        status = "✅ Kept & Encoded"
        boi_report += f"| `{feat}` | {feat_type} | {raw_missing:.2f}% | {max_corr:.4f} (max) | {best_rf_rank} (best) | {best_mi_rank} (best) | {status} |\n"
        
    else:
        # Continuous/Binary
        feat_type = "Continuous" if feat in preprocessor.continuous_cols else "Binary"
        
        # Check if kept
        if feat in X_clean.columns:
            corr_val = abs(X_clean[feat].corr(y_shuffled))
            rf_rank = feature_importance_df['Feature'].tolist().index(feat) + 1
            mi_sorted = feature_importance_df.sort_values(by='MI Score', ascending=False)['Feature'].tolist()
            mi_rank = mi_sorted.index(feat) + 1
            status = "✅ Kept"
            boi_report += f"| `{feat}` | {feat_type} | {raw_missing:.2f}% | {corr_val:.4f} | {rf_rank} | {mi_rank} | {status} |\n"
        else:
            status = "❌ Dropped (Redundant)"
            boi_report += f"| `{feat}` | {feat_type} | {raw_missing:.2f}% | - | - | - | {status} |\n"

# Add detailed profiles for selected key BOI features
boi_report += """
## Detailed Class Distribution Profiles

Below are the class-specific statistical profiles for key continuous BOI features that show predictive signal.

"""

key_boi_numeric = [f for f in boi_feats if f in X_clean.columns and f not in categorical_list]
for feat in key_boi_numeric[:5]: # Show top 5 numeric BOI features
    stats = X_clean.groupby(y_shuffled)[feat].describe()
    boi_report += f"### Feature `{feat}` Profile\n\n"
    boi_report += f"| Target Class | Count | Mean | Std | Min | Median (50%) | Max |\n"
    boi_report += f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    boi_report += f"| **Normal (0)** | {stats.loc[0, 'count']:.0f} | {stats.loc[0, 'mean']:.4f} | {stats.loc[0, 'std']:.4f} | {stats.loc[0, 'min']:.4f} | {stats.loc[0, '50%']:.4f} | {stats.loc[0, 'max']:.4f} |\n"
    boi_report += f"| **Mule (1)** | {stats.loc[1, 'count']:.0f} | {stats.loc[1, 'mean']:.4f} | {stats.loc[1, 'std']:.4f} | {stats.loc[1, 'min']:.4f} | {stats.loc[1, '50%']:.4f} | {stats.loc[1, 'max']:.4f} |\n\n"

with open(os.path.join(DATA_PHASE1, "boi_feature_report.md"), "w", encoding="utf-8") as f:
    f.write(boi_report)
print("Saved boi_feature_report.md to phase1/")


print("\n--- Step 2.10: Comparative Feature Selection ---")
# Perform Stratified Train-Validation split to evaluate feature subsets
X_train, X_val, y_train, y_val = train_test_split(
    X_clean, 
    y_shuffled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_shuffled
)
print(f"Train split shape: {X_train.shape}, Validation split shape: {X_val.shape}")

# Evaluate feature selection candidate sizes K = 300, 400, 500
selection_results = []

for K in [300, 400, 500]:
    print(f"\nEvaluating K = {K} Features:")
    
    # Method A: Top K Mutual Information
    top_k_mi = feature_importance_df.sort_values(by='MI Score', ascending=False).head(K)['Feature'].tolist()
    # Ensure they are in X_train
    top_k_mi = [c for c in top_k_mi if c in X_train.columns]
    
    rf_mi = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_mi.fit(X_train[top_k_mi], y_train)
    val_preds_mi = rf_mi.predict(X_val[top_k_mi])
    
    mi_rec = recall_score(y_val, val_preds_mi)
    mi_prec = precision_score(y_val, val_preds_mi, zero_division=0)
    mi_f1 = f1_score(y_val, val_preds_mi, zero_division=0)
    
    print(f"  Method A (MI): Recall = {mi_rec:.4f}, Precision = {mi_prec:.4f}, F1 = {mi_f1:.4f}")
    selection_results.append({'K': K, 'Method': 'Mutual Information', 'Recall': mi_rec, 'Precision': mi_prec, 'F1-Score': mi_f1, 'Features': top_k_mi})
    
    # Method B: Top K Random Forest Importance
    top_k_rf = feature_importance_df.sort_values(by='RF Importance', ascending=False).head(K)['Feature'].tolist()
    top_k_rf = [c for c in top_k_rf if c in X_train.columns]
    
    rf_rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_rf.fit(X_train[top_k_rf], y_train)
    val_preds_rf = rf_rf.predict(X_val[top_k_rf])
    
    rf_rec = recall_score(y_val, val_preds_rf)
    rf_prec = precision_score(y_val, val_preds_rf, zero_division=0)
    rf_f1 = f1_score(y_val, val_preds_rf, zero_division=0)
    
    print(f"  Method B (RF): Recall = {rf_rec:.4f}, Precision = {rf_prec:.4f}, F1 = {rf_f1:.4f}")
    selection_results.append({'K': K, 'Method': 'Random Forest', 'Recall': rf_rec, 'Precision': rf_prec, 'F1-Score': rf_f1, 'Features': top_k_rf})

# Method C: L1 Logistic Regression Feature Selection
print("\nRunning Method C: L1 Logistic Regression...")
# Fit L1 Logistic Regression on train split, tune C to get feature count between 300 and 500
best_l1_feats = None
best_l1_rec = 0
best_l1_f1 = 0
best_l1_prec = 0
best_l1_c = None

for C_val in [0.05, 0.1, 0.2, 0.5]:
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=C_val, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    non_zero_coefs = (lr.coef_ != 0).sum()
    print(f"  For C = {C_val:.2f}: Non-zero coefficients = {non_zero_coefs}")
    
    if 100 <= non_zero_coefs <= 600: # allow reasonable range for evaluation
        # Extract features
        l1_feats = X_clean.columns[lr.coef_[0] != 0].tolist()
        
        # Evaluate using RF Classifier for consistency
        rf_l1 = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_l1.fit(X_train[l1_feats], y_train)
        val_preds_l1 = rf_l1.predict(X_val[l1_feats])
        
        l1_rec = recall_score(y_val, val_preds_l1)
        l1_prec = precision_score(y_val, val_preds_l1, zero_division=0)
        l1_f1 = f1_score(y_val, val_preds_l1, zero_division=0)
        
        print(f"    RF Evaluation on L1 features: Recall = {l1_rec:.4f}, Precision = {l1_prec:.4f}, F1 = {l1_f1:.4f}")
        selection_results.append({'K': non_zero_coefs, 'Method': f'L1 Logistic Regression (C={C_val})', 'Recall': l1_rec, 'Precision': l1_prec, 'F1-Score': l1_f1, 'Features': l1_feats})
        
        # Track the best L1 feature set
        if l1_f1 >= best_l1_f1:
            best_l1_f1 = l1_f1
            best_l1_rec = l1_rec
            best_l1_prec = l1_prec
            best_l1_feats = l1_feats
            best_l1_c = C_val

# Determine the best feature selection method (highest Recall, breaking ties with F1)
print("\n--- Feature Selection Results Summary ---")
selection_df = pd.DataFrame(selection_results)
print(selection_df[['K', 'Method', 'Recall', 'Precision', 'F1-Score']].to_string())

# Find the best result
best_idx = selection_df['Recall'].idxmax()
best_method_row = selection_df.loc[best_idx]
print(f"\nBest Method Chosen: {best_method_row['Method']} (K = {best_method_row['K']})")
print(f"Validation Recall: {best_method_row['Recall']:.4f}, F1: {best_method_row['F1-Score']:.4f}")

# Extract final selected features
final_selected_features = best_method_row['Features']

# Ensure we limit to 300-500 features. If K is outside this range, we take Top 400 from the chosen method rank
if len(final_selected_features) > 500:
    print(f"Trimming feature count from {len(final_selected_features)} down to 450 based on importance rankings...")
    if 'Mutual Information' in best_method_row['Method']:
        final_selected_features = feature_importance_df.sort_values(by='MI Score', ascending=False).head(450)['Feature'].tolist()
    else:
        final_selected_features = feature_importance_df.sort_values(by='RF Importance', ascending=False).head(450)['Feature'].tolist()
elif len(final_selected_features) < 300:
    print(f"Extending feature count from {len(final_selected_features)} up to 350 based on importance rankings...")
    if 'Mutual Information' in best_method_row['Method']:
        final_selected_features = feature_importance_df.sort_values(by='MI Score', ascending=False).head(350)['Feature'].tolist()
    else:
        final_selected_features = feature_importance_df.sort_values(by='RF Importance', ascending=False).head(350)['Feature'].tolist()

# Final validation evaluation
final_selected_features = [c for c in final_selected_features if c in X_clean.columns]
print(f"Final feature count selected for export: {len(final_selected_features)}")

print("\n--- Step 2.11: Building Final Preprocessing Pipeline & Modeling Datasets ---")
# 1. Update preprocessor with final feature list
preprocessor.selected_features = final_selected_features

# 2. Re-fit and transform the entire dataset to ensure complete fitting on all 9082 rows
preprocessor.fit(X_raw)
X_final = preprocessor.transform(X_raw)

print(f"Final features shape (X_final): {X_final.shape}")
print(f"Final target shape (y_final): {y_shuffled.shape}")

# Save model-ready features as X_final.csv
X_final.to_csv(os.path.join(data_dir, "X_final.csv"), index=False)
print("Saved X_final.csv successfully.")

# Save the complete integrated preprocessor pipeline
joblib.dump(preprocessor, os.path.join(data_dir, "preprocessing_pipeline.pkl"))
print("Saved preprocessing_pipeline.pkl successfully.")

# Final check/verification
print("\n--- Final Pipeline Verification ---")
# Reload pipeline and check transformation
pipeline_reload = joblib.load(os.path.join(data_dir, "preprocessing_pipeline.pkl"))
X_test_transform = pipeline_reload.transform(X_raw.head(5))
print(f"Verification reload transform successful. Out shape: {X_test_transform.shape}")
print(f"First 5 rows features check:\n{X_test_transform.columns.tolist()[:10]}...")

elapsed = time.time() - start_time
print(f"\nAll Phase 2 tasks completed successfully in {elapsed:.1f} seconds!")
