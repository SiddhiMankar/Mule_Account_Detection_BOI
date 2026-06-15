import os
import sys
import pandas as pd
import time

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

start_time = time.time()
print("Starting Excel conversion...")

xlsx_path = os.path.join(DATA_PHASE1, "DataSet.xlsx")
csv_path = os.path.join(DATA_PHASE1, "dataset.csv")

# Load spreadsheet
print("Reading Excel file (this might take a couple of minutes)...")
excel_file = pd.ExcelFile(xlsx_path, engine='openpyxl')
print("Sheet names in Excel:", excel_file.sheet_names)

df = excel_file.parse(excel_file.sheet_names[0])
print(f"Loaded DataFrame with shape: {df.shape}")

# Save as CSV
print("Saving to CSV...")
df.to_csv(csv_path, index=False)
print(f"Saved to CSV successfully at {csv_path}")
print(f"Time taken for Excel to CSV conversion: {time.time() - start_time:.2f} seconds")

# Run Step 1.1 Tasks
print("\n--- Step 1.1 Dataset Audit ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())

print("\nData type counts:")
print(df.dtypes.value_counts())

# Duplicates
print("\nChecking duplicates...")
duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    df.to_csv(csv_path, index=False)
    print("Duplicates removed and CSV updated.")
    
# Missing value analysis
print("\n--- Step 1.2 Missing Value Analysis ---")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100

missing_summary = pd.DataFrame({
    'missing_count': missing_counts,
    'missing_percent': missing_percent
})

# Missing categories
keep_cols = missing_summary[missing_summary['missing_percent'] == 0].index.tolist()
impute_cols = missing_summary[(missing_summary['missing_percent'] > 0) & (missing_summary['missing_percent'] < 10)].index.tolist()
investigate_cols = missing_summary[(missing_summary['missing_percent'] >= 10) & (missing_summary['missing_percent'] <= 40)].index.tolist()
remove_cols = missing_summary[missing_summary['missing_percent'] > 40].index.tolist()

print(f"Keep cols (0% missing): {len(keep_cols)}")
print(f"Impute cols (<10% missing): {len(impute_cols)}")
print(f"Investigate cols (10%-40% missing): {len(investigate_cols)}")
print(f"Candidate for removal cols (>40% missing): {len(remove_cols)}")

if len(remove_cols) > 0:
    print(f"Sample of cols with >40% missing (showing top 10): {remove_cols[:10]}")
if len(investigate_cols) > 0:
    print(f"Sample of cols with 10%-40% missing (showing top 10): {investigate_cols[:10]}")

# Save these results to a temporary file
import pickle
scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
with open(os.path.join(scratch_dir, "audit_results.pkl"), "wb") as f:
    pickle.dump({
        'shape': df.shape,
        'dtypes_val_counts': df.dtypes.value_counts().to_dict(),
        'duplicates': duplicates,
        'missing_summary': missing_summary,
        'keep_cols': keep_cols,
        'impute_cols': impute_cols,
        'investigate_cols': investigate_cols,
        'remove_cols': remove_cols
    }, f)

print("\nAudit completed.")
