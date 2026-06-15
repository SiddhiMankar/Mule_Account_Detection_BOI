import pandas as pd
import os

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"

print("Checking dataset.csv columns...")
df_raw = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"), nrows=5)
print("dataset.csv columns:", list(df_raw.columns)[:15])

print("\nChecking dataset_cleaned.csv columns...")
df_clean = pd.read_csv(os.path.join(BASE_DIR, "dataset_cleaned.csv"), nrows=5)
print("dataset_cleaned.csv columns:", list(df_clean.columns)[:15])
