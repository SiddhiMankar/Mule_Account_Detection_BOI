import numpy as np
import pandas as pd
import os

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE3_DIR = os.path.join(BASE_DIR, "phase3")
PHASE4_DIR = os.path.join(BASE_DIR, "phase4")
PHASE4B_DIR = os.path.join(BASE_DIR, "phase4b")

test_idx = np.load(os.path.join(PHASE3_DIR, "test_indices.npy"))
y_final = pd.read_csv(os.path.join(BASE_DIR, "y_final.csv"))
y_test = y_final.iloc[test_idx]["F3924"].values

df_4a = pd.read_csv(os.path.join(PHASE4_DIR, "anomaly_scores.csv"))
df_4b = pd.read_csv(os.path.join(PHASE4B_DIR, "behavioral_anomaly_scores.csv"))

print(f"y_test sum (mules): {y_test.sum()}")
print(f"df_4a target sum: {df_4a['target'].sum()}")
print(f"df_4b target sum: {df_4b['target'].sum()}")

# Check alignment
match_4a = np.array_equal(y_test, df_4a["target"].values)
match_4b = np.array_equal(y_test, df_4b["target"].values)
match_4a_4b = np.array_equal(df_4a["target"].values, df_4b["target"].values)

print(f"y_test aligns with df_4a target: {match_4a}")
print(f"y_test aligns with df_4b target: {match_4b}")
print(f"df_4a target aligns with df_4b target: {match_4a_4b}")
