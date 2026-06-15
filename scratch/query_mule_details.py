import pandas as pd
import os

BASE_DIR = r"c:\Projects\bank_of_India\mule_account_detection"
PHASE5_DIR = os.path.join(BASE_DIR, "phase5")

df = pd.read_csv(os.path.join(PHASE5_DIR, "risk_scores.csv"))

# 1. Accounts under Monitor
monitor_df = df[df["risk_band"] == "Monitor"]
monitor_mules = monitor_df[monitor_df["target"] == 1]

print("--- Mules in Monitor Band ---")
print(f"Total accounts in Monitor band: {len(monitor_df)}")
print(f"Mule accounts in Monitor band: {len(monitor_mules)}")
print(monitor_mules[["account_id", "risk_score", "ml_score", "stat_score", "behavior_score"]])

# 2. Accounts under Normal
normal_df = df[df["risk_band"] == "Normal"]
normal_mules = normal_df[normal_df["target"] == 1]

print("\n--- Mules in Normal Band ---")
print(f"Total accounts in Normal band: {len(normal_df)}")
print(f"Mule accounts in Normal band: {len(normal_mules)}")
print(normal_mules[["account_id", "risk_score", "ml_score", "stat_score", "behavior_score"]])
