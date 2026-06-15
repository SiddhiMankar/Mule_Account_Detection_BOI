import os
import sys
import subprocess

def check_artifacts():
    required_files = [
        "phase2/preprocessing_pipeline.pkl",
        "phase3/best_model.pkl",
        "phase4/isolation_forest.pkl",
        "phase4/isolation_forest_scaler.pkl",
        "phase4b/behavioral_lof.pkl",
        "phase4b/behavior_scaler.pkl",
        "phase5/risk_scores.csv",
        "phase6/feature_dictionary.json",
        "phase7/investigation_queue.csv"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    return missing

def main():
    print("🏦 BOI Fraud Intelligence System Launcher 🏦")
    print("============================================")
    
    missing_artifacts = check_artifacts()
    
    if missing_artifacts:
        print(f"⚠️ Found {len(missing_artifacts)} missing model/data artifacts.")
        print("Checking for dataset to start pipeline retraining...")
        
        has_xlsx = os.path.exists("phase1/DataSet.xlsx")
        has_csv = os.path.exists("phase1/dataset.csv")
        
        if not has_csv and has_xlsx:
            print("  Found phase1/DataSet.xlsx. Converting to CSV first...")
            subprocess.run([sys.executable, "phase1/convert_and_audit.py"], check=True)
            has_csv = True
            
        if not has_csv:
            print("\n❌ Error: Neither phase1/dataset.csv nor phase1/DataSet.xlsx was found.")
            print("Please place the dataset file in the phase1/ directory and try again.")
            sys.exit(1)
            
        print("\n🚀 Starting pipeline training sequence...")
        pipeline_steps = [
            ("Phase 2: Preprocessing & Clean", "phase2/preprocess_pipeline.py"),
            ("Phase 3: Model Training & Tuning", "phase3/train_model.py"),
            ("Phase 4: Statistical Anomaly Detection", "phase4/anomaly_detection.py"),
            ("Phase 4b: Behavioral Feature Prep", "phase4b/build_behavior_features.py"),
            ("Phase 4b: Behavioral Outliers", "phase4b/behavioral_anomaly_detection.py"),
            ("Phase 5: Score Fusion Engine (ML)", "phase5/generate_ml_scores.py"),
            ("Phase 5: Score Fusion Calibration", "phase5/generate_risk_scores.py"),
            ("Phase 6: Explainability (SHAP)", "phase6/generate_explanations.py"),
            ("Phase 7: GenAI narrative Generation", "phase7/generate_genai_reports.py")
        ]
        
        for name, script in pipeline_steps:
            print(f"\n▶️ Running {name} ({script})...")
            result = subprocess.run([sys.executable, script])
            if result.returncode != 0:
                print(f"\n❌ Step failed: {script}. Aborting startup.")
                sys.exit(1)
        
        print("\n✅ All pipeline stages trained and artifacts saved successfully!")
    else:
        print("⚡ All model artifacts are up-to-date. Skipping retraining.")
        
    print("\n🖥️ Starting Streamlit Fraud Intelligence Dashboard...")
    try:
        subprocess.run(["streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")

if __name__ == "__main__":
    main()
