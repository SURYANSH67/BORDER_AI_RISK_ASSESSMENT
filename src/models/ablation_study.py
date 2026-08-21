import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def run_ablation_study(master_dataset_path="data/processed/master_risk_dataset.csv",
                       output_path="models/ablation_study_results.csv",
                       seed=42):
    """
    Executes a 4-Stage Feature Modality Ablation Study to empirically measure
    how incremental intelligence layers (Environmental -> Operational -> Historical -> Camera & Sensors)
    impact predictive accuracy and multi-class F1-score.
    """
    print(f"[Ablation Study] Running multi-modal feature ablation study on {master_dataset_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(master_dataset_path)

    # Feature groups
    env_cols = [
        "Avg_Elevation", "Min_Elevation", "Max_Elevation", "Avg_Slope", "Max_Slope", "Terrain_Risk",
        "Temperature", "Humidity", "Pressure", "Visibility", "Wind_Speed", "Clouds", "Visibility_Risk", "Weather_Risk",
        "NDVI", "Vegetation_Density", "Vegetation_Risk"
    ]
    
    op_cols = [
        "Coverage_Pct", "Blind_Spots", "Max_Gap_Hours", "Patrols_Conducted", "Night_Shifts",
        "Vehicles_Available", "Drones_Deployed", "Readiness_Score", "Patrol_Risk"
    ]
    
    hist_cols = [
        "Historical_Event_Count", "Successful_Detection_Count", "Detection_Rate",
        "Average_Response_Time", "Night_Event_Count", "Historical_Risk"
    ]
    
    surv_cols = [
        "Human_Detection_Count", "Vehicle_Detection_Count", "Avg_Camera_Confidence",
        "Night_Movement_Count", "Camera_Activity_Score", "Motion_Score", "Infrared_Score",
        "Acoustic_Score", "Seismic_Score", "Sensor_Activity_Score"
    ]

    experiments = {
        "Exp 1: Environmental Only": env_cols,
        "Exp 2: Env + Operational": env_cols + op_cols,
        "Exp 3: Env + Op + Historical": env_cols + op_cols + hist_cols,
        "Exp 4: Full Multi-Modal System": env_cols + op_cols + hist_cols + surv_cols
    }

    target_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    y = df["Risk_Level"].map(target_map)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ablation_results = []

    print("\n" + "=" * 90)
    print("                    MULTI-MODAL ABLATION STUDY RESULTS")
    print("=" * 90)

    for exp_name, feat_subset in experiments.items():
        X_sub = df[feat_subset].copy()

        # Evaluate Logistic Regression on subset
        lr_accs, lr_f1s, lr_aucs = [], [], []
        rf_accs, rf_f1s, rf_aucs = [], [], []

        for train_idx, val_idx in skf.split(X_sub, y):
            X_tr, X_val = X_sub.iloc[train_idx], X_sub.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # LR
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
            lr.fit(X_tr, y_tr)
            y_pred_lr = lr.predict(X_val)
            lr_accs.append(accuracy_score(y_val, y_pred_lr))
            lr_f1s.append(f1_score(y_val, y_pred_lr, average="weighted", zero_division=0))

            if hasattr(lr, "predict_proba") and len(np.unique(y_val)) > 1:
                try:
                    lr_aucs.append(roc_auc_score(y_val, lr.predict_proba(X_val), multi_class="ovr", average="weighted"))
                except Exception:
                    pass

            # RF
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
            rf.fit(X_tr, y_tr)
            y_pred_rf = rf.predict(X_val)
            rf_accs.append(accuracy_score(y_val, y_pred_rf))
            rf_f1s.append(f1_score(y_val, y_pred_rf, average="weighted", zero_division=0))

            if hasattr(rf, "predict_proba") and len(np.unique(y_val)) > 1:
                try:
                    rf_aucs.append(roc_auc_score(y_val, rf.predict_proba(X_val), multi_class="ovr", average="weighted"))
                except Exception:
                    pass

        ablation_results.append({
            "Experiment": exp_name,
            "Feature_Count": len(feat_subset),
            "LR_Accuracy": f"{np.mean(lr_accs):.4f} ± {np.std(lr_accs):.4f}",
            "LR_F1_Score": f"{np.mean(lr_f1s):.4f} ± {np.std(lr_f1s):.4f}",
            "LR_ROC_AUC": f"{np.mean(lr_aucs):.4f} ± {np.std(lr_aucs):.4f}" if lr_aucs else "N/A",
            "RF_Accuracy": f"{np.mean(rf_accs):.4f} ± {np.std(rf_accs):.4f}",
            "RF_F1_Score": f"{np.mean(rf_f1s):.4f} ± {np.std(rf_f1s):.4f}",
            "RF_ROC_AUC": f"{np.mean(rf_aucs):.4f} ± {np.std(rf_aucs):.4f}" if rf_aucs else "N/A"
        })

    ablation_df = pd.DataFrame(ablation_results)
    print(ablation_df.to_string(index=False))
    print("=" * 90 + "\n")

    ablation_df.to_csv(output_path, index=False)
    print(f"[Ablation Study] Saved ablation study results to: {output_path}")
    return output_path

if __name__ == "__main__":
    run_ablation_study()
