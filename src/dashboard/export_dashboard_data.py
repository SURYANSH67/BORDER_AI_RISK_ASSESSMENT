import os
import json
import pandas as pd
import numpy as np
from src.api.zone_analysis_api import get_zone_analysis

def export_dashboard_data(master_path="data/processed/master_risk_dataset.csv",
                         model_path="models/random_forest_model.joblib",
                         cm_path="models/confusion_matrices.json",
                         shap_imp_path="models/shap_feature_importance.csv",
                         ablation_path="models/ablation_study_results.csv",
                         output_path="dashboard/dashboard_data.json"):
    """
    Exports complete JSON payload for the 9-Page GIS Border Security Intelligence Dashboard
    using single-source get_zone_analysis API for all 150 border zones.
    """
    print("[Dashboard Data Exporter] Generating dashboard data payload...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(master_path)
    
    cm_data = {}
    if os.path.exists(cm_path):
        with open(cm_path) as f:
            cm_data = json.load(f)

    metrics_df_data = []
    metrics_path = "models/model_comparison_metrics.csv"
    if os.path.exists(metrics_path):
        metrics_df_data = pd.read_csv(metrics_path).to_dict(orient="records")

    ablation_df_data = []
    if os.path.exists(ablation_path):
        ablation_df_data = pd.read_csv(ablation_path).to_dict(orient="records")

    counts = df["Risk_Level"].value_counts().to_dict()
    summary_cards = {
        "total_zones": int(len(df)),
        "critical_count": int(counts.get("Critical", 0)),
        "high_count": int(counts.get("High", 0)),
        "medium_count": int(counts.get("Medium", 0)),
        "low_count": int(counts.get("Low", 0)),
        "avg_risk_score": float(np.round(df["Risk_Score"].mean(), 2))
    }

    feature_importances = []
    if os.path.exists(shap_imp_path):
        feature_importances = pd.read_csv(shap_imp_path).to_dict(orient="records")

    # Generate zone payload for all 150 zones via get_zone_analysis API
    zones = []
    for z_id in df["Zone_ID"]:
        z_payload = get_zone_analysis(str(z_id), master_path=master_path)
        zones.append(z_payload)

    # Sample records for Dataset Explorer tab
    explorer_samples = {
        "terrain_sample": pd.read_csv("data/processed/terrain_zone_features.csv").head(5).to_dict(orient="records"),
        "weather_sample": pd.read_csv("data/processed/weather_features.csv").head(5).to_dict(orient="records"),
        "vegetation_sample": pd.read_csv("data/processed/vegetation_features.csv").head(5).to_dict(orient="records"),
        "patrol_sample": pd.read_csv("data/processed/patrol_features.csv").head(5).to_dict(orient="records"),
        "historical_sample": pd.read_csv("data/processed/historical_features.csv").head(5).to_dict(orient="records"),
        "camera_sample": pd.read_csv("data/processed/camera_features.csv").head(5).to_dict(orient="records"),
        "sensor_sample": pd.read_csv("data/processed/sensor_features.csv").head(5).to_dict(orient="records")
    }

    dashboard_payload = {
        "summary": summary_cards,
        "feature_importances": feature_importances,
        "models_evaluation": metrics_df_data,
        "ablation_study": ablation_df_data,
        "confusion_matrices": cm_data,
        "explorer_samples": explorer_samples,
        "zones": zones
    }

    with open(output_path, "w") as f:
        json.dump(dashboard_payload, f, indent=2)

    print(f"[Dashboard Data Exporter] Exported complete dashboard payload to: {output_path}")
    return output_path

if __name__ == "__main__":
    export_dashboard_data()
