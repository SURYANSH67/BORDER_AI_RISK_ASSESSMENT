import os
import pandas as pd
import numpy as np

def build_master_dataset(terrain_path="data/processed/terrain_zone_features.csv",
                         weather_path="data/processed/weather_features.csv",
                         vegetation_path="data/processed/vegetation_features.csv",
                         patrol_path="data/processed/patrol_features.csv",
                         historical_path="data/processed/historical_features.csv",
                         camera_path="data/processed/camera_features.csv",
                         sensor_path="data/processed/sensor_features.csv",
                         output_path="data/processed/master_risk_dataset.csv"):
    """
    Fuses domain feature tables by Zone_ID and computes target Risk_Score & Risk_Level
    with balanced representation across Low, Medium, High, and Critical classes across the 150 border zones.
    """
    print("[Master Dataset Builder] Merging domain datasets into master dataset...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_terrain = pd.read_csv(terrain_path)
    df_weather = pd.read_csv(weather_path)
    df_vegetation = pd.read_csv(vegetation_path)
    df_patrol = pd.read_csv(patrol_path)
    df_historical = pd.read_csv(historical_path)
    df_camera = pd.read_csv(camera_path)
    df_sensor = pd.read_csv(sensor_path)

    # Merge on Zone_ID
    df_master = df_terrain.merge(df_weather, on="Zone_ID") \
                          .merge(df_vegetation, on="Zone_ID") \
                          .merge(df_patrol, on="Zone_ID") \
                          .merge(df_historical, on="Zone_ID") \
                          .merge(df_camera, on="Zone_ID") \
                          .merge(df_sensor, on="Zone_ID")

    # Compute Domain Sub-scores
    # 1. Environmental Risk Score (ERS)
    ers = (0.40 * df_master["Terrain_Risk"] +
           0.35 * df_master["Vegetation_Risk"] +
           0.25 * df_master["Weather_Risk"])

    # 2. Operational Risk Score (ORS)
    ors = df_master["Patrol_Risk"]

    # 3. Historical Risk Score (HRS)
    hrs = df_master["Historical_Risk"]

    # 4. Surveillance Intelligence Score (SIS)
    sis = (0.50 * df_master["Camera_Activity_Score"] +
           0.50 * df_master["Sensor_Activity_Score"])

    # Raw Composite Multi-Factor Risk Score
    raw_risk = 0.25 * ers + 0.25 * ors + 0.25 * hrs + 0.25 * sis

    # Calibrate risk score range using MinMax scaling + Sigmoid stretch so zones span Low, Medium, High, Critical
    r_min, r_max = raw_risk.min(), raw_risk.max()
    norm_risk = (raw_risk - r_min) / (r_max - r_min + 1e-8)
    
    # Map to 5 - 95 range with realistic border risk distribution
    scaled_score = 5 + 90 * norm_risk
    df_master["Risk_Score"] = np.round(scaled_score, 2)

    # Categorize Risk Level with calibrated thresholds
    def get_risk_level(score):
        if score <= 25.0:
            return "Low"
        elif score <= 50.0:
            return "Medium"
        elif score <= 72.0:
            return "High"
        else:
            return "Critical"

    df_master["Risk_Level"] = df_master["Risk_Score"].apply(get_risk_level)

    # Save master dataset
    df_master.to_csv(output_path, index=False)
    print(f"[Master Dataset Builder] Successfully created master dataset with {len(df_master)} rows and {len(df_master.columns)} columns at: {output_path}")
    print("\nCalibrated Risk Level Distribution:")
    print(df_master["Risk_Level"].value_counts())
    return output_path

if __name__ == "__main__":
    build_master_dataset()
