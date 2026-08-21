import os
import sys

# Set writable MPLCONFIGDIR
os.environ["MPLCONFIGDIR"] = os.path.abspath("data/processed/.matplotlib_cache")

import json
import joblib
import pandas as pd
import numpy as np
import shap

def predict_live_zone_risk(zone_id="SGR_ZONE_042",
                           human_count=None,
                           vehicle_count=None,
                           night_movements=None,
                           motion_score=None,
                           seismic_score=None,
                           master_path="data/processed/master_risk_dataset.csv",
                           model_path="models/random_forest_model.joblib"):
    """
    Performs real-time single-zone infiltration risk prediction, local SHAP attribution,
    and generates actionable tactical defense recommendations based on live telemetry inputs.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run run_pipeline.py first.")

    artifact = joblib.load(model_path)
    rf_model = artifact["model"]
    feature_names = artifact["feature_names"]
    target_map = artifact["target_map"]
    inv_target_map = {v: k for k, v in target_map.items()}

    df = pd.read_csv(master_path)
    zone_rows = df[df["Zone_ID"] == zone_id]

    if len(zone_rows) == 0:
        print(f"[Predictor] Zone {zone_id} not found in master dataset. Using baseline zone SGR_ZONE_001.")
        zone_row = df.iloc[0].copy()
        zone_id = zone_row["Zone_ID"]
    else:
        zone_row = zone_rows.iloc[0].copy()

    # Apply live camera / sensor telemetry overrides if provided
    if human_count is not None:
        zone_row["Human_Detection_Count"] = human_count
    if vehicle_count is not None:
        zone_row["Vehicle_Detection_Count"] = vehicle_count
    if night_movements is not None:
        zone_row["Night_Movement_Count"] = night_movements
    if human_count is not None or vehicle_count is not None:
        h = zone_row["Human_Detection_Count"]
        v = zone_row["Vehicle_Detection_Count"]
        n = zone_row["Night_Movement_Count"]
        zone_row["Camera_Activity_Score"] = round(min(1.0, (h * 1.5 + v * 1.0 + n * 0.8) / 30.0), 4)

    if motion_score is not None:
        zone_row["Motion_Score"] = motion_score
    if seismic_score is not None:
        zone_row["Seismic_Score"] = seismic_score
    if motion_score is not None or seismic_score is not None:
        m = zone_row["Motion_Score"]
        ir = zone_row["Infrared_Score"]
        ac = zone_row["Acoustic_Score"]
        seis = zone_row["Seismic_Score"]
        zone_row["Sensor_Activity_Score"] = round(0.35 * m + 0.30 * ir + 0.20 * ac + 0.15 * seis, 4)

    X_single = pd.DataFrame([zone_row[feature_names]])
    
    # Model Prediction
    pred_class_idx = rf_model.predict(X_single)[0]
    pred_class = inv_target_map.get(pred_class_idx, "Medium")
    pred_proba = rf_model.predict_proba(X_single)[0]

    # Re-calculate composite risk score
    ers = 0.40 * zone_row["Terrain_Risk"] + 0.35 * zone_row["Vegetation_Risk"] + 0.25 * zone_row["Weather_Risk"]
    ors = zone_row["Patrol_Risk"]
    hrs = zone_row["Historical_Risk"]
    sis = 0.50 * zone_row["Camera_Activity_Score"] + 0.50 * zone_row["Sensor_Activity_Score"]
    raw_score = 0.25 * ers + 0.25 * ors + 0.25 * hrs + 0.25 * sis
    
    # Scale to 0-100%
    risk_score_pct = round(float(np.clip(10 + 82 * raw_score, 0.0, 100.0)), 2)

    # SHAP Attribution
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_single)

    if isinstance(shap_vals, list):
        feat_shap = shap_vals[pred_class_idx][0]
    elif len(shap_vals.shape) == 3:
        feat_shap = shap_vals[0, :, pred_class_idx]
    else:
        feat_shap = shap_vals[0]

    shap_pairs = sorted(zip(feature_names, feat_shap, X_single.iloc[0]), key=lambda x: abs(x[1]), reverse=True)
    top_escalating = [{"feature": f, "impact": round(float(s), 4), "value": float(v)} for f, s, v in shap_pairs if s > 0][:4]
    top_mitigating = [{"feature": f, "impact": round(float(s), 4), "value": float(v)} for f, s, v in shap_pairs if s < 0][:4]

    # Tactical Recommendation Engine
    if pred_class == "Critical" or risk_score_pct >= 75.0:
        alert_level = "RED (CRITICAL THREAT)"
        tactical_action = (
            "1. Scramble Reconnaissance Drone to Sector coordinates immediately.\n"
            "2. Dispatch Quick Reaction Team (QRT) / Mobile Patrol to cover blind spot gap.\n"
            "3. Elevate perimeter thermal imaging camera zoom and alert Outpost Command."
        )
    elif pred_class == "High" or risk_score_pct >= 55.0:
        alert_level = "AMBER (HIGH THREAT)"
        tactical_action = (
            "1. Increase thermal surveillance camera sweep frequency.\n"
            "2. Direct nearby foot patrol to conduct targeted sweep along vegetation corridor.\n"
            "3. Verify sensor telemetry alerts with secondary infrared array."
        )
    elif pred_class == "Medium":
        alert_level = "YELLOW (MODERATE RISK)"
        tactical_action = (
            "1. Continue routine automated camera and sensor telemetry monitoring.\n"
            "2. Maintain planned patrol rotation schedule."
        )
    else:
        alert_level = "GREEN (LOW RISK)"
        tactical_action = "Routine surveillance operations. All systems nominal."

    response = {
        "Zone_ID": zone_id,
        "Risk_Score": risk_score_pct,
        "Risk_Level": pred_class,
        "Alert_Level": alert_level,
        "Class_Probabilities": {inv_target_map[i]: round(float(p), 4) for i, p in enumerate(pred_proba)},
        "Tactical_Recommendation": tactical_action,
        "SHAP_Escalating_Factors": top_escalating,
        "SHAP_Mitigating_Factors": top_mitigating
    }

    print("\n" + "=" * 75)
    print(f"🎯 REAL-TIME LIVE INFERENCE REPORT — ZONE: {zone_id}")
    print("=" * 75)
    print(f"Infiltration Risk Score : {risk_score_pct}%")
    print(f"Risk Classification     : {pred_class} ({alert_level})")
    print("\n🚨 Tactical Action Directives:")
    print(tactical_action)
    print("\n📈 Primary SHAP Escalating Factors:")
    for item in top_escalating:
        print(f"   • {item['feature']:28s} = {item['value']:8.2f} (Impact: +{item['impact']:.4f})")
    print("=" * 75 + "\n")

    return response

if __name__ == "__main__":
    # Test with simulated live threat trigger event in Zone SGR_ZONE_042
    predict_live_zone_risk(
        zone_id="SGR_ZONE_042",
        human_count=6,
        vehicle_count=2,
        night_movements=5,
        motion_score=0.88,
        seismic_score=0.74
    )
