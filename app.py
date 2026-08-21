import os
import sys

# Set writable MPLCONFIGDIR inside local data directory
os.environ["MPLCONFIGDIR"] = os.path.abspath("data/processed/.matplotlib_cache")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import pandas as pd
from flask import Flask, render_template, jsonify, request
from src.api.zone_analysis_api import get_zone_analysis
from src.models.predict_zone_risk import predict_live_zone_risk

app = Flask(__name__)

DATA_DIR = os.path.abspath("data/processed")
MODELS_DIR = os.path.abspath("models")
MASTER_PATH = os.path.join(DATA_DIR, "master_risk_dataset.csv")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/overview")
def get_overview():
    if not os.path.exists(MASTER_PATH):
        return jsonify({"error": "Master dataset not found. Run pipeline first."}), 404
    df = pd.read_csv(MASTER_PATH)
    counts = df["Risk_Level"].value_counts().to_dict()
    return jsonify({
        "total_zones": int(len(df)),
        "critical_count": int(counts.get("Critical", 0)),
        "high_count": int(counts.get("High", 0)),
        "medium_count": int(counts.get("Medium", 0)),
        "low_count": int(counts.get("Low", 0)),
        "avg_risk_score": float(round(df["Risk_Score"].mean(), 2))
    })

@app.route("/api/zones")
def get_zones():
    if not os.path.exists(MASTER_PATH):
        return jsonify({"error": "Master dataset not found"}), 404
    df = pd.read_csv(MASTER_PATH)
    zones = df[["Zone_ID", "Latitude", "Longitude", "Risk_Score", "Risk_Level"]].to_dict(orient="records")
    return jsonify(zones)

@app.route("/api/zone/<zone_id>")
def get_zone_detail(zone_id):
    try:
        analysis = get_zone_analysis(zone_id, master_path=MASTER_PATH)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/simulate", methods=["POST"])
def simulate_threat():
    data = request.get_json() or {}
    zone_id = data.get("zone_id", "SGR_ZONE_001")
    human_count = data.get("human_count")
    vehicle_count = data.get("vehicle_count")
    night_movements = data.get("night_movements")
    motion_score = data.get("motion_score")
    seismic_score = data.get("seismic_score")

    try:
        result = predict_live_zone_risk(
            zone_id=zone_id,
            human_count=human_count,
            vehicle_count=vehicle_count,
            night_movements=night_movements,
            motion_score=motion_score,
            seismic_score=seismic_score,
            master_path=MASTER_PATH
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/models")
def get_models_metrics():
    metrics_path = os.path.join(MODELS_DIR, "model_comparison_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])

@app.route("/api/ablation")
def get_ablation_results():
    ablation_path = os.path.join(MODELS_DIR, "ablation_study_results.csv")
    if os.path.exists(ablation_path):
        df = pd.read_csv(ablation_path)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])

@app.route("/api/shap/global")
def get_shap_global():
    shap_path = os.path.join(MODELS_DIR, "shap_feature_importance.csv")
    if os.path.exists(shap_path):
        df = pd.read_csv(shap_path)
        return jsonify(df.head(15).to_dict(orient="records"))
    return jsonify([])

@app.route("/api/datasets/summary")
def get_datasets_summary():
    return jsonify({
        "terrain": {"name": "Terrain Elevation & Slope", "points": "519,841 points", "zones": 150, "type": "Open Elevation DEM"},
        "weather": {"name": "Weather & Meteorological", "points": "Continuous", "zones": 150, "type": "Atmospheric Sensors"},
        "vegetation": {"name": "Remote Sensing NDVI", "points": "150 Grid Cells", "zones": 150, "type": "Satellite Multispectral"},
        "patrol": {"name": "Operational Patrol Logs", "points": "1,800 Weekly Logs", "zones": 150, "type": "Sector Shift Logs"},
        "historical": {"name": "Historical Infiltrations", "points": "5,231 Event Records", "zones": 150, "type": "Incident Logbook"},
        "camera": {"name": "Simulated Optical/Thermal Camera", "points": "1,200 Trigger Events", "zones": 150, "type": "Synthetic Intelligence"},
        "sensors": {"name": "Simulated Acoustic/Seismic Sensors", "points": "1,500 Telemetry Records", "zones": 150, "type": "Synthetic Intelligence"}
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("=" * 75)
    print("🛡️  EMBARIA BORDER RISK INTELLIGENCE & EXPLAINABILITY WEB APPLICATION")
    print(f"🚀  Running web application server at: http://localhost:{port}")
    print("=" * 75)
    app.run(host="0.0.0.0", port=port, debug=False)


