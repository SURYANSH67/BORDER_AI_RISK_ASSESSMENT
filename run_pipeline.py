import os
import sys

# Set writable MPLCONFIGDIR inside local data directory
os.environ["MPLCONFIGDIR"] = os.path.abspath("data/processed/.matplotlib_cache")

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing.terrain_processor import process_terrain
from src.preprocessing.weather_processor import process_weather
from src.preprocessing.vegetation_processor import process_vegetation
from src.preprocessing.patrol_processor import process_patrol
from src.preprocessing.historical_processor import process_historical
from src.simulation.camera_sim import generate_camera_data
from src.simulation.sensor_sim import generate_sensor_data
from src.fusion.build_master_dataset import build_master_dataset
from src.models.train_evaluate import train_and_evaluate
from src.models.ablation_study import run_ablation_study
from src.models.explainability_shap import run_shap_explainability
from src.vision.yolo_detector import extract_camera_features_from_images
from src.dashboard.export_dashboard_data import export_dashboard_data
from src.reporting.generate_research_report import generate_research_report

def main():
    print("=" * 80)
    print("🚀 BORDER SECURITY & INFILTRATION HYBRID MULTI-MODAL ML PIPELINE RUNNER")
    print("=" * 80)

    print("\nPhase 1: Preprocessing & Domain Data Standardization...")
    process_terrain()
    process_weather()
    process_vegetation()
    process_patrol()
    process_historical()

    print("\nPhase 2: Generating Simulated Intelligence (Camera & Sensors)...")
    generate_camera_data()
    generate_sensor_data()

    print("\nPhase 3: Fusing Multi-Modal Features into Master Risk Dataset...")
    master_path = build_master_dataset()

    print("\nPhase 4: Training & Evaluating ML Models (5-Fold Cross Validation)...")
    rf_model_path, metrics_df = train_and_evaluate()

    print("\nPhase 5: Running Multi-Modal Feature Ablation Study...")
    run_ablation_study()

    print("\nPhase 6: Running SHAP Explainability & Zone Feature Attribution Analysis...")
    run_shap_explainability()

    print("\nPhase 7: Executing Modular Computer Vision (YOLO/CNN) Feature Extractor Test...")
    extract_camera_features_from_images()

    print("\nPhase 8: Exporting Payload for GIS Border Security Intelligence Dashboard...")
    export_dashboard_data()

    print("\nPhase 9: Generating Academic Research Paper Summary Report...")
    generate_research_report()

    print("\n" + "=" * 80)
    print("✅ END-TO-END MULTI-MODAL ML, ABLATION STUDY & GIS DASHBOARD PIPELINE COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
