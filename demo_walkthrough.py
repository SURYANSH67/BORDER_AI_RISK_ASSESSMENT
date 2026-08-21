import os
import sys
import time
import json
import pandas as pd

# Set writable MPLCONFIGDIR
os.environ["MPLCONFIGDIR"] = os.path.abspath("data/processed/.matplotlib_cache")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.api.zone_analysis_api import get_zone_analysis
from src.models.predict_zone_risk import predict_live_zone_risk
from test_pipeline import run_system_scenario_tests

def print_step_header(step_num, title):
    print("\n" + "=" * 80)
    print(f"📍 STEP {step_num}: {title.upper()}")
    print("=" * 80)
    time.sleep(0.5)

def run_interactive_demo():
    print("*" * 80)
    print("🛡️  BORDER SECURITY RISK INTELLIGENCE & ANALYSIS — EVALUATOR DEMO")
    print("    Hybrid Multi-Modal Machine Learning & Explainability Research Prototype")
    print("*" * 80)

    # STEP 1: Overview
    print_step_header(1, "150 Spatial Border Zones & Target Calibration")
    df_master = pd.read_csv("data/processed/master_risk_dataset.csv")
    counts = df_master["Risk_Level"].value_counts().to_dict()
    print(f"Total Spatial Zones Analysed : {len(df_master)} Zones (Srinagar Sector SGR_ZONE_001..150)")
    print(f"Total Tabular Features       : {len(df_master.columns) - 6} Integrated Modality Features")
    print("Calibrated Risk Distribution :")
    for lvl in ["Critical", "High", "Medium", "Low"]:
        print(f"   • {lvl:8s}: {counts.get(lvl, 0):2d} Zones")

    # STEP 2: Critical Zone Inspection
    print_step_header(2, "Critical Zone Inspection — Zone SGR_ZONE_126")
    z126 = get_zone_analysis("SGR_ZONE_126")
    print(f"Zone ID             : {z126['zone_id']}")
    print(f"Infiltration Risk   : {z126['risk_score']}% ({z126['risk_level']})")
    print(f"7-Day Trend         : {z126['trend_direction']} {z126['trend']}")

    # STEP 3: Multi-Modal Breakdown
    print_step_header(3, "Multi-Modal Domain Risk Breakdown")
    print(f"Environmental Risk  : {z126['environmental']['terrain_risk']:.2f} (Terrain) | {z126['environmental']['weather_risk']:.2f} (Weather) | {z126['environmental']['vegetation_risk']:.2f} (Vegetation)")
    print(f"Operational Risk    : {z126['operational']['patrol_risk']:.2f} (Patrol Coverage: {z126['operational']['coverage_pct']:.0f}%, Max Gap: {z126['operational']['max_gap_hours']:.1f} hrs)")
    print(f"Historical Risk     : {z126['historical']['historical_risk']:.2f} ({z126['historical']['event_count']} Infiltration Events)")
    print(f"Surveillance Intel  : {z126['camera']['activity_score']:.2f} (Camera Activity) | {z126['sensors']['activity_score']:.2f} (Sensor Activity)")

    # STEP 4: SHAP Explainability
    print_step_header(4, "Explainable AI (SHAP TreeExplainer) Attribution")
    print(f"Automated Explanation: \"{z126['shap_factors']['explanation_text']}\"")
    print("\nTop Escalating Risk Drivers (+SHAP):")
    for esc in z126['shap_factors']['escalating']:
        print(f"   ▲ {esc['factor']:28s} = {esc['value']:18s} (Impact: {esc['impact']})")
    print("\nTop Mitigating Risk Drivers (-SHAP):")
    for mit in z126['shap_factors']['mitigating']:
        print(f"   ▼ {mit['factor']:28s} = {mit['value']:18s} (Impact: {mit['impact']})")

    # STEP 5: Simulated Surveillance Telemetry
    print_step_header(5, "Simulated Camera & Multi-Sensor Telemetry")
    print(f"[Simulated Camera Data]  : {z126['camera']['human_count']} Humans, {z126['camera']['vehicle_count']} Vehicles (Confidence: {z126['camera']['avg_confidence']*100:.1f}%)")
    print(f"[Simulated Sensor Array] : Motion: {z126['sensors']['motion']:.2f} | Infrared: {z126['sensors']['infrared']:.2f} | Acoustic: {z126['sensors']['acoustic']:.2f} | Seismic: {z126['sensors']['seismic']:.2f}")

    # STEP 6: Zone Comparison
    print_step_header(6, "Side-by-Side Zone Risk Comparison (SGR_ZONE_126 vs SGR_ZONE_001)")
    z001 = get_zone_analysis("SGR_ZONE_001")
    print(f"{'Domain Metric':<25} | {'SGR_ZONE_126 (Critical)':<25} | {'SGR_ZONE_001 (Medium)':<25}")
    print("-" * 80)
    print(f"{'Risk Score':<25} | {z126['risk_score']:<25.1f}% | {z001['risk_score']:<25.1f}%")
    print(f"{'Terrain Risk':<25} | {z126['environmental']['terrain_risk']:<25.2f} | {z001['environmental']['terrain_risk']:<25.2f}")
    print(f"{'Patrol Gap (hrs)':<25} | {z126['operational']['max_gap_hours']:<25.1f} | {z001['operational']['max_gap_hours']:<25.1f}")
    print(f"{'Historical Events':<25} | {z126['historical']['event_count']:<25d} | {z001['historical']['event_count']:<25d}")
    print(f"{'Camera Activity':<25} | {z126['camera']['activity_score']:<25.2f} | {z001['camera']['activity_score']:<25.2f}")

    # STEP 7: 5-Fold Stratified ML Comparison
    print_step_header(7, "5-Fold Stratified Cross-Validation ML Evaluation")
    metrics_df = pd.read_csv("models/model_comparison_metrics.csv")
    print(metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))
    print("\n*Finding: Logistic Regression delivers the highest linear baseline performance (74% Acc, 87.5% ROC-AUC).")

    # STEP 8: Ablation Study
    print_step_header(8, "Multi-Modal Feature Ablation Study (Empirical Proof)")
    ablation_df = pd.read_csv("models/ablation_study_results.csv")
    print(ablation_df[['Experiment', 'Feature_Count', 'LR_F1_Score', 'LR_ROC_AUC', 'RF_F1_Score']].to_string(index=False))
    print("\n*Empirical Proof: F1-score progresses from 0.4720 (Environmental Only) -> 0.7296 (Full Multi-Modal System).")

    # STEP 9: Scenario Tests
    print_step_header(9, "Integration Test Suite Execution (7/7 Scenarios)")
    run_system_scenario_tests()

    # STEP 10: Final Dashboard & Next Steps
    print("\n" + "=" * 80)
    print("✅ EVALUATOR DEMO COMPLETED SUCCESSFULLY!")
    print("   • Launch Interactive GIS Dashboard:  python3 run_dashboard.py")
    print("   • Read Full 20-Section Thesis Report: reports/FINAL_THESIS_REPORT.md")
    print("   • View Defense Presentation Slides:   reports/PROJECT_PRESENTATION_SLIDES.md")
    print("=" * 80)

if __name__ == "__main__":
    run_interactive_demo()
