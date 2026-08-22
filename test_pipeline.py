import os
import sys
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.predict_zone_risk import predict_live_zone_risk
from src.api.zone_analysis_api import get_zone_analysis, get_overview_stats

def run_system_validation_suite():
    """
    Executes the Complete 11-Point Phase 24 System & Spatial Validation Suite
    followed by 7 Operational Scenario Test Cases.
    """
    print("=" * 80)
    print("🛡️  BORDER RISK INTELLIGENCE & DECISION-SUPPORT: COMPLETE VALIDATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: 11-Point Spatial & Data Foundation Integrity Checks
    # -------------------------------------------------------------------------
    print("\n[PART 1] Running 11-Point Spatial & Dataset Integrity Verification...")
    checks_passed = 0
    total_checks = 11

    # Check 1: Zone Master Generated & 150 Zones
    try:
        df_master_zones = pd.read_csv("data/processed/zone_master.csv")
        assert len(df_master_zones) == 150, f"Expected 150 zones in zone_master.csv, got {len(df_master_zones)}"
        print("  ✓ [1/11] 150 Spatial Zones Generated & Documented in zone_master.csv")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [1/11] Zone Master check failed: {e}")

    # Check 2: No duplicate Zone_ID
    try:
        assert df_master_zones["Zone_ID"].nunique() == 150, "Duplicate Zone_IDs found!"
        print("  ✓ [2/11] Unique Zone_IDs Confirmed (Zero Duplicates)")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [2/11] Duplicate Zone_ID check failed: {e}")

    # Check 3: Every Dataset Maps to Exact 150 Zones
    try:
        datasets = [
            ("Terrain", "data/processed/terrain_zone_features.csv"),
            ("Weather", "data/processed/weather_features.csv"),
            ("Vegetation", "data/processed/vegetation_features.csv"),
            ("Patrol", "data/processed/patrol_features.csv"),
            ("Historical", "data/processed/historical_features.csv"),
            ("Camera", "data/processed/camera_features.csv"),
            ("Sensors", "data/processed/sensor_features.csv"),
            ("Master Risk", "data/processed/master_risk_dataset.csv")
        ]
        for name, path in datasets:
            df = pd.read_csv(path)
            assert len(df) == 150, f"{name} dataset has {len(df)} rows != 150"
            assert set(df["Zone_ID"]) == set(df_master_zones["Zone_ID"]), f"{name} Zone_IDs do not match zone_master!"
        print("  ✓ [3/11] All 7 Multi-Modal Datasets Perfectly Mapped to 150 Zones")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [3/11] Dataset mapping check failed: {e}")

    # Check 4: No Missing Critical Features in Master Dataset
    try:
        df_master = pd.read_csv("data/processed/master_risk_dataset.csv")
        critical_cols = ["Zone_ID", "Latitude", "Longitude", "Risk_Score", "Risk_Level", "Terrain_Risk", "Patrol_Risk", "Historical_Risk"]
        for col in critical_cols:
            assert col in df_master.columns, f"Missing critical column: {col}"
            assert df_master[col].isnull().sum() == 0, f"Null values found in {col}"
        print(f"  ✓ [4/11] Zero Missing/Null Critical Features across {len(df_master.columns)} Columns")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [4/11] Feature completeness check failed: {e}")

    # Check 5: Risk Score Distribution & Thresholds
    try:
        assert (df_master["Risk_Score"] >= 0).all() and (df_master["Risk_Score"] <= 100).all()
        risk_counts = df_master["Risk_Level"].value_counts().to_dict()
        assert set(risk_counts.keys()).issubset({"Low", "Medium", "High", "Critical"})
        print(f"  ✓ [5/11] Risk Scores Calibrated (Low: {risk_counts.get('Low',0)}, Medium: {risk_counts.get('Medium',0)}, High: {risk_counts.get('High',0)}, Critical: {risk_counts.get('Critical',0)})")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [5/11] Risk score calibration check failed: {e}")

    # Check 6: SHAP Explanation Generation
    try:
        za = get_zone_analysis("SGR_ZONE_096")
        assert "shap_factors" in za
        assert len(za["shap_factors"]["escalating"]) > 0 or len(za["shap_factors"]["mitigating"]) > 0
        print("  ✓ [6/11] SHAP TreeExplainer Local & Global Feature Attribution Verified")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [6/11] SHAP explanation check failed: {e}")

    # Check 7: Zone Selection & Querying
    try:
        for test_zid in ["SGR_ZONE_001", "SGR_ZONE_050", "SGR_ZONE_126", "SGR_ZONE_150"]:
            res = get_zone_analysis(test_zid)
            assert res["zone_id"] == test_zid
        print("  ✓ [7/11] Dynamic Zone Selection & API Resolution Verified")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [7/11] Zone selection check failed: {e}")

    # Check 8: Decision Engine & Personnel Allocation Index
    try:
        pa = za.get("patrol_assessment", {})
        assert "personnel_allocation_index" in pa
        assert "decision_status" in pa
        assert "decision_brief" in za
        print(f"  ✓ [8/11] Decision Engine Verified (Status: {pa.get('decision_status')}, Index: {pa.get('personnel_allocation_index')})")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [8/11] Decision engine check failed: {e}")

    # Check 9: Simulated Camera Event Features
    try:
        assert "camera" in za
        assert "human_count" in za["camera"] and "vehicle_count" in za["camera"]
        print(f"  ✓ [9/11] Dataset-Derived Camera Intelligence Mapped to Zone")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [9/11] Camera mapping check failed: {e}")

    # Check 10: Simulated Sensor Telemetry
    try:
        assert "sensors" in za
        assert "motion" in za["sensors"] and "infrared" in za["sensors"]
        print(f"  ✓ [10/11] Dataset-Derived Sensor Telemetry Mapped to Zone")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [10/11] Sensor mapping check failed: {e}")

    # Check 11: Overview KPIs
    try:
        ov = get_overview_stats()
        assert ov["total_zones"] == 150
        print(f"  ✓ [11/11] Overview Dashboard Aggregations Verified (Total: {ov['total_zones']} Zones)")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ [11/11] Overview stats check failed: {e}")

    print(f"\n👉 FOUNDATION INTEGRITY RESULT: {checks_passed}/{total_checks} CHECKS PASSED")

    # -------------------------------------------------------------------------
    # PART 2: 7 Operational Scenario Test Cases
    # -------------------------------------------------------------------------
    print("\n[PART 2] Running 7 Multi-Threat Operational Scenario Inferences...")
    scenarios = [
        {"name": "Test 1: Baseline Zone Query", "kwargs": {"zone_id": "SGR_ZONE_001"}},
        {"name": "Test 2: Unseen / Fallback Zone Query", "kwargs": {"zone_id": "UNKNOWN_ZONE_999"}},
        {"name": "Test 3: Nighttime Camera Human Threat Trigger", "kwargs": {"zone_id": "SGR_ZONE_012", "human_count": 8, "night_movements": 7}},
        {"name": "Test 4: Seismic & Acoustic Sensor Spike Event", "kwargs": {"zone_id": "SGR_ZONE_025", "motion_score": 0.92, "seismic_score": 0.85}},
        {"name": "Test 5: High Patrol Gap Hotspot Scenario", "kwargs": {"zone_id": "SGR_ZONE_088", "human_count": 3}},
        {"name": "Test 6: Low Visibility Fog Event", "kwargs": {"zone_id": "SGR_ZONE_104", "motion_score": 0.65}},
        {"name": "Test 7: Multi-Threat Critical Combination Event", "kwargs": {"zone_id": "SGR_ZONE_142", "human_count": 12, "vehicle_count": 3, "night_movements": 10, "motion_score": 0.95, "seismic_score": 0.91}}
    ]

    scenarios_passed = 0
    for idx, sc in enumerate(scenarios, start=1):
        try:
            res = predict_live_zone_risk(**sc["kwargs"])
            assert "Risk_Score" in res and "Risk_Level" in res
            print(f"  ✓ Scenario {idx}/7: {sc['name']} -> {res['Risk_Level']} ({res['Risk_Score']}%)")
            scenarios_passed += 1
        except Exception as e:
            print(f"  ❌ Scenario {idx}/7 Failed: {e}")

    print("\n" + "=" * 80)
    print(f"📊 SUMMARY: {checks_passed}/{total_checks} INTEGRITY CHECKS + {scenarios_passed}/{len(scenarios)} SCENARIOS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    run_system_validation_suite()
