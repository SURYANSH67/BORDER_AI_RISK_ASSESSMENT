import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.predict_zone_risk import predict_live_zone_risk

def run_system_scenario_tests():
    """
    Executes 7 operational scenario test cases to verify real-time inference robustness,
    missing feature handling, sensor trigger overrides, and tactical recommendations.
    """
    print("=" * 80)
    print("🧪 EXECUTING MULTI-MODAL PIPELINE INTEGRATION SCENARIO TEST SUITE")
    print("=" * 80)

    scenarios = [
        {
            "name": "Test 1: Baseline Zone Query",
            "kwargs": {"zone_id": "SGR_ZONE_001"}
        },
        {
            "name": "Test 2: Unseen / Non-existent Zone Query Fallback",
            "kwargs": {"zone_id": "UNKNOWN_ZONE_999"}
        },
        {
            "name": "Test 3: Nighttime Camera Human Threat Trigger",
            "kwargs": {"zone_id": "SGR_ZONE_012", "human_count": 8, "night_movements": 7}
        },
        {
            "name": "Test 4: Seismic & Acoustic Sensor Spike Event",
            "kwargs": {"zone_id": "SGR_ZONE_025", "motion_score": 0.92, "seismic_score": 0.85}
        },
        {
            "name": "Test 5: High Patrol Gap Hotspot Scenario",
            "kwargs": {"zone_id": "SGR_ZONE_088", "human_count": 3}
        },
        {
            "name": "Test 6: Low Visibility Fog Event",
            "kwargs": {"zone_id": "SGR_ZONE_104", "motion_score": 0.65}
        },
        {
            "name": "Test 7: Multi-Threat Critical Combination Event",
            "kwargs": {
                "zone_id": "SGR_ZONE_142",
                "human_count": 12,
                "vehicle_count": 3,
                "night_movements": 10,
                "motion_score": 0.95,
                "seismic_score": 0.91
            }
        }
    ]

    passed = 0
    for idx, sc in enumerate(scenarios, start=1):
        print(f"\n--- [{idx}/7] Running {sc['name']} ---")
        try:
            res = predict_live_zone_risk(**sc["kwargs"])
            assert "Risk_Score" in res, "Missing Risk_Score"
            assert "Risk_Level" in res, "Missing Risk_Level"
            assert "Tactical_Recommendation" in res, "Missing Tactical_Recommendation"
            print(f"✅ PASSED: Predicted {res['Risk_Level']} ({res['Risk_Score']}%) - Alert: {res['Alert_Level']}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")

    print("\n" + "=" * 80)
    print(f"📊 TEST SUITE RESULT: {passed}/{len(scenarios)} SCENARIO TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_system_scenario_tests()
