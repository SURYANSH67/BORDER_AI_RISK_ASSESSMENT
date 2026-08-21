import os
import csv
import numpy as np

def generate_camera_data(raw_events_path="data/processed/camera_events.csv",
                         features_path="data/processed/camera_features.csv",
                         num_zones=150, num_events=1200, seed=42):
    """
    Generates simulated optical and thermal camera surveillance events
    and aggregates zone-level surveillance intelligence features.
    """
    print(f"[Camera Simulation] Generating {num_events} camera detection events across {num_zones} zones...")
    os.makedirs(os.path.dirname(raw_events_path), exist_ok=True)
    np.random.seed(seed)

    detection_types = ["Human", "Vehicle", "Animal/False Alarm"]
    detection_weights = [0.45, 0.35, 0.20]

    raw_records = []
    zone_human_counts = np.zeros(num_zones)
    zone_vehicle_counts = np.zeros(num_zones)
    zone_conf_sums = np.zeros(num_zones)
    zone_total_events = np.zeros(num_zones)
    zone_night_counts = np.zeros(num_zones)

    for evt_idx in range(1, num_events + 1):
        zone_num = int(np.random.randint(1, num_zones + 1))
        zone_id = f"SGR_ZONE_{zone_num:03d}"
        camera_id = f"CAM_{zone_num:03d}_{np.random.randint(1, 4):02d}"
        
        det_type = str(np.random.choice(detection_types, p=detection_weights))
        confidence = round(float(np.random.uniform(0.70, 0.99)), 4)
        movement_level = round(float(np.random.uniform(0.20, 0.95)), 4)
        
        hour = int(np.random.randint(0, 24))
        timestamp = f"2026-08-21 {hour:02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}"

        raw_records.append({
            "Event_ID": f"CAM_EVT_{evt_idx:05d}",
            "Zone_ID": zone_id,
            "Camera_ID": camera_id,
            "Timestamp": timestamp,
            "Hour": hour,
            "Detection_Type": det_type,
            "Detection_Confidence": confidence,
            "Movement_Level": movement_level
        })

        z_i = zone_num - 1
        zone_total_events[z_i] += 1
        zone_conf_sums[z_i] += confidence
        if det_type == "Human":
            zone_human_counts[z_i] += 1
        elif det_type == "Vehicle":
            zone_vehicle_counts[z_i] += 1
        if hour >= 20 or hour <= 5:
            zone_night_counts[z_i] += 1

    # Save raw camera events log
    with open(raw_events_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Event_ID", "Zone_ID", "Camera_ID", "Timestamp", "Hour",
                      "Detection_Type", "Detection_Confidence", "Movement_Level"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_records)

    # Build aggregated camera zone features
    zone_features = []
    for i in range(num_zones):
        zone_id = f"SGR_ZONE_{i + 1:03d}"
        tot = zone_total_events[i]
        avg_conf = round(float(zone_conf_sums[i] / tot), 4) if tot > 0 else 0.85
        human_cnt = int(zone_human_counts[i])
        veh_cnt = int(zone_vehicle_counts[i])
        night_cnt = int(zone_night_counts[i])

        # Camera activity risk score (weighted by human/vehicle activity & night movements)
        act_score = round(min(1.0, (human_cnt * 1.5 + veh_cnt * 1.0 + night_cnt * 0.8) / 30.0), 4)

        zone_features.append({
            "Zone_ID": zone_id,
            "Human_Detection_Count": human_cnt,
            "Vehicle_Detection_Count": veh_cnt,
            "Avg_Camera_Confidence": avg_conf,
            "Night_Movement_Count": night_cnt,
            "Camera_Activity_Score": act_score
        })

    with open(features_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Zone_ID", "Human_Detection_Count", "Vehicle_Detection_Count",
                      "Avg_Camera_Confidence", "Night_Movement_Count", "Camera_Activity_Score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(zone_features)

    print(f"[Camera Simulation] Successfully generated raw events ({raw_events_path}) and zone features ({features_path})")
    return features_path

if __name__ == "__main__":
    generate_camera_data()
