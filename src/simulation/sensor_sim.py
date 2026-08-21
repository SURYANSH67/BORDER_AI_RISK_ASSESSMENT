import os
import csv
import numpy as np

def generate_sensor_data(raw_events_path="data/processed/sensor_events.csv",
                        features_path="data/processed/sensor_features.csv",
                        num_zones=150, num_readings=1500, seed=42):
    """
    Generates simulated multi-sensor telemetry (Motion, Infrared, Acoustic, Seismic, Vibration)
    and aggregates zone-level sensor intelligence scores.
    """
    print(f"[Sensor Simulation] Generating {num_readings} sensor readings across {num_zones} zones...")
    os.makedirs(os.path.dirname(raw_events_path), exist_ok=True)
    np.random.seed(seed)

    sensor_types = ["Motion", "Infrared", "Acoustic", "Seismic", "Vibration"]

    raw_records = []
    zone_sensor_vals = {s_type: np.zeros(num_zones) for s_type in sensor_types}
    zone_sensor_counts = {s_type: np.zeros(num_zones) for s_type in sensor_types}

    for read_idx in range(1, num_readings + 1):
        zone_num = int(np.random.randint(1, num_zones + 1))
        zone_id = f"SGR_ZONE_{zone_num:03d}"
        sensor_type = str(np.random.choice(sensor_types))
        sensor_id = f"SNS_{sensor_type[:3].upper()}_{zone_num:03d}_{np.random.randint(1, 5):02d}"
        
        reading_val = round(float(np.random.beta(2, 5)), 4)  # right-skewed normal background telemetry with occasional spikes
        if np.random.rand() < 0.15:  # 15% high activity trigger events
            reading_val = round(float(np.random.uniform(0.70, 0.98)), 4)

        hour = int(np.random.randint(0, 24))
        timestamp = f"2026-08-21 {hour:02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}"

        raw_records.append({
            "Event_ID": f"SNS_EVT_{read_idx:05d}",
            "Zone_ID": zone_id,
            "Sensor_ID": sensor_id,
            "Sensor_Type": sensor_type,
            "Reading": reading_val,
            "Timestamp": timestamp
        })

        z_i = zone_num - 1
        zone_sensor_vals[sensor_type][z_i] += reading_val
        zone_sensor_counts[sensor_type][z_i] += 1

    # Save raw sensor log
    with open(raw_events_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Event_ID", "Zone_ID", "Sensor_ID", "Sensor_Type", "Reading", "Timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_records)

    # Aggregate zone features
    zone_features = []
    for i in range(num_zones):
        zone_id = f"SGR_ZONE_{i + 1:03d}"
        
        m_score = round(float(zone_sensor_vals["Motion"][i] / max(1, zone_sensor_counts["Motion"][i])), 4)
        ir_score = round(float(zone_sensor_vals["Infrared"][i] / max(1, zone_sensor_counts["Infrared"][i])), 4)
        ac_score = round(float(zone_sensor_vals["Acoustic"][i] / max(1, zone_sensor_counts["Acoustic"][i])), 4)
        seis_score = round(float(zone_sensor_vals["Seismic"][i] / max(1, zone_sensor_counts["Seismic"][i])), 4)

        # Composite sensor activity score
        sensor_activity_score = round(0.35 * m_score + 0.30 * ir_score + 0.20 * ac_score + 0.15 * seis_score, 4)

        zone_features.append({
            "Zone_ID": zone_id,
            "Motion_Score": m_score,
            "Infrared_Score": ir_score,
            "Acoustic_Score": ac_score,
            "Seismic_Score": seis_score,
            "Sensor_Activity_Score": sensor_activity_score
        })

    with open(features_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Zone_ID", "Motion_Score", "Infrared_Score", "Acoustic_Score",
                      "Seismic_Score", "Sensor_Activity_Score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(zone_features)

    print(f"[Sensor Simulation] Successfully generated raw events ({raw_events_path}) and zone features ({features_path})")
    return features_path

if __name__ == "__main__":
    generate_sensor_data()
