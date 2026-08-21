import os
import csv
from collections import defaultdict

def process_historical(input_path="DATASETS-2/Simulated data/EMBARIA_Historical_Infiltration.csv",
                       output_path="data/processed/historical_features.csv",
                       num_zones=150):
    """
    Aggregates event-level historical infiltration records into zone risk metrics.
    """
    print(f"[Historical Processor] Reading raw infiltration dataset from: {input_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Historical infiltration file not found at {input_path}")

    zone_events = defaultdict(list)
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            z_id = row['Zone_ID']
            zone_events[z_id].append(row)

    zone_records = []

    for i in range(num_zones):
        zone_id = f"SGR_ZONE_{i + 1:03d}"
        events = zone_events.get(zone_id, [])

        event_count = len(events)
        if event_count > 0:
            detected_count = sum(1 for e in events if str(e.get('Detection_Status', '0')) == '1')
            detection_rate = detected_count / event_count
            resp_times = [float(e['Response_Time_Min']) for e in events if e.get('Response_Time_Min')]
            avg_response_time = sum(resp_times) / len(resp_times) if resp_times else 25.0
            
            night_count = sum(1 for e in events if int(e.get('Hour', 12)) >= 20 or int(e.get('Hour', 12)) <= 5)
        else:
            event_count = 0
            detected_count = 0
            detection_rate = 1.0
            avg_response_time = 15.0
            night_count = 0

        # Historical Risk Score: higher past event frequency + slower response + lower detection rate -> higher historical risk
        freq_factor = min(1.0, event_count / 80.0)
        resp_factor = min(1.0, avg_response_time / 60.0)
        det_factor = 1.0 - detection_rate

        historical_risk = round(0.45 * freq_factor + 0.35 * det_factor + 0.20 * resp_factor, 4)

        zone_records.append({
            "Zone_ID": zone_id,
            "Historical_Event_Count": event_count,
            "Successful_Detection_Count": detected_count,
            "Detection_Rate": round(detection_rate, 4),
            "Average_Response_Time": round(avg_response_time, 2),
            "Night_Event_Count": night_count,
            "Historical_Risk": historical_risk
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Zone_ID", "Historical_Event_Count", "Successful_Detection_Count",
                      "Detection_Rate", "Average_Response_Time", "Night_Event_Count", "Historical_Risk"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(zone_records)

    print(f"[Historical Processor] Successfully written {len(zone_records)} historical records to {output_path}")
    return output_path

if __name__ == "__main__":
    process_historical()
