import os
import csv
from collections import defaultdict

def process_patrol(input_path="DATASETS-2/Simulated data/EMBARIA_Patrol_Coverage.csv",
                   output_path="data/processed/patrol_features.csv",
                   num_zones=150):
    """
    Aggregates weekly patrol coverage records into zone-level operational features.
    """
    print(f"[Patrol Processor] Reading raw patrol dataset from: {input_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Patrol dataset not found at {input_path}")

    zone_data = defaultdict(list)
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            z_id = row['Zone_ID']
            zone_data[z_id].append(row)

    zone_records = []
    
    # Process each of the 150 zones
    for i in range(num_zones):
        zone_id = f"SGR_ZONE_{i + 1:03d}"
        rows = zone_data.get(zone_id, [])
        
        if rows:
            coverage_pct = sum(float(r['Coverage_Pct']) for r in rows) / len(rows)
            blind_spots = sum(float(r['Blind_Spots']) for r in rows) / len(rows)
            max_gap_hours = sum(float(r['Max_Gap_Hours']) for r in rows) / len(rows)
            patrols_conducted = sum(float(r['Patrols_Conducted']) for r in rows) / len(rows)
            night_shifts = sum(float(r['Night_Shifts']) for r in rows) / len(rows)
            vehicles_available = sum(float(r['Vehicles_Available']) for r in rows) / len(rows)
            drones_deployed = sum(float(r['Drones_Deployed']) for r in rows) / len(rows)
            readiness_score = sum(float(r['Readiness_Score']) for r in rows) / len(rows)
        else:
            # Fallback if zone not present in raw file
            coverage_pct = 75.0
            blind_spots = 2.0
            max_gap_hours = 6.0
            patrols_conducted = 15.0
            night_shifts = 5.0
            vehicles_available = 2.0
            drones_deployed = 1.0
            readiness_score = 70.0

        # Operational Patrol Risk Formula: higher gaps & blind spots, lower readiness & coverage -> higher risk
        gap_factor = min(1.0, max_gap_hours / 24.0)
        blind_spot_factor = min(1.0, blind_spots / 10.0)
        coverage_factor = 1.0 - (coverage_pct / 100.0)
        readiness_factor = 1.0 - (readiness_score / 100.0)

        patrol_risk = round(0.35 * gap_factor + 0.25 * blind_spot_factor + 0.20 * coverage_factor + 0.20 * readiness_factor, 4)

        zone_records.append({
            "Zone_ID": zone_id,
            "Coverage_Pct": round(coverage_pct, 2),
            "Blind_Spots": round(blind_spots, 2),
            "Max_Gap_Hours": round(max_gap_hours, 2),
            "Patrols_Conducted": round(patrols_conducted, 2),
            "Night_Shifts": round(night_shifts, 2),
            "Vehicles_Available": round(vehicles_available, 2),
            "Drones_Deployed": round(drones_deployed, 2),
            "Readiness_Score": round(readiness_score, 2),
            "Patrol_Risk": patrol_risk
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Zone_ID", "Coverage_Pct", "Blind_Spots", "Max_Gap_Hours",
                      "Patrols_Conducted", "Night_Shifts", "Vehicles_Available",
                      "Drones_Deployed", "Readiness_Score", "Patrol_Risk"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(zone_records)

    print(f"[Patrol Processor] Successfully written {len(zone_records)} patrol records to {output_path}")
    return output_path

if __name__ == "__main__":
    process_patrol()
