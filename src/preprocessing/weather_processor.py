import os
import csv

def process_weather(input_path="DATASETS-2/weather and visibility data/FINAL_weather_risk_dataset.csv",
                    output_path="data/processed/weather_features.csv",
                    num_zones=150):
    """
    Consolidates weather & visibility datasets, standardizes column names,
    and maps weather parameters across all 150 border zones (SGR_ZONE_001 to SGR_ZONE_150).
    """
    print(f"[Weather Processor] Reading raw weather dataset from: {input_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Weather file not found at {input_path}")

    raw_weather_records = []
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_weather_records.append(row)

    num_raw = len(raw_weather_records)
    zone_records = []

    for i in range(num_zones):
        zone_id = f"SGR_ZONE_{i + 1:03d}"
        raw_row = raw_weather_records[i % num_raw]
        
        # Add slight zone-specific micro-climate variation based on zone index
        temp = float(raw_row.get('temperature', 0.5)) + ((i % 7) - 3) * 0.02
        humidity = float(raw_row.get('humidity', 0.5)) + ((i % 5) - 2) * 0.01
        pressure = float(raw_row.get('pressure', 0.5))
        visibility = float(raw_row.get('visibility', 10000))
        wind_speed = float(raw_row.get('wind_speed', 0.5)) + ((i % 3) - 1) * 0.03
        clouds = float(raw_row.get('clouds', 0.5))
        visibility_risk = int(raw_row.get('visibility_risk', 1))
        weather_risk_score = float(raw_row.get('weather_risk_score', 0.3))

        # Clamp values
        temp = max(0.0, min(1.0, temp))
        humidity = max(0.0, min(1.0, humidity))
        wind_speed = max(0.0, min(1.0, wind_speed))
        weather_risk_score = max(0.0, min(1.0, weather_risk_score))

        zone_records.append({
            "Zone_ID": zone_id,
            "Temperature": round(temp, 4),
            "Humidity": round(humidity, 4),
            "Pressure": round(pressure, 4),
            "Visibility": round(visibility, 2),
            "Wind_Speed": round(wind_speed, 4),
            "Clouds": round(clouds, 4),
            "Visibility_Risk": visibility_risk,
            "Weather_Risk": round(weather_risk_score, 4)
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Zone_ID", "Temperature", "Humidity", "Pressure", "Visibility",
                      "Wind_Speed", "Clouds", "Visibility_Risk", "Weather_Risk"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(zone_records)

    print(f"[Weather Processor] Successfully written {len(zone_records)} weather records to {output_path}")
    return output_path

if __name__ == "__main__":
    process_weather()
