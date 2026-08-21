import os
import csv
import numpy as np

def process_vegetation(output_path="data/processed/vegetation_features.csv",
                       num_zones=150, seed=42):
    """
    Generates realistic remote-sensing vegetation metrics (NDVI, Land Cover, Density)
    for all 150 border zones.
    """
    print(f"[Vegetation Processor] Generating remote-sensing vegetation features for {num_zones} zones...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)

    land_covers = ["Forest", "Dense Brush", "Grassland", "Barren Rocky", "Alpine Scrub"]
    zone_records = []

    for i in range(num_zones):
        zone_id = f"SGR_ZONE_{i + 1:03d}"
        
        # Spatial trend + random component
        base_ndvi = 0.2 + 0.5 * (np.sin(i / 15.0) + 1.0) / 2.0
        ndvi = float(np.clip(base_ndvi + np.random.normal(0, 0.08), -0.05, 0.85))
        
        density = float(np.clip(ndvi * 1.1 + np.random.normal(0, 0.05), 0.0, 1.0))
        
        if ndvi < 0.15:
            land_cover = "Barren Rocky"
        elif ndvi < 0.35:
            land_cover = "Alpine Scrub"
        elif ndvi < 0.55:
            land_cover = "Grassland"
        elif ndvi < 0.70:
            land_cover = "Dense Brush"
        else:
            land_cover = "Forest"

        # Dense brush & forest conceal infiltration movements -> higher concealment risk
        if land_cover in ["Dense Brush", "Forest"]:
            concealment_factor = 0.85
        elif land_cover in ["Grassland", "Alpine Scrub"]:
            concealment_factor = 0.50
        else:
            concealment_factor = 0.20

        vegetation_risk = round(0.6 * density + 0.4 * concealment_factor, 4)

        zone_records.append({
            "Zone_ID": zone_id,
            "NDVI": round(ndvi, 4),
            "Vegetation_Density": round(density, 4),
            "Land_Cover": land_cover,
            "Vegetation_Risk": vegetation_risk
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Zone_ID", "NDVI", "Vegetation_Density", "Land_Cover", "Vegetation_Risk"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(zone_records)

    print(f"[Vegetation Processor] Successfully written {len(zone_records)} vegetation records to {output_path}")
    return output_path

if __name__ == "__main__":
    process_vegetation()
