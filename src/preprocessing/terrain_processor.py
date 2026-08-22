import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.geo.srinagar_sector_zones import generate_srinagar_150_zones, HEX_RADIUS_DEG

def process_terrain(input_path="DATASETS-2/Terrain data/terrain_final.csv",
                    output_path="data/processed/terrain_zone_features.csv"):
    """
    Processes 519k point-level terrain elevation and slope data and spatially joins them
    to the 150 Srinagar Sector Study Boundary zones using cKDTree nearest-neighbor aggregation.
    """
    print(f"[Terrain Processor] Reading raw terrain data from: {input_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input terrain file not found at {input_path}")
        
    df_raw = pd.read_csv(input_path)
    
    # 150 Official Srinagar Sector zones
    df_zones = generate_srinagar_150_zones()
    
    # Build spatial index on raw terrain points (Lat, Lon)
    tree = cKDTree(df_raw[['Latitude', 'Longitude']].values)
    
    zone_coords = df_zones[['Latitude', 'Longitude']].values
    
    # Query 100 nearest terrain points for each Srinagar zone
    distances, indices = tree.query(zone_coords, k=100)
    
    zone_records = []
    for idx, row in df_zones.iterrows():
        zone_id = row['Zone_ID']
        lat = row['Latitude']
        lon = row['Longitude']
        
        neighbor_pts = df_raw.iloc[indices[idx]]
        
        avg_elev = float(neighbor_pts['Elevation'].mean())
        min_elev = float(neighbor_pts['Elevation'].min())
        max_elev = float(neighbor_pts['Elevation'].max())
        avg_slope = float(neighbor_pts['Slope'].mean())
        max_slope = float(neighbor_pts['Slope'].max())
        
        norm_slope = min(1.0, max(0.0, avg_slope / 45.0))
        norm_elev = min(1.0, max(0.0, (avg_elev - 1500.0) / 3500.0))
        terrain_risk = round(0.6 * norm_slope + 0.4 * norm_elev, 4)
        
        zone_records.append({
            "Zone_ID": zone_id,
            "Latitude": lat,
            "Longitude": lon,
            "Avg_Elevation": round(avg_elev, 2),
            "Min_Elevation": round(min_elev, 2),
            "Max_Elevation": round(max_elev, 2),
            "Avg_Slope": round(avg_slope, 2),
            "Max_Slope": round(max_slope, 2),
            "Terrain_Risk": terrain_risk
        })
        
    res_df = pd.DataFrame(zone_records)
    res_df.to_csv(output_path, index=False)
    print(f"[Terrain Processor] Successfully wrote {len(zone_records)} Srinagar zone records to {output_path}")
    return output_path

if __name__ == "__main__":
    process_terrain()
