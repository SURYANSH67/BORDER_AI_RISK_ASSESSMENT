import os
import numpy as np
import pandas as pd

def process_terrain(input_path="DATASETS-2/Terrain data/terrain_final.csv",
                    output_path="data/processed/terrain_zone_features.csv",
                    num_zones=150):
    """
    Processes point-level terrain elevation and slope data (519k points) using pandas groupby
    and aggregates them instantly into 150 spatial zones (SGR_ZONE_001 to SGR_ZONE_150).
    """
    print(f"[Terrain Processor] Reading raw terrain data from: {input_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input terrain file not found at {input_path}")
        
    df = pd.read_csv(input_path)

    lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
    lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()
    
    # Map 150 zones on a 15 x 10 geographic grid across Lat/Lon
    num_lat_bins = 15
    num_lon_bins = 10
    
    lat_edges = np.linspace(lat_min, lat_max + 1e-6, num_lat_bins + 1)
    lon_edges = np.linspace(lon_min, lon_max + 1e-6, num_lon_bins + 1)
    
    lat_indices = np.clip(np.digitize(df['Latitude'].values, lat_edges) - 1, 0, num_lat_bins - 1)
    lon_indices = np.clip(np.digitize(df['Longitude'].values, lon_edges) - 1, 0, num_lon_bins - 1)
    
    df['Zone_Idx'] = lat_indices * num_lon_bins + lon_indices

    # Fast groupby aggregation
    grouped = df.groupby('Zone_Idx').agg(
        center_lat=('Latitude', 'mean'),
        center_lon=('Longitude', 'mean'),
        avg_elev=('Elevation', 'mean'),
        min_elev=('Elevation', 'min'),
        max_elev=('Elevation', 'max'),
        avg_slope=('Slope', 'mean'),
        max_slope=('Slope', 'max')
    ).to_dict('index')

    zone_records = []
    
    for z_idx in range(num_zones):
        zone_id = f"SGR_ZONE_{z_idx + 1:03d}"
        
        if z_idx in grouped:
            z_data = grouped[z_idx]
            center_lat = z_data['center_lat']
            center_lon = z_data['center_lon']
            avg_elev = z_data['avg_elev']
            min_elev = z_data['min_elev']
            max_elev = z_data['max_elev']
            avg_slope = z_data['avg_slope']
            max_slope = z_data['max_slope']
        else:
            avg_elev = 2500.0
            min_elev = 1800.0
            max_elev = 3200.0
            avg_slope = 15.0
            max_slope = 30.0
            r_lat = z_idx // num_lon_bins
            c_lon = z_idx % num_lon_bins
            center_lat = float((lat_edges[r_lat] + lat_edges[r_lat+1])/2)
            center_lon = float((lon_edges[c_lon] + lon_edges[c_lon+1])/2)

        norm_slope = min(1.0, max(0.0, avg_slope / 45.0))
        norm_elev = min(1.0, max(0.0, (avg_elev - 1000) / 4000.0))
        terrain_risk = round(0.6 * norm_slope + 0.4 * norm_elev, 4)

        zone_records.append({
            "Zone_ID": zone_id,
            "Latitude": round(center_lat, 6),
            "Longitude": round(center_lon, 6),
            "Avg_Elevation": round(avg_elev, 2),
            "Min_Elevation": round(min_elev, 2),
            "Max_Elevation": round(max_elev, 2),
            "Avg_Slope": round(avg_slope, 2),
            "Max_Slope": round(max_slope, 2),
            "Terrain_Risk": terrain_risk
        })

    res_df = pd.DataFrame(zone_records)
    res_df.to_csv(output_path, index=False)

    print(f"[Terrain Processor] Successfully wrote {len(zone_records)} zone records to {output_path}")
    return output_path

if __name__ == "__main__":
    process_terrain()
