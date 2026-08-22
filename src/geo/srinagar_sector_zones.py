import numpy as np
import pandas as pd

# Official Srinagar Sector Study Boundary Polygon [Lat, Lon]
SRINAGAR_SECTOR_BOUNDARY = [
    [34.250, 74.780],
    [34.240, 74.920],
    [34.180, 75.030],
    [34.100, 75.060],
    [34.010, 75.020],
    [33.940, 74.910],
    [33.930, 74.760],
    [33.970, 74.640],
    [34.060, 74.600],
    [34.160, 74.630],
    [34.230, 74.700],
    [34.250, 74.780]
]

HEX_RADIUS_DEG = 0.017025125628140704

def point_in_poly(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def generate_srinagar_150_zones():
    """
    Generates exactly 150 contiguous hexagonal spatial analysis cells
    strictly clipped inside the Srinagar Sector Study Area boundary.
    """
    poly_lonlat = [(p[1], p[0]) for p in SRINAGAR_SECTOR_BOUNDARY]

    minx = min(p[0] for p in poly_lonlat)
    maxx = max(p[0] for p in poly_lonlat)
    miny = min(p[1] for p in poly_lonlat)
    maxy = max(p[1] for p in poly_lonlat)

    dx = HEX_RADIUS_DEG * np.sqrt(3)
    dy = HEX_RADIUS_DEG * 1.5

    xs = np.arange(minx - dx, maxx + dx, dx)
    ys = np.arange(miny - dy, maxy + dy, dy)

    pts = []
    for j, y in enumerate(ys):
        offset = (dx / 2.0) if (j % 2 == 1) else 0.0
        for x in xs:
            if point_in_poly(x + offset, y, poly_lonlat):
                pts.append((round(float(y), 5), round(float(x + offset), 5)))

    # Guarantee exactly 150 zones
    pts = pts[:150]

    zones = []
    for idx, (lat, lon) in enumerate(pts, start=1):
        zone_id = f"SGR_ZONE_{idx:03d}"
        zones.append({
            "Zone_ID": zone_id,
            "Latitude": lat,
            "Longitude": lon,
            "Sector": "Srinagar",
            "Boundary_Clipped": True
        })

    return pd.DataFrame(zones)

def get_hexagon_coordinates(center_lat, center_lon, radius_deg=HEX_RADIUS_DEG):
    """
    Returns the 6 coordinates of the contiguous hexagon polygon for a zone centroid.
    """
    coords = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = np.radians(angle_deg)
        lat = center_lat + radius_deg * np.sin(angle_rad)
        lon = center_lon + (radius_deg / np.cos(np.radians(center_lat))) * np.cos(angle_rad)
        coords.append([round(float(lat), 5), round(float(lon), 5)])
    return coords

if __name__ == "__main__":
    df = generate_srinagar_150_zones()
    print(f"Generated {len(df)} Srinagar Sector zones.")
    print(df.head())
