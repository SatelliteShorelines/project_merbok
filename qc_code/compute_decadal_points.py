import geopandas as gpd
import numpy as np
import time
import sys
from shapely.geometry import Point

def utm_crs_for_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone:02d}"
    else:
        return f"EPSG:327{zone:02d}"

def project_cross_distance(line_wgs84, cross_distance):
    # detect UTM zone from transect start
    x0, y0 = line_wgs84.coords[0]
    utm_crs = utm_crs_for_lonlat(x0, y0)

    # move transect into its UTM CRS
    line_utm = gpd.GeoSeries([line_wgs84], crs=4326).to_crs(utm_crs).iloc[0]

    # clamp distance
    length = line_utm.length
    d = max(0, min(cross_distance, length))

    # interpolate in UTM
    p_utm = line_utm.interpolate(d)

    # convert back to WGS84
    return gpd.GeoSeries([p_utm], crs=utm_crs).to_crs(4326).iloc[0]

def _print_progress(i, total, start_time):
    frac = i / total
    bar_len = 30
    filled = int(frac * bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)

    elapsed = time.time() - start_time
    eta = elapsed * (1/frac - 1) if frac > 0 else 0

    sys.stdout.write(
        f"\r[{bar}] {frac*100:5.1f}%  ETA: {int(eta):02d}s"
    )
    sys.stdout.flush()

def compute_decadal_shoreline_points_with_ci(
    annual_shoreline_points,
    transects
):
    gdf = annual_shoreline_points.copy()
    gdf["decade"] = (gdf["year"] // 10) * 10

    merged = gdf.merge(
        transects[["transect_id", "geometry"]],
        on="transect_id",
        how="left",
        suffixes=("", "_transect")
    )

    groups = list(merged.groupby(["transect_id", "decade"]))
    total = len(groups)
    start = time.time()

    records = []
    for idx, ((tid, dec), grp) in enumerate(groups, start=1):

        cd = grp["cross_distance"].values

        mean_cd = float(np.mean(cd))
        std_cd  = float(np.std(cd, ddof=1)) if len(cd) > 1 else 0.0
        n_cd    = int(len(cd))
        ci_cd   = float(1.96 * std_cd / np.sqrt(n_cd)) if n_cd > 1 else 0.0

        line = grp["geometry_transect"].iloc[0]

        projected_point = project_cross_distance(line, mean_cd)

        records.append({
            "transect_id": tid,
            "decade": dec,
            "cross_distance": mean_cd,
            "ci": ci_cd,
            "geometry": projected_point
        })

        _print_progress(idx, total, start)

    sys.stdout.write("\n")

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

# index_path = "/mnt/f/SDSDataService_c/14_index.gpkg"
# tier_2_path = "/mnt/f/SDSDataService_c/14_tier_2.gpkg"

# annual_shoreline_points = gpd.read_file(tier_2_path, layer='00_annual_shoreline_points')
# transects = gpd.read_file(index_path, layer='00_transects')

# decadal_points = compute_decadal_shoreline_points_with_ci(
#     annual_shoreline_points,
#     transects
# )

# decadal_points.to_file(
#     "/mnt/f/SDSDataService_c/14_tier_2.gpkg",
#     layer='03_decadal_shoreline_points'
# )