import geopandas as gpd
from shapely.geometry import LineString
import sys, time

def points_to_lines_no_gaps(points, kind):
    gdf = points.copy()

    # detect time column
    if kind=="decadal":
        time_col = "decade"
    elif kind=="annual":
        time_col = "year"
    elif kind=="instantaneous":
        time_col = "dates"
    else:
        raise ValueError("No valid time column found.")

    gdf["transect_id"] = gdf["transect_id"].astype(str).str.strip()

    # parse components
    gdf["G"]   = gdf["transect_id"].str[0]
    gdf["C"]   = gdf["transect_id"].str[1]
    gdf["RR"]  = gdf["transect_id"].str[2:4]
    gdf["SSS"] = gdf["transect_id"].str[4:7]
    gdf["longshore_index"] = gdf["transect_id"].str[8:14].astype(int)

    records = []

    groups = list(gdf.groupby([time_col, "G", "C", "RR", "SSS"]))
    total = len(groups)
    start = time.time()

    def _progress(i):
        frac = i / total
        bar_len = 30
        filled = int(frac * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start
        eta = elapsed * (1/frac - 1) if frac > 0 else 0
        sys.stdout.write(f"\r[{bar}] {frac*100:5.1f}%  ETA {int(eta):02d}s")
        sys.stdout.flush()

    for idx, ((tval, G, C, RR, SSS), grp) in enumerate(groups, start=1):

        grp_sorted = grp.sort_values("longshore_index")

        segment = []
        last_L = None

        for _, row in grp_sorted.iterrows():
            L = row["longshore_index"]

            # adjacency = +50
            if last_L is None or L != last_L + 50:
                if len(segment) >= 2:
                    records.append({
                        "G": G,
                        "C": C,
                        "RR": RR,
                        "SSS": SSS,
                        time_col: tval,
                        "geometry": LineString([p.geometry for p in segment])
                    })
                segment = []

            segment.append(row)
            last_L = L

        if len(segment) >= 2:
            records.append({
                "G": G,
                "C": C,
                "RR": RR,
                "SSS": SSS,
                time_col: tval,
                "geometry": LineString([p.geometry for p in segment])
            })

        _progress(idx)

    sys.stdout.write("\n")

    return gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)

import geopandas as gpd
from shapely.geometry import LineString

def points_to_lines_from_components_no_time(points):
    gdf = points.copy()

    gdf["transect_id"] = gdf["transect_id"].astype(str).str.strip()
    gdf["longshore_index"] = gdf["transect_id"].str[8:14].astype(int)

    records = []

    groups = gdf.groupby(["G", "C", "RR", "SSS"])

    for (G, C, RR, SSS), grp in groups:
        grp_sorted = grp.sort_values("longshore_index")

        segment = []
        last_L = None

        for _, row in grp_sorted.iterrows():
            L = row["longshore_index"]

            if last_L is None or L != last_L + 50:
                if len(segment) >= 2:
                    records.append({
                        "G": G,
                        "C": C,
                        "RR": RR,
                        "SSS": SSS,
                        "geometry": LineString([p.geometry for p in segment])
                    })
                segment = []

            segment.append(row)
            last_L = L

        if len(segment) >= 2:
            records.append({
                "G": G,
                "C": C,
                "RR": RR,
                "SSS": SSS,
                "geometry": LineString([p.geometry for p in segment])
            })

    return gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)

# tier_1 = "/mnt/f/SDSDataService_c/14_tier_1.gpkg"
# tier_2 = "/mnt/f/SDSDataService_c/14_tier_2.gpkg"

# shoreline_points = gpd.read_file(tier_1, layer='01_shoreline_points')
# annual_shoreline_points = gpd.read_file(tier_2, layer="00_annual_shoreline_points")
# decadal_shoreline_points = gpd.read_file(tier_2, layer='03_decadal_shoreline_points')

# shoreline_lines = points_to_lines_no_gaps(shoreline_points, kind="instantaneous")
# annual_shoreline_lines = points_to_lines_no_gaps(annual_shoreline_points, kind="annual")
# decadal_shoreline_lines = points_to_lines_no_gaps(decadal_shoreline_points, kind="decadal")

# shoreline_lines.to_file(tier_1, layer="02_shoreline_lines")
# annual_shoreline_lines.to_file(tier_2, layer="01_annual_shoreline_lines")
# decadal_shoreline_lines.to_file(tier_2, layer="04_decadal_shoreline_lines")

