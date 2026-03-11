import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely import get_coordinates
import os

def select_rows_with_bad_latitudes(gdf, max_abs_lat=85):
    """
    Fast vectorized detection of rows containing any |lat| > max_abs_lat.
    Returns a subset of the original GeoDataFrame.
    """
    coords = get_coordinates(gdf.geometry)
    ys = coords[:, 1]
    bad_coord_mask = np.abs(ys) > max_abs_lat

    if not bad_coord_mask.any():
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

    geom_lengths = gdf.geometry.apply(lambda g: len(g.coords) if g is not None else 0).to_numpy()
    parent_index = np.repeat(gdf.index.to_numpy(), geom_lengths)
    bad_parent_indices = np.unique(parent_index[bad_coord_mask])

    return gdf.loc[bad_parent_indices]


def break_at_bad_latitudes(linestring, max_abs_lat=85):
    """
    Split a LineString into valid segments by breaking at any vertex
    where |lat| > max_abs_lat.
    Returns a list of LineStrings (valid segments only).
    """
    coords = list(linestring.coords)
    segments = []
    current = []

    for x, y in coords:
        if abs(y) > max_abs_lat:
            if len(current) >= 2:
                segments.append(LineString(current))
            current = []
        else:
            current.append((x, y))

    if len(current) >= 2:
        segments.append(LineString(current))

    return segments


def replace_row_with_segments(gdf, row_index, max_abs_lat=85):
    """
    Replace the geometry at row_index with multiple rows,
    one per valid segment after breaking at bogus latitudes.
    Preserves original index ordering.
    """
    row = gdf.loc[row_index]
    geom = row.geometry
    segments = break_at_bad_latitudes(geom, max_abs_lat=max_abs_lat)

    # Build new rows with the SAME index as the original row
    new_rows = []
    for seg in segments:
        r = row.copy()
        r.geometry = seg
        new_rows.append(r)

    # Convert to GeoDataFrame
    new_gdf = gpd.GeoDataFrame(new_rows, columns=gdf.columns, crs=gdf.crs)

    # Split original gdf into before / after the replaced row
    before = gdf.loc[gdf.index < row_index]
    after  = gdf.loc[gdf.index > row_index]

    # Concatenate in correct order
    gdf_out = pd.concat([before, new_gdf, after])

    return gdf_out


# tier_1_path = os.path.join('/', 'mnt', 'f', 'SDSDataService_c', '14_tier_0.gpkg')
# gdf = gpd.read_file(tier_1_path, layer='03_nir_waterlines')

# bad_rows = select_rows_with_bad_latitudes(gdf, max_abs_lat=85)

# if not bad_rows.empty:
#     idx = bad_rows.index[0]
#     gdf = replace_row_with_segments(gdf, idx, max_abs_lat=85)
#     gdf.to_file(tier_1_path, layer='03_nir_waterlines')

