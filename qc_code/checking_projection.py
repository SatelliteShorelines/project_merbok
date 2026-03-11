import os
import numpy as np
import geopandas as gpd
import pyproj
import shapely
from tqdm import tqdm
from shapely.geometry import Point

def project_points_seaward_wgs84(
    tier_gpkg_path: str,
    gpkg_layer: str,
    index_gpkg_path: str,
    transects_layer: str,
    cross_distance_column: str,
) -> gpd.GeoDataFrame:

    pts = gpd.read_file(tier_gpkg_path, layer=gpkg_layer)
    tx  = gpd.read_file(index_gpkg_path, layer=transects_layer)

    # --- REQUIREMENT: transects must contain utm_zone_epsg field ---
    if "utm_zone_epsg" not in tx.columns:
        raise ValueError("transects_layer is missing required 'utm_zone_epsg' column.")

    # --- Build transformers with progress ---
    tx["_to_utm"] = [
        pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
        for epsg in tqdm(tx["utm_zone_epsg"], desc="Building to_utm transformers")
    ]

    tx["_to_wgs"] = [
        pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
        for epsg in tqdm(tx["utm_zone_epsg"], desc="Building to_wgs transformers")
    ]

    # --- Compute UTM start/end coords (with progress) ---
    start_lon = np.array([g.coords[0][0] for g in tx.geometry], float)
    start_lat = np.array([g.coords[0][1] for g in tx.geometry], float)
    end_lon   = np.array([g.coords[-1][0] for g in tx.geometry], float)
    end_lat   = np.array([g.coords[-1][1] for g in tx.geometry], float)

    start_x = []
    start_y = []
    end_x   = []
    end_y   = []

    for tr, lon0, lat0, lon1, lat1 in tqdm(
        zip(tx["_to_utm"], start_lon, start_lat, end_lon, end_lat),
        total=len(tx),
        desc="Transforming start/end to UTM"
    ):
        sx, sy = tr.transform(lon0, lat0)
        ex, ey = tr.transform(lon1, lat1)

        start_x.append(sx)
        start_y.append(sy)
        end_x.append(ex)
        end_y.append(ey)

    start_x = np.array(start_x)
    start_y = np.array(start_y)
    end_x   = np.array(end_x)
    end_y   = np.array(end_y)

    tx["_start_x_utm"] = start_x
    tx["_start_y_utm"] = start_y
    tx["_angle_rad"] = np.arctan2(end_y - start_y, end_x - start_x)

    # --- Map per-transect values to points ---
    idx = tx.set_index("transect_id")

    pts["_start_x_utm"] = pts["transect_id"].map(idx["_start_x_utm"])
    pts["_start_y_utm"] = pts["transect_id"].map(idx["_start_y_utm"])
    pts["_angle_rad"]   = pts["transect_id"].map(idx["_angle_rad"])
    pts["_to_wgs"]      = pts["transect_id"].map(idx["_to_wgs"])

    # --- Option A: drop points with missing transect_id ---
    missing = pts["_to_wgs"].isna()
    if missing.any():
        print(f"Dropping {missing.sum()} points with missing transect_id.")
        pts = pts.loc[~missing].copy()

    # --- Compute projected UTM point coords ---
    dist = pts[cross_distance_column].to_numpy(float)
    angle = pts["_angle_rad"].to_numpy(float)

    shore_x = pts["_start_x_utm"].to_numpy(float) + np.cos(angle) * dist
    shore_y = pts["_start_y_utm"].to_numpy(float) + np.sin(angle) * dist

    # --- Transform back to WGS84 with tqdm ---
    lon_out = []
    lat_out = []

    for tr, x, y in tqdm(
        zip(pts["_to_wgs"], shore_x, shore_y),
        total=len(pts),
        desc="Transforming UTM→WGS84"
    ):
        lon, lat = tr.transform(x, y)
        lon_out.append(lon)
        lat_out.append(lat)

    # Update geometry
    pts.geometry = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(lon_out, lat_out)],
        crs="EPSG:4326"
    )

    # Cleanup
    pts = pts.drop(columns=["_start_x_utm", "_start_y_utm", "_angle_rad", "_to_wgs"],
                   errors="ignore")

    return pts


# #######tier0
# ##zoo_rgb
# print('projecting zoo rgb')
# out_gdf = project_points_seaward_wgs84(
#     tier_gpkg_path="/mnt/f/SDSDataService_c/14_tier_0.gpkg",
#     gpkg_layer='02_zoo_rgb_time_series',
#     index_gpkg_path="/mnt/f/SDSDataService_c_qc/14_index.gpkg",
#     transects_layer="01_transects_attributes",
#     cross_distance_column="cross_distance_tidally_corrected",
# )
# # Save — only geometry is changed
# out_gdf.to_file("/mnt/f/SDSDataService_c_qc/14_tier_0.gpkg",
#                 layer='02_zoo_rgb_time_series',
#                 driver="GPKG")
# out_gdf = None
# print('done projecting zoo rgb')
# ###nir
# print('projecting nir')
# out_gdf = project_points_seaward_wgs84(
#     tier_gpkg_path="/mnt/f/SDSDataService_c/14_tier_0.gpkg",
#     gpkg_layer='05_nir_time_series',
#     index_gpkg_path="/mnt/f/SDSDataService_c_qc/14_index.gpkg",
#     transects_layer="01_transects_attributes",
#     cross_distance_column="cross_distance_tidally_corrected",
# )
# # Save — only geometry is changed
# out_gdf.to_file("/mnt/f/SDSDataService_c_qc/14_tier_0.gpkg",
#                 layer='05_nir_time_series',
#                 driver="GPKG")
# out_gdf = None
# print('done projecing nir')
# ###swir
# print('projecting swir')
# out_gdf = project_points_seaward_wgs84(
#     tier_gpkg_path="/mnt/f/SDSDataService_c/14_tier_0.gpkg",
#     gpkg_layer='08_swir_time_series',
#     index_gpkg_path="/mnt/f/SDSDataService_c_qc/14_index.gpkg",
#     transects_layer="01_transects_attributes",
#     cross_distance_column="cross_distance_tidally_corrected",
# )
# # Save — only geometry is changed
# out_gdf.to_file("/mnt/f/SDSDataService_c_qc/14_tier_0.gpkg",
#                 layer='08_swir_time_series',
#                 driver="GPKG")
# out_gdf = None
# print('done prjojecting swir')