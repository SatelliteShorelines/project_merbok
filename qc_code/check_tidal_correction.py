import geopandas as gpd
import os
import pandas as pd
import numpy as np

def check_tidal_correction(gkpg_path, layer):
    gdf = gpd.read_file(gpkg_path, layer=layer)
    gdf['cross_distance_tidally_corrected'] = gdf['cross_distance'] - (gdf['tide']-0)/gdf['avg_slope_cleaned']
    gdf.to_file(gpkg_path,layer=layer)

# gpkg_path = os.path.join('/', 'mnt', 'f', 'SDSDataService_c_qc2', '14_tier_0.gpkg')
# layers = ['02_zoo_rgb_time_series', '05_nir_time_series', '08_swir_time_series']
# for layer in layers:
#     check_tidal_correction(gpkg_path, layer)

