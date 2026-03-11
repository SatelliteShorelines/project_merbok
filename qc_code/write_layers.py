import geopandas as gpd
import numpy as np
from tqdm import tqdm
import os

# layers = [
#     '00_transects',
#     '01_transects_attributes',
#     '02_rois',
#     '03_reference_shorelines',
#     '04_reference_polygons',
#     '05_crest1s',
#     '06_crest2s',
#     '07_crest3s',
#     '08_inflection_points',
#     '09_toes'
# ]

# gpkg = "/mnt/f/SDSDataService_c_qc/14_index.gpkg"
# out_gpkg = "/mnt/f/SDSDataService_c_qc2/14_index.gpkg"

# chunk_size = 5000   # adjust as needed

# for layer in layers:
#     gdf = gpd.read_file(gpkg, layer=layer)
#     total = len(gdf)
#     print(f"Writing {layer} ({total} features)")

#     first = True

#     for start in tqdm(range(0, total, chunk_size)):
#         end = start + chunk_size
#         chunk = gdf.iloc[start:end]

#         chunk.to_file(
#             out_gpkg,
#             layer=layer,
#             driver="GPKG",
#             mode="w" if first else "a"
#         )

#         first = False
# layers = [
#     '00_zoo_rgb_waterlines',
#     '01_zoo_rgb_waterlines_filter',
#     '02_zoo_rgb_time_series',
#     '03_nir_waterlines',
#     '04_nir_waterlines_filter',
#     '05_nir_time_series',
#     '06_swir_waterlines',
#     '07_swir_waterlines_filter',
#     '08_swir_time_series'
# ]

# gpkg = "/mnt/f/SDSDataService_c_qc/14_tier_0.gpkg"
# out_gpkg = "/mnt/f/SDSDataService_c_qc2/14_tier_0.gpkg"

# chunk_size = 5000   # adjust as needed

# for layer in layers:
#     gdf = gpd.read_file(gpkg, layer=layer)
#     total = len(gdf)
#     print(f"Writing {layer} ({total} features)")

#     first = True

#     for start in tqdm(range(0, total, chunk_size)):
#         end = start + chunk_size
#         chunk = gdf.iloc[start:end]

#         chunk.to_file(
#             out_gpkg,
#             layer=layer,
#             driver="GPKG",
#             mode="w" if first else "a"
#         )

#         first = False


# layers = [
#     '00_unfiltered_shoreline_points',
#     '01_shoreline_points',
#     '02_shoreline_lines'
# ]

# gpkg = "/mnt/f/SDSDataService_c_qc/14_tier_1.gpkg"
# out_gpkg = "/mnt/f/SDSDataService_c_qc2/14_tier_1.gpkg"

# chunk_size = 5000   # adjust as needed

# for layer in layers:
#     gdf = gpd.read_file(gpkg, layer=layer)
#     total = len(gdf)
#     print(f"Writing {layer} ({total} features)")

#     first = True

#     for start in tqdm(range(0, total, chunk_size)):
#         end = start + chunk_size
#         chunk = gdf.iloc[start:end]

#         chunk.to_file(
#             out_gpkg,
#             layer=layer,
#             driver="GPKG",
#             mode="w" if first else "a"
#         )

#         first = False

# layers = [
#     '00_annual_shoreline_points',
#     '01_annual_shoreline_lines',
#     '02_annual_trends',
#     '03_decadal_shoreline_points',
#     '04_decadal_shoreline_lines',
#     '05_shoreline_change_envelopes'
# ]

# gpkg = "/mnt/f/SDSDataService_c_qc/14_tier_2.gpkg"
# out_gpkg = "/mnt/f/SDSDataService_c_qc2/14_tier_2.gpkg"

# chunk_size = 5000   # adjust as needed

# for layer in layers:
#     gdf = gpd.read_file(gpkg, layer=layer)
#     total = len(gdf)
#     print(f"Writing {layer} ({total} features)")

#     first = True

#     for start in tqdm(range(0, total, chunk_size)):
#         end = start + chunk_size
#         chunk = gdf.iloc[start:end]

#         chunk.to_file(
#             out_gpkg,
#             layer=layer,
#             driver="GPKG",
#             mode="w" if first else "a"
#         )

#         first = False

# layers = [
#     '00_min_shoreline_points',
#     '01_min_shoreline_lines',
#     '02_q1_shoreline_points',
#     '03_q1_shoreline_lines',
#     '04_mean_shoreline_points',
#     '05_mean_shoreline_lines',
#     '06_median_shoreline_points',
#     '07_median_shoreline_lines',
#     '08_q3_shoreline_points',
#     '09_q3_shoreline_lines',
#     '10_max_shoreline_points',
#     '11_max_shoreline_lines'
# ]

# gpkg = "/mnt/f/SDSDataService_c_qc/14_tier_3.gpkg"
# out_gpkg = "/mnt/f/SDSDataService_c_qc2/14_tier_3.gpkg"

# chunk_size = 5000   # adjust as needed

# for layer in layers:
#     gdf = gpd.read_file(gpkg, layer=layer)
#     total = len(gdf)
#     print(f"Writing {layer} ({total} features)")

#     first = True

#     for start in tqdm(range(0, total, chunk_size)):
#         end = start + chunk_size
#         chunk = gdf.iloc[start:end]

#         chunk.to_file(
#             out_gpkg,
#             layer=layer,
#             driver="GPKG",
#             mode="w" if first else "a"
#         )

#         first = False