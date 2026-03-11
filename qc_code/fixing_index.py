
import geopandas as gpd
import pandas as pd
import os
from pyproj import Transformer


def normalize_ids(gdf, key='transect_id'):
    """Ensure key is string, trimmed; return mutated gdf."""
    if key in gdf.columns:
        gdf[key] = gdf[key].astype(str).str.strip()
    return gdf


def report_uniqueness(name, gdf, key='transect_id', show=20):
    """Print a small report on uniqueness of key."""
    s = gdf[key].astype(str).str.strip().value_counts(dropna=False)
    n_dups = int((s > 1).sum())
    print(f"[{name}] rows={len(gdf)}, unique {key}={s.index.size}, duplicated IDs={n_dups}")
    if n_dups:
        print(f"[{name}] top duplicated {key} (count):")
        print(s[s > 1].head(show))


def dedup_by_first(gdf, key='transect_id'):
    """Collapse duplicates to a single row per key by taking the first occurrence."""
    gdf = normalize_ids(gdf, key)
    return (
        gdf.sort_values(key)  # stable selection; change sort if you prefer a rule
           .groupby(key, as_index=False)
           .first()
    )


def add_utm_attributes_to_transects(input_gpkg,
                                    layer_name='01_transects_attributes',
                                    output_gpkg=None):
    """
    Adds UTM-based attributes to transect LineStrings stored in WGS84.

    Creates these new fields:
        - utm_zone_epsg
        - utm_start_x, utm_start_y
        - utm_end_x, utm_end_y
        - utm_midpoint_x, utm_midpoint_y
    """

    gdf = gpd.read_file(input_gpkg, layer=layer_name)

    # Force CRS to WGS84 if missing or incorrect
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Helper: compute UTM EPSG code from latitude & longitude
    def utm_epsg(lat, lon):
        zone = int((lon + 180) // 6) + 1
        return 32600 + zone if lat >= 0 else 32700 + zone

    # Prepare output columns
    for col in [
        'utm_zone_epsg', 'utm_start_x', 'utm_start_y',
        'utm_end_x', 'utm_end_y', 'utm_midpoint_x', 'utm_midpoint_y'
    ]:
        if col not in gdf.columns:
            gdf[col] = None

    # Iterate rows
    for idx, geom in gdf.geometry.items():
        # Extract coordinates
        x0, y0 = geom.coords[0]
        x1, y1 = geom.coords[-1]
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2

        # Determine UTM zone from midpoint
        epsg = utm_epsg(ym, xm)
        transformer = Transformer.from_crs(4326, epsg, always_xy=True)

        # Transform
        sx, sy = transformer.transform(x0, y0)
        ex, ey = transformer.transform(x1, y1)
        mx, my = transformer.transform(xm, ym)

        # Store results
        gdf.at[idx, 'utm_zone_epsg'] = epsg
        gdf.at[idx, 'utm_start_x'] = sx
        gdf.at[idx, 'utm_start_y'] = sy
        gdf.at[idx, 'utm_end_x'] = ex
        gdf.at[idx, 'utm_end_y'] = ey
        gdf.at[idx, 'utm_midpoint_x'] = mx
        gdf.at[idx, 'utm_midpoint_y'] = my

    # Cast types
    gdf['utm_zone_epsg'] = gdf['utm_zone_epsg'].astype(int)
    for col in ['utm_start_x', 'utm_start_y', 'utm_end_x', 'utm_end_y', 'utm_midpoint_x', 'utm_midpoint_y']:
        gdf[col] = gdf[col].astype(float)

    # Write output
    if output_gpkg is None:
        gdf.to_file(input_gpkg, layer=layer_name, driver='GPKG')
    else:
        gdf.to_file(output_gpkg, layer=layer_name, driver='GPKG')


# --- Paths
index_path = os.path.join('/', 'mnt', 'f', 'SDSDataService_c', '14_index.gpkg')
new_index = os.path.join('/', 'mnt', 'f', 'SDSDataService_c_qc', '14_index.gpkg')

# --- Read inputs
transects = gpd.read_file(index_path, layer='00_transects')
arctic_dem_slopes = gpd.read_file(index_path, layer='09_ArcticDEM_slopes')
alaska_dsm_slopes = gpd.read_file(index_path, layer='10_AlaskaDSM_slopes')
tbdem_slopes = gpd.read_file(index_path, layer='11_tbdem_slopes')

# Normalize IDs early
transects = normalize_ids(transects, 'transect_id')
arctic_dem_slopes = normalize_ids(arctic_dem_slopes, 'transect_id')
alaska_dsm_slopes = normalize_ids(alaska_dsm_slopes, 'transect_id')
tbdem_slopes = normalize_ids(tbdem_slopes, 'transect_id')

# Columns of interest for slope tables
columns = ['transect_id', 'max_slope', 'median_slope', 'avg_slope', 'avg_slope_cleaned']
arctic_dem_slopes = arctic_dem_slopes[columns].rename(columns={
    'max_slope': 'max_slope_arctic_dem',
    'median_slope': 'median_slope_arctic_dem',
    'avg_slope': 'avg_slope_arctic_dem',
    'avg_slope_cleaned': 'avg_slope_cleaned_arctic_dem'
})
alaska_dsm_slopes = alaska_dsm_slopes[columns].rename(columns={
    'max_slope': 'max_slope_alaska_dsm',
    'median_slope': 'median_slope_alaska_dsm',
    'avg_slope': 'avg_slope_alaska_dsm',
    'avg_slope_cleaned': 'avg_slope_cleaned_alaska_dsm'
})
tbdem_slopes = tbdem_slopes[columns].rename(columns={
    'max_slope': 'max_slope_tbdem',
    'median_slope': 'median_slope_tbdem',
    'avg_slope': 'avg_slope_tbdem',
    'avg_slope_cleaned': 'avg_slope_cleaned_tbdem'
})

# --- Report uniqueness before merges
report_uniqueness("00_transects", transects)
report_uniqueness("arctic_dem_slopes", arctic_dem_slopes)
report_uniqueness("alaska_dsm_slopes", alaska_dsm_slopes)
report_uniqueness("tbdem_slopes", tbdem_slopes)

# --- Deduplicate slope tables if needed (use a chosen rule)
arctic_dem_slopes = dedup_by_first(arctic_dem_slopes)
alaska_dsm_slopes = dedup_by_first(alaska_dsm_slopes)
tbdem_slopes = dedup_by_first(tbdem_slopes)

# --- Merges with validation and post-merge checks
try:
    m1 = transects.merge(arctic_dem_slopes, on='transect_id', how='left', validate='m:1')
except Exception as e:
    print("Merge with ArcticDEM failed validation (expected m:1). Investigate duplicates in RHS:")
    report_uniqueness("arctic_dem_slopes (pre-merge)", arctic_dem_slopes)
    raise

report_uniqueness("after arctic_dem merge", m1)

try:
    m2 = m1.merge(alaska_dsm_slopes, on='transect_id', how='left', validate='m:1')
except Exception as e:
    print("Merge with AlaskaDSM failed validation (expected m:1). Investigate duplicates in RHS:")
    report_uniqueness("alaska_dsm_slopes (pre-merge)", alaska_dsm_slopes)
    raise

report_uniqueness("after alaska_dsm merge", m2)

try:
    m3 = m2.merge(tbdem_slopes, on='transect_id', how='left', validate='m:1')
except Exception as e:
    print("Merge with TBDem failed validation (expected m:1). Investigate duplicates in RHS:")
    report_uniqueness("tbdem_slopes (pre-merge)", tbdem_slopes)
    raise

report_uniqueness("after tbdem merge", m3)

# --- ESI shorezone geometry-based join
esi_path = os.path.join('/', 'mnt', 'f', 'SDSDataService_c_qc3_old', '14_index.gpkg')
esi_shorezone = gpd.read_file(esi_path, layer='01_transects_attributes')

# Build geometry key (WKT)
esi_shorezone["geom_key"] = esi_shorezone.geometry.to_wkt()
m3["geom_key"] = m3.geometry.to_wkt()

# Diagnose duplicates in ESI geom_key
esi_dup_geom = esi_shorezone["geom_key"].value_counts()
n_dup_geom = int((esi_dup_geom > 1).sum())
print(f"[esi_shorezone] duplicated geom_key count: {n_dup_geom}")
if n_dup_geom:
    print(esi_dup_geom[esi_dup_geom > 1].head(20))
    # Choose a rule: keep first per geometry
    esi_shorezone = esi_shorezone.sort_values("geom_key").drop_duplicates("geom_key", keep="first")

# Select only needed ESI columns before merge
esi_cols = ["geom_key", "shoretype_esi", "shoretype_shorezone"]
esi_shorezone = esi_shorezone[esi_cols]

# Merge ESI attributes; expect many-to-one by geom_key
try:
    merged = m3.merge(esi_shorezone, on="geom_key", how="left", validate="m:1")
except Exception as e:
    print("ESI merge failed validation (expected m:1). You have multiple ESI rows per geom_key.")
    print("Consider aggregating ESI per geom_key to a single row before merging.")
    raise

# --- Final column ordering
desired_order = [
    'G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
    'max_slope_arctic_dem', 'median_slope_arctic_dem',
    'avg_slope_arctic_dem', 'avg_slope_cleaned_arctic_dem',
    'max_slope_alaska_dsm', 'median_slope_alaska_dsm',
    'avg_slope_alaska_dsm', 'avg_slope_cleaned_alaska_dsm',
    'max_slope_tbdem', 'median_slope_tbdem', 'avg_slope_tbdem',
    'avg_slope_cleaned_tbdem', 'shoretype_esi', 'shoretype_shorezone', 'geometry'
]
# Keep only those that actually exist (some inputs may not have all of these)
desired_order = [c for c in desired_order if c in merged.columns]
merged = merged[desired_order]

# --- Assert uniqueness before write
counts = merged['transect_id'].astype(str).str.strip().value_counts()
n_dups_final = int((counts > 1).sum())
print(f"[final merged] duplicated transect_id count: {n_dups_final}")
if n_dups_final:
    print(counts[counts > 1].head(20))
    # If you want to hard-fail here:
    # raise ValueError("Duplicate transect_id found in final merged; check merges and inputs.")

# --- Write layers
transects.to_file(new_index, layer='00_transects')
merged.to_file(new_index, layer='01_transects_attributes')

layer_dict = {'01_rois':'02_rois',
              '02_reference_shorelines':'03_reference_shorelines',
              '03_reference_polygons':'04_reference_polygons',
              '04_crest1s':'05_crest1s',
              '05_crest2s':'06_crest2s',
              '06_crest3s':'07_crest3s',
              '07_inflection_points':'08_inflection_points',
              '08_toes':'09_toes'}

# Copy through other layers (drop extra columns if present)
for layer in layer_dict:
    gdf = gpd.read_file(index_path, layer=layer)
    for col in ['longshore_length', 'id']:
        if col in gdf.columns:
            gdf = gdf.drop(columns=[col])
    gdf.to_file(new_index, layer=layer_dict[layer])

# --- Add UTM attributes
add_utm_attributes_to_transects(new_index,
                                layer_name='01_transects_attributes',
                                output_gpkg=None)
