import os
import shutil
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import LineString
import sys, time
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


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def tier_first_section(g, c, rr, sss, home, r_home_analysis):
    section = 'SSS'+sss
    section_dir = os.path.join(r_home_analysis, section)
    section_dir_data_service  = (os.path.join(r_home_data_service, section))
    section_string = g+c+rr+sss
    index_dir = os.path.join(home, 'Index', 'G'+g, 'C'+c, 'RR'+rr, 'SSS'+sss)
    tier_0_dir = os.path.join(home, 'Tier0', 'G'+g, 'C'+c, 'RR'+rr, 'SSS'+sss)
    tier_1_dir = os.path.join(home, 'Tier1', 'G'+g, 'C'+c, 'RR'+rr, 'SSS'+sss)
    tier_2_dir = os.path.join(home, 'Tier2', 'G'+g, 'C'+c, 'RR'+rr, 'SSS'+sss)
    tier_2_timeseries_dir = os.path.join(home, 'Tier2', 'G'+g, 'C'+c, 'RR'+rr, 'SSS'+sss, 'timeseries')
    tier_3_dir = os.path.join(home, 'Tier3', 'G'+g, 'C'+c, 'RR'+rr, 'SSS'+sss)
    dirs = [index_dir,tier_0_dir,tier_1_dir,tier_2_dir,tier_3_dir]
    for d in dirs:
        try:
            os.makedirs(d)
        except:
            pass

    ##Index Files: ROIs, reference shoreline, reference polygon, transects
    index = [
             os.path.join(section_dir, section_string + '_reference_shoreline.geojson'),
             os.path.join(section_dir, section_string + '_reference_polygon.geojson'),
             os.path.join(section_dir, section_string + '_transects.geojson'),
             os.path.join(section_dir, section_string + '_rois.geojson'),
             os.path.join(section_dir, section_string + '_transects_slopes_TBDEM.geojson'),
             os.path.join(section_dir, section_string + '_transects_slopes_AlaskaDSM.geojson'),
             os.path.join(section_dir, section_string + '_transects_slopes_.geojson'),
             os.path.join(section_dir, section_string + '_slopes_.csv'),
             os.path.join(section_dir, section_string + '_slopes_AlaskaDSM.csv'),
             os.path.join(section_dir, section_string + '_slopes_TBDEM.csv'),
             os.path.join(section_dir, section_string + '_tides.csv'),
             os.path.join(section_dir, section_string + '_esi_transects.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_crest_points_smooth.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_crest2_points_smooth.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_crest3_points_smooth.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_inflection_points_smooth.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_toe_points_smooth.geojson')
    ]
    
    ##Tier 0: Unfiltered raw and tidally corrected data 
    tier_0 = [os.path.join(section_dir, section_string + '_extracted_shorelines.geojson'), ##shorelines rgb
              os.path.join(section_dir, section_string + '_extracted_shorelines_filter.geojson'), ##filtered shorelines_rgb
              os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix.csv'), ##raw intersections matrix rgb
              os.path.join(section_dir, section_string + '_raw_transect_time_series_merged.csv'), ##raw intersections dataframe rgb
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_mat.csv'), ##tidally corrected intersections matrix rgb
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged.csv'), ##tidally corrected intersections dataframe rgb
              os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson'), ##shorelines nir
              os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh_filter.geojson'), ##filtered shorelines nir
              os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix_nir_thresh.csv'), ##raw intersections matrix nir
              os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_nir_thresh.csv'), ##raw intersections dataframe nir
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_mat_nir_thresh.csv'), ##tidally corrected intersections matrix nir
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_nir_thresh.csv'), ##tidally corrected intersections dataframe nir
              os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson'), ##shorelines swir
              os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh_filter.geojson'), ##filtered shorelines swir
              os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix_swir_thresh.csv'), ##raw intersections matrix swir
              os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_swir_thresh.csv'), ##raw intersections dataframe swir
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_mat_swir_thresh.csv'), ##tidally corrected intersections matrix swir
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_swir_thresh.csv'), ##tidally corrected intersections dataframe swir
    ]

    ##Tier 1: filtered/ensembled data
    tier_1 = [os.path.join(section_dir, section_string + '_unfiltered_tidally_corrected_points.geojson'),
              os.path.join(section_dir, section_string + '_filtered_tidally_corrected_transect_time_series_merged.csv'),
              os.path.join(section_dir, section_string + '_filtered_tidally_corrected_points.geojson'),
              os.path.join(section_dir, section_string + '_spatial_kde.tif'),
              os.path.join(section_dir, section_string + '_spatial_kde_otsu.tif'),
              os.path.join(section_dir, section_string + '_spatial_kde_otsu.geojson')] ##filtered and ensembled tidally corrected dataframe

    ##Tier 2: Resampled data
    tier_2 = [os.path.join(section_dir, section_string + '_resampled_tidally_corrected_transect_time_series_matrix.csv'), ##resampled with CI data matrix
              os.path.join(section_dir, section_string + '_resampled_tidally_corrected_transect_time_series_merged.csv'), ##resampled with CI dataframe
              os.path.join(section_dir, section_string + '_reprojected_points.geojson'), ##resampled shoreline points with CI
              os.path.join(section_dir, section_string + '_transects_trends.geojson'), ##transects with trend values (overall, 1980s, 1990s, 2000s, 2010s, 2020s)
              ]
    ##Tier 2: Individual timeseries
    final_timeseries = os.path.join(section_dir, 'final_timeseries')

    ##Tier 3: analysis outputs/metrics/more
    tier_3 = [os.path.join(section_dir, section_string + '_mean_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_mean_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_median_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_median_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_min_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_min_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_max_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_max_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_q1_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_q1_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_q3_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_q3_shoreline.geojson'),

    ]

    ##Other: Preliminary results
    other = [os.path.join(section_dir, section_string + '_extracted_shorelines_points_zoo_v1.geojson'),
             os.path.join(section_dir, section_string + '_extracted_shorelines_zoo_v1.geojson')
    ]

    ##copy index files
    for file in index:
        old_file = file
        new_file = os.path.join(index_dir, os.path.basename(file))
        try:
            shutil.copyfile(old_file, new_file)
        except:
            print(old_file)
    ##copy tier 0 files
    for file in tier_0:
        old_file = file
        new_file = os.path.join(tier_0_dir, os.path.basename(file))
        try:
            shutil.copyfile(old_file, new_file)
        except:
            print(old_file)
    ##copy tier 1 files
    for file in tier_1:
        old_file = file
        new_file = os.path.join(tier_1_dir, os.path.basename(file))
        try:
            shutil.copyfile(old_file, new_file)
        except:
            print(old_file)
    ##copy tier 2 files
    for file in tier_2:
        old_file = file
        new_file = os.path.join(tier_2_dir, os.path.basename(file))
        if old_file == os.path.join(section_dir, section_string + '_transects_trends.geojson'):
            gdf = gpd.read_file(old_file)
            fig_names = [os.path.join(tier_2_timeseries_dir, os.path.basename(fp)) for fp in gdf["figure_path"]]
            gdf['figure_path'] = fig_names
            csv_names = [os.path.join(tier_2_timeseries_dir, os.path.basename(fp)) for fp in gdf["csv_path"]]
            gdf['figure_path'] = fig_names
            gdf.to_file(new_file)
        
        else:
            try:
                shutil.copyfile(old_file, new_file)
            except:
                print(old_file)
    ##copy tier 3 files
    for file in tier_3:
        old_file = file
        new_file = os.path.join(tier_3_dir, os.path.basename(file))
        try:
            shutil.copyfile(old_file, new_file)
        except:
            print(old_file)

def tier_last_section(g, c, rr, sss, r_home_analysis, r_home_data_service):
    section = 'SSS'+sss
    section_dir = os.path.join(r_home_analysis, section)
    section_dir_data_service  = (os.path.join(r_home_data_service, section))
    section_string = g+c+rr+sss
    index_dir = os.path.join(section_dir_data_service, 'Index')
    tier_0_dir = os.path.join(section_dir_data_service, 'Tier0')
    tier_1_dir = os.path.join(section_dir_data_service, 'Tier1')
    tier_1_timeseries_dir = os.path.join(section_dir_data_service, 'Tier1', 'timeseries')
    tier_2_dir = os.path.join(section_dir_data_service, 'Tier2')
    tier_2_timeseries_dir = os.path.join(section_dir_data_service, 'Tier2', 'timeseries')
    tier_3_dir = os.path.join(section_dir_data_service, 'Tier3')
    dirs = [section_dir_data_service, index_dir,tier_0_dir,tier_1_dir,tier_2_dir,tier_3_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass

    ##Index Files: ROIs, reference shoreline, reference polygon, transects
    index = [
             os.path.join(section_dir, section_string + '_reference_shoreline.geojson'),
             os.path.join(section_dir, section_string + '_reference_polygon.geojson'),
             os.path.join(section_dir, section_string + '_transects.geojson'),
             os.path.join(section_dir, section_string + '_rois.geojson'),
             os.path.join(section_dir, section_string + '_transects_slopes_TBDEM.geojson'),
             os.path.join(section_dir, section_string + '_transects_slopes_AlaskaDSM.geojson'),
             os.path.join(section_dir, section_string + '_transects_slopes_.geojson'),
             os.path.join(section_dir, section_string + '_slopes_.csv'), ##ArcticDEM slopes dataframe
             os.path.join(section_dir, section_string + '_slopes_AlaskaDSM.csv'), ##AlaskaDSM slopes dataframe
             os.path.join(section_dir, section_string + '_slopes_TBDEM.csv'), ##TBDEM slopes dataframe
             os.path.join(section_dir, section_string + '_tides.csv'), ##tide dataframe
             os.path.join(section_dir, section_string + '_esi_transects.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_crest_points_smooth.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_crest2_points_smooth.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_crest3_points_smooth.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_inflection_points_smooth.geojson'),
             os.path.join(section_dir, 'elevation_profile_lines_', section_string + '_toe_points_smooth.geojson')
    ]
    
    ##Tier 0: Unfiltered raw and tidally corrected data 
    tier_0 = [os.path.join(section_dir, section_string + '_extracted_shorelines.geojson'), ##shorelines rgb
              os.path.join(section_dir, section_string + '_extracted_shorelines_filter.geojson'), ##filtered shorelines_rgb
              os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix.csv'), ##raw intersections matrix rgb
              os.path.join(section_dir, section_string + '_raw_transect_time_series_merged.csv'), ##raw intersections dataframe rgb
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_mat.csv'), ##tidally corrected intersections matrix rgb
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged.csv'), ##tidally corrected intersections dataframe rgb
              os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson'), ##shorelines nir
              os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh_filter.geojson'), ##filtered shorelines nir
              os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix_nir_thresh.csv'), ##raw intersections matrix nir
              os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_nir_thresh.csv'), ##raw intersections dataframe nir
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_mat_nir_thresh.csv'), ##tidally corrected intersections matrix nir
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_nir_thresh.csv'), ##tidally corrected intersections dataframe nir
              os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson'), ##shorelines swir
              os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh_filter.geojson'), ##filtered shorelines swir
              os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix_swir_thresh.csv'), ##raw intersections matrix swir
              os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_swir_thresh.csv'), ##raw intersections dataframe swir
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_mat_swir_thresh.csv'), ##tidally corrected intersections matrix swir
              os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_swir_thresh.csv'), ##tidally corrected intersections dataframe swir
    ]

    ##Tier 1: filtered/ensembled data
    tier_1 = [os.path.join(section_dir, section_string + '_unfiltered_tidally_corrected_points.geojson'),
              os.path.join(section_dir, section_string + '_filtered_tidally_corrected_transect_time_series_merged.csv'),
              os.path.join(section_dir, section_string + '_filtered_tidally_corrected_points.geojson'),
              os.path.join(section_dir, section_string + '_spatial_kde.tif'),
              os.path.join(section_dir, section_string + '_spatial_kde_otsu.tif'),
              os.path.join(section_dir, section_string + '_spatial_kde_otsu.geojson')] ##filtered and ensembled tidally corrected dataframe
    ##tier 1 ensembel timeseries data
    ensembled_timeseries = os.path.join(section_dir, section_string + 'ensemble_timeseries')

    ##Tier 2: Resampled data
    tier_2 = [os.path.join(section_dir, section_string + '_resampled_tidally_corrected_transect_time_series_matrix.csv'), ##resampled with CI data matrix
              os.path.join(section_dir, section_string + '_resampled_tidally_corrected_transect_time_series_merged.csv'), ##resampled with CI dataframe
              os.path.join(section_dir, section_string + '_reprojected_points.geojson'), ##resampled shoreline points with CI
              os.path.join(section_dir, section_string + '_transects_trends.geojson') ##transects with trend values (overall, 1980s, 1990s, 2000s, 2010s, 2020s)
              ]

    ##Tier 2: Individual timeseries
    resampled_timeseries = os.path.join(section_dir, 'resampled_timeseries')

    ##Tier 3: analysis outputs/metrics/more
    tier_3 = [os.path.join(section_dir, section_string + '_mean_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_mean_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_median_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_median_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_min_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_min_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_max_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_max_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_q1_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_q1_shoreline.geojson'),
              os.path.join(section_dir, section_string + '_q3_shoreline_points.geojson'),
              os.path.join(section_dir, section_string + '_q3_shoreline.geojson')
    ]

    ##Other: Preliminary results
    other = [os.path.join(section_dir, section_string + '_extracted_shorelines_points_zoo_v1.geojson'),
             os.path.join(section_dir, section_string + '_extracted_shorelines_zoo_v1.geojson')
    ]

    ##copy index files
    for file in index:
        old_file = file
        new_file = os.path.join(index_dir, os.path.basename(file))
        try:
            shutil.copyfile(old_file, new_file)
        except:
            print(old_file)
    ##copy tier 0 files
    for file in tier_0:
        old_file = file
        new_file = os.path.join(tier_0_dir, os.path.basename(file))
        try:
            shutil.copyfile(old_file, new_file)
        except:
            print(old_file)
    ##copy tier 1 files
    for file in tier_1:
        old_file = file
        new_file = os.path.join(tier_1_dir, os.path.basename(file))
        try:
            shutil.copyfile(old_file, new_file)
        except:
            print(old_file)
    ##copy tier 2 files
    for file in tier_2:
        old_file = file
        new_file = os.path.join(tier_2_dir, os.path.basename(file))
        if old_file == os.path.join(section_dir, section_string + '_transects_trends.geojson'):
            gdf = gpd.read_file(old_file)
            fig_names = [os.path.join(tier_2_timeseries_dir, os.path.basename(fp)) for fp in gdf["figure_path"]]
            gdf['figure_path'] = fig_names
            csv_names = [os.path.join(tier_2_timeseries_dir, os.path.basename(fp)) for fp in gdf["csv_path"]]
            gdf['figure_path'] = fig_names
            gdf.to_file(new_file)
        
        else:
            try:
                shutil.copyfile(old_file, new_file)
            except:
                print(old_file)
    ##copy tier 3 files
    for file in tier_3:
        old_file = file
        new_file = os.path.join(tier_3_dir, os.path.basename(file))
        try:
            shutil.copyfile(old_file, new_file)
        except:
            print(old_file)


def get_tiered_files(g, c, rr, sss, r_home_data_service):
    section = 'SSS'+sss
    section_dir = os.path.join(r_home_data_service, section)
    section_dir_data_service  = (os.path.join(r_home_data_service, section))
    section_string = g+c+rr+sss
    index_dir = os.path.join(section_dir_data_service, 'Index')
    tier_0_dir = os.path.join(section_dir_data_service, 'Tier0')
    tier_1_dir = os.path.join(section_dir_data_service, 'Tier1')
    tier_2_dir = os.path.join(section_dir_data_service, 'Tier2')
    tier_2_timeseries_dir = {'timeseries_dir':os.path.join(section_dir_data_service, 'Tier2', 'timeseries')}
    tier_3_dir = os.path.join(section_dir_data_service, 'Tier3')

    ##Index Files: ROIs, reference shoreline, reference polygon, transects
    index = {
             'reference_shoreline_gdf':os.path.join(index_dir, section_string + '_reference_shoreline.geojson'), 
             'reference_polygon_gdf':os.path.join(index_dir, section_string + '_reference_polygon.geojson'),
             'transects_gdf':os.path.join(index_dir, section_string + '_transects.geojson'),
             'rois_gdf':os.path.join(index_dir, section_string + '_rois.geojson'),
             'slopes_TBDEM_gdf':os.path.join(index_dir, section_string + '_transects_slopes_TBDEM.geojson'),
             'slopes_AlaskaDSM_gdf':os.path.join(index_dir, section_string + '_transects_slopes_AlaskaDSM.geojson'),
             'slopes_ArcticDEM_gdf':os.path.join(index_dir, section_string + '_transects_slopes_.geojson'),
             'slopes_ArcticDEM_df':os.path.join(index_dir, section_string + '_slopes_.csv'), ##ArcticDEM slopes dataframe
             'slopes_AlaskaDSM_df':os.path.join(index_dir, section_string + '_slopes_AlaskaDSM.csv'), ##AlaskaDSM slopes dataframe
             'slopes_TBDEM_df':os.path.join(index_dir, section_string + '_slopes_TBDEM.csv'), ##TBDEM slopes dataframe
             'tides_df':os.path.join(index_dir, section_string + '_tides.csv'), ##tide dataframe
             'esi_gdf':os.path.join(index_dir, section_string + '_esi_transects.geojson') ##esi classification
    }
    
    ##Tier 0: Unfiltered raw and tidally corrected data 
    tier_0 = {'zoo_rgb_shorelines_gdf':os.path.join(tier_0_dir, section_string + '_extracted_shorelines.geojson'), ##shorelines rgb
              'zoo_rgb_shorelines_filter_gdf':os.path.join(tier_0_dir, section_string + '_extracted_shorelines_filter.geojson'), ##filtered shorelines_rgb
              'zoo_rgb_time_series_stacked_df':os.path.join(tier_0_dir, section_string + '_raw_transect_time_series_merged.csv'), ##raw intersections dataframe rgb
              'zoo_rgb_tidally_corrected_time_series_stacked_df':os.path.join(tier_0_dir, section_string + '_tidally_corrected_transect_time_series_merged.csv'), ##tidally corrected intersections dataframe rgb
              'nir_thresh_shorelines_gdf':os.path.join(tier_0_dir, section_string + '_extracted_shorelines_nir_thresh.geojson'), ##shorelines nir
              'nir_thresh_shorelines_filter_gdf':os.path.join(tier_0_dir, section_string + '_extracted_shorelines_nir_thresh_filter.geojson'), ##filtered shorelines nir
              'nir_thresh_time_series_stacked_df':os.path.join(tier_0_dir, section_string + '_raw_transect_time_series_merged_nir_thresh.csv'), ##raw intersections dataframe nir
              'nir_thresh_tidally_corrected_time_series_stacked_df':os.path.join(tier_0_dir, section_string + '_tidally_corrected_transect_time_series_merged_nir_thresh.csv'), ##tidally corrected intersections dataframe nir
              'nir_thresh_shorelines_gdf':os.path.join(tier_0_dir, section_string + '_extracted_shorelines_swir_thresh.geojson'), ##shorelines swir
              'nir_thresh_shorelines_filter_gdf':os.path.join(tier_0_dir, section_string + '_extracted_shorelines_swir_thresh_filter.geojson'), ##filtered shorelines swir
              'swir_thresh_time_series_stacked_df':os.path.join(tier_0_dir, section_string + '_raw_transect_time_series_merged_swir_thresh.csv'), ##raw intersections dataframe swir
              'swir_thresh_tidally_corrected_time_series_stacked_df':os.path.join(tier_0_dir, section_string + '_tidally_corrected_transect_time_series_merged_swir_thresh.csv'), ##tidally corrected intersections dataframe swir
    }

    ##Tier 1: filtered/ensembled data
    tier_1 = {'filtered_tidally_corrected_time_series_stacked_df':os.path.join(tier_1_dir, section_string + '_filtered_tidally_corrected_transect_time_series_merged.csv'),
              'spatial_kde_tif':os.path.join(tier_1_dir, section_string + '_spatial_kde.tif'),
              'spatial_kde_otsu_tif':os.path.join(tier_1_dir, section_string + '_spatial_kde_otsu.tif'),
              'spatial_kde_otsu_gdf':os.path.join(tier_1_dir, section_string + '_spatial_kde_otsu.geojson')} ##filtered and ensembled tidally corrected dataframe

    ##Tier 2: Resampled data
    tier_2 = {'resampled_reprojected_time_series_stacked_df':os.path.join(tier_2_dir, section_string + '_resampled_tidally_corrected_transect_time_series_merged.csv'), ##resampled with CI dataframe
              'resampled_reprojected_shorelines_gdf':os.path.join(tier_2_dir, section_string + '_reprojected_points.geojson'), ##resampled shoreline points with CI
              'trends_gdf':os.path.join(tier_2_dir, section_string + '_transects_trends.geojson') ##transects with trend values (overall, 1980s, 1990s, 2000s, 2010s, 2020s)
    }

    ##Tier 2: Individual timeseries
    final_timeseries = os.path.join(section_dir, 'final_timeseries')

    ##Tier 3: analysis outputs/metrics/more
    tier_3 = {'mean_shorelines_points_gdf':os.path.join(tier_3_dir, section_string + '_mean_shoreline_points.geojson'),
              'mean_shorelines_line_gdf':os.path.join(tier_3_dir, section_string + '_mean_shoreline.geojson'),
              'median_shorelines_points_gdf':os.path.join(tier_3_dir, section_string + '_median_shoreline_points.geojson'),
              'median_shorelines_line_gdf':os.path.join(tier_3_dir, section_string + '_median_shoreline.geojson'),
              'min_shorelines_points_gdf':os.path.join(tier_3_dir, section_string + '_min_shoreline_points.geojson'),
              'min_shorelines_line_gdf':os.path.join(tier_3_dir, section_string + '_min_shoreline.geojson'),
              'max_shorelines_points_gdf':os.path.join(tier_3_dir, section_string + '_max_shoreline_points.geojson'),
              'max_shorelines_line_gdf':os.path.join(tier_3_dir, section_string + '_max_shoreline.geojson'),
              'q1_shorelines_points_gdf':os.path.join(tier_3_dir, section_string + '_q1_shoreline_points.geojson'),
              'q1_shorelines_line_gdf':os.path.join(tier_3_dir, section_string + '_q1_shoreline.geojson'),
              'q3_shorelines_points_gdf':os.path.join(tier_3_dir, section_string + '_q3_shoreline_points.geojson'),
              'q3_shorelines_line_gdf':os.path.join(tier_3_dir, section_string + '_q3_shoreline.geojson'),
    }

    ##Other: Preliminary results
    other = {os.path.join(section_dir, section_string + '_extracted_shorelines_points_zoo_v1.geojson'),
             os.path.join(section_dir, section_string + '_extracted_shorelines_zoo_v1.geojson')
    }

    return sss, index, tier_0, tier_1, tier_2, tier_2_timeseries_dir, tier_3

def gdf_dictionary_to_geopackage(dictionary, geopackage_path):
    for key in dictionary.keys():
        gdf = dictionary[key]
        gdf.to_file(geopackage_path, layer=key, driver="GPKG")

def tif_dictionary_to_geopackage(dictionary, geopackage_path):
    for key in dictionary.keys():
        tif_list = dictionary[key]
        for tif in tif_list:
            cmd = 'gdal_translate -of GPKG ' + tif + ' ' + geopackage_path + ' -co APPEND_SUBDATASET=YES -co RASTER_TABLE=' + os.path.splitext(os.path.basename(tif))[0]
            os.system(cmd)

def df_to_gdf(df_path,geom_type):
    gdf_path = os.path.splitext(df_path)[0]+'.geojson'
    df = pd.read_csv(df_path)
    if geom_type == 'xy':
        geometry = gpd.points_from_xy(df['x'], df['y'])
    elif geom_type == 'shorexy':
        geometry = gpd.points_from_xy(df['shore_x'], df['shore_y'])
    elif geom_type == 'intersectxy':
        geometry = gpd.points_from_xy(df['intersect_x'], df['intersect_y'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326") # Add CRS if needed
    gdf.to_file(gdf_path)
    return gdf_path

def fix_cols_gdf(gdf_path, g, c, rr, sss):
    if os.path.isfile(gdf_path)==True:
        gdf = gpd.read_file(gdf_path)
        gdf['G'] = [g]*len(gdf)
        gdf['C'] = [c]*len(gdf)
        gdf['RR'] = [rr]*len(gdf)
        gdf['SSS'] = [sss]*len(gdf)
        try:
            gdf = gdf.drop(columns=['fid'])
        except:
            pass
        try:
            gdf = gdf.drop(columns=['OBJECTID'])
        except:
            pass
        try:
            gdf = gdf.drop(columns=['Shape_Length'])
        except:
            pass 
        try:
            gdf = gdf.drop(columns=['index_right'])
        except:
            pass 
        try:
            gdf = gdf.drop(columns=['Unnamed: 0'])
        except:
            pass
        
        gdf.to_file(gdf_path)

def fix_cols_df(df_path, g, c, rr, sss):
    if os.path.isfile(df_path)==True:
        df = gpd.read_file(df_path)
        df['G'] = [g]*len(df)
        df['C'] = [c]*len(df)
        df['RR'] = [rr]*len(df)
        df['SSS'] = [sss]*len(df)
        try:
            df = df.drop(columns=['Unnamed: 0'])
        except:
            pass
        try:
            df = df.drop(columns=['field_1'])
        except:
            pass 
        try:
            df = df.drop(columns=['geometry'])
        except:
            pass 
        df.to_csv(df_path)

def tier_to_gpkg(g, c, rrs, home_data_service, home_geopackage, tiers):
    print('getting number of sections')
    ##getting number of sections
    all_sections = []
    for rr in rrs:
        r_home = os.path.join(home_data_service, 'Index', 'G'+g, 'C'+c, 'RR'+rr)
        sections = sorted(get_immediate_subdirectories(r_home))
        sections_full_path = [os.path.join(r_home, section) for section in sections]
        all_sections.append(sections_full_path)
    flat_all_sections = []
    for sublist in all_sections:
        for item in sublist:
            flat_all_sections.append(item)
    num_sections = len(flat_all_sections)
    print('number of sections ' + str(num_sections))

    if 'index' in tiers:
        print('getting index files')
        ##getting index files
        index_reference_shoreline_gdfs = [None]*num_sections
        index_reference_polygon_gdfs = [None]*num_sections
        index_transect_gdfs = [None]*num_sections
        index_roi_gdfs = [None]*num_sections
        index_crest1_gdfs = [None]*num_sections
        index_crest2_gdfs = [None]*num_sections
        index_crest3_gdfs = [None]*num_sections
        index_inflection_point_gdfs = [None]*num_sections
        index_toe_gdfs = [None]*num_sections
        index_slopes_TBDEM_gdfs = [None]*num_sections
        index_slopes_AlaskaDSM_gdfs = [None]*num_sections
        index_slopes_ArcticDEM_gdfs = [None]*num_sections
        index_tides_dfs = [None]*num_sections
        index_esi_gdfs = [None]*num_sections
        i=0
        for rr in tqdm(rrs):
            r_home = os.path.join(home_data_service, 'Index', 'G'+g, 'C'+c, 'RR'+rr)
            sections = sorted(get_immediate_subdirectories(r_home))
            for section in tqdm(sections):
                section_dir = os.path.join(r_home, section)
                sss = section[3:]
                section_string = g+c+rr+sss
                reference_shoreline_path = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
                fix_cols_gdf(reference_shoreline_path, g, c, rr, sss)
                index_reference_shoreline_gdfs[i] = reference_shoreline_path

                reference_polygon_path = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
                fix_cols_gdf(reference_polygon_path, g, c, rr, sss)
                index_reference_polygon_gdfs[i] = reference_polygon_path 

                transects_path = os.path.join(section_dir, section_string + '_transects.geojson')
                fix_cols_gdf(transects_path, g, c, rr, sss)
                index_transect_gdfs[i] = transects_path

                rois_path = os.path.join(section_dir, section_string + '_rois.geojson')
                fix_cols_gdf(rois_path, g, c, rr, sss)
                index_roi_gdfs[i] = rois_path

                crest1_path = os.path.join(section_dir, section_string + '_crest_points_smooth.geojson')
                fix_cols_gdf(crest1_path, g, c, rr, sss)
                index_crest1_gdfs[i] = crest1_path

                crest2_path = os.path.join(section_dir, section_string + '_crest2_points_smooth.geojson')
                fix_cols_gdf(crest2_path, g, c, rr, sss)
                index_crest2_gdfs[i] = crest2_path

                crest3_path = os.path.join(section_dir, section_string + '_crest3_points_smooth.geojson')
                fix_cols_gdf(crest3_path, g, c, rr, sss)
                index_crest3_gdfs[i] = crest3_path

                inflection_point_path = os.path.join(section_dir, section_string + '_inflection_points_smooth.geojson')
                fix_cols_gdf(inflection_point_path, g, c, rr, sss)
                index_inflection_point_gdfs[i] = inflection_point_path

                toe_path = os.path.join(section_dir, section_string + '_toe_points_smooth.geojson')
                fix_cols_gdf(toe_path, g, c, rr, sss)
                index_toe_gdfs[i] = toe_path

                slopes_tbdem_path = os.path.join(section_dir, section_string + '_transects_slopes_TBDEM.geojson')
                fix_cols_gdf(slopes_tbdem_path, g, c, rr, sss)
                index_slopes_TBDEM_gdfs[i] = slopes_tbdem_path

                slopes_AlaskaDSM_path = os.path.join(section_dir, section_string + '_transects_slopes_AlaskaDSM.geojson')
                fix_cols_gdf(slopes_AlaskaDSM_path, g, c, rr, sss)
                index_slopes_AlaskaDSM_gdfs[i] = slopes_AlaskaDSM_path

                slopes_ArcticDEM_path = os.path.join(section_dir, section_string + '_transects_slopes_.geojson')
                fix_cols_gdf(slopes_ArcticDEM_path, g, c, rr, sss)            
                index_slopes_ArcticDEM_gdfs[i] = slopes_ArcticDEM_path

                i=i+1
        print('concatenating index files')
        ##Concatenating files
        #index files
        print('ref shorelines')
        index_reference_shoreline_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_reference_shoreline_gdfs) if os.path.isfile(f)==True])
        print('ref polygons')
        index_reference_polygon_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_reference_polygon_gdfs) if os.path.isfile(f)==True])
        print('transects')
        index_transect_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_transect_gdfs) if os.path.isfile(f)==True])
        print('rois')
        index_roi_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_roi_gdfs) if os.path.isfile(f)==True])
        print('crest1s')
        index_crest1_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_crest1_gdfs) if os.path.isfile(f)==True])
        print('crest2s')
        index_crest2_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_crest2_gdfs) if os.path.isfile(f)==True])
        print('crest3s')
        index_crest3_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_crest3_gdfs) if os.path.isfile(f)==True])
        print('inflection points')
        index_inflection_point_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_inflection_point_gdfs) if os.path.isfile(f)==True])
        print('toes')
        index_toe_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_toe_gdfs) if os.path.isfile(f)==True])
        print('slopes tbdem')
        index_slopes_TBDEM_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_slopes_TBDEM_gdfs) if os.path.isfile(f)==True])
        print('slopes alaska dsm')
        index_slopes_AlaskaDSM_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_slopes_AlaskaDSM_gdfs) if os.path.isfile(f)==True])
        print('slopes arctic dem')
        index_slopes_ArcticDEM_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(index_slopes_ArcticDEM_gdfs) if os.path.isfile(f)==True])
        print('making index dictionaries')


        index_transect_gdfs_concat['G'] = index_transect_gdfs_concat['G'].astype(str)
        index_transect_gdfs_concat['C'] = index_transect_gdfs_concat['C'].astype(str)
        index_transect_gdfs_concat['RR'] = index_transect_gdfs_concat['RR'].astype(str)
        index_transect_gdfs_concat['SSS'] = index_transect_gdfs_concat['SSS'].astype(str)
        index_transect_gdfs_concat['V'] = index_transect_gdfs_concat['V'].astype(str)
        index_transect_gdfs_concat['LLLLLL'] = [l[-6:] for l in index_transect_gdfs_concat['transect_id']]
        index_transect_gdfs_concat['LLLLLL'] = (index_transect_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        index_transect_gdfs_concat['transect_id'] = index_transect_gdfs_concat['transect_id'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 'geometry']
        index_transect_gdfs_concat = index_transect_gdfs_concat[column_order]

        index_roi_gdfs_concat['G'] = index_roi_gdfs_concat['G'].astype(str)
        index_roi_gdfs_concat['C'] = index_roi_gdfs_concat['C'].astype(str)
        index_roi_gdfs_concat['RR'] = index_roi_gdfs_concat['RR'].astype(str)
        index_roi_gdfs_concat['SSS'] = index_roi_gdfs_concat['SSS'].astype(str)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
        index_roi_gdfs_concat = index_roi_gdfs_concat[columns_to_keep]

        index_reference_shoreline_gdfs_concat['G'] = index_reference_shoreline_gdfs_concat['G'].astype(str)
        index_reference_shoreline_gdfs_concat['C'] = index_reference_shoreline_gdfs_concat['C'].astype(str)
        index_reference_shoreline_gdfs_concat['RR'] = index_reference_shoreline_gdfs_concat['RR'].astype(str)
        index_reference_shoreline_gdfs_concat['SSS'] = index_reference_shoreline_gdfs_concat['SSS'].astype(str)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
        index_reference_shoreline_gdfs_concat = index_reference_shoreline_gdfs_concat[columns_to_keep]

        index_reference_polygon_gdfs_concat['G'] = index_reference_polygon_gdfs_concat['G'].astype(str)
        index_reference_polygon_gdfs_concat['C'] = index_reference_polygon_gdfs_concat['C'].astype(str)
        index_reference_polygon_gdfs_concat['RR'] = index_reference_polygon_gdfs_concat['RR'].astype(str)
        index_reference_polygon_gdfs_concat['SSS'] = index_reference_polygon_gdfs_concat['SSS'].astype(str)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
        index_reference_polygon_gdfs_concat = index_reference_polygon_gdfs_concat[columns_to_keep]

        index_crest1_gdfs_concat['G'] = index_crest1_gdfs_concat['G'].astype(str)
        index_crest1_gdfs_concat['C'] = index_crest1_gdfs_concat['C'].astype(str)
        index_crest1_gdfs_concat['RR'] = index_crest1_gdfs_concat['RR'].astype(str)
        index_crest1_gdfs_concat['SSS'] = index_crest1_gdfs_concat['SSS'].astype(str)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
        index_crest1_gdfs_concat = index_crest1_gdfs_concat[columns_to_keep]

        index_crest2_gdfs_concat['G'] = index_crest2_gdfs_concat['G'].astype(str)
        index_crest2_gdfs_concat['C'] = index_crest2_gdfs_concat['C'].astype(str)
        index_crest2_gdfs_concat['RR'] = index_crest2_gdfs_concat['RR'].astype(str)
        index_crest2_gdfs_concat['SSS'] = index_crest2_gdfs_concat['SSS'].astype(str)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
        index_crest2_gdfs_concat = index_crest2_gdfs_concat[columns_to_keep]

        index_crest3_gdfs_concat['G'] = index_crest3_gdfs_concat['G'].astype(str)
        index_crest3_gdfs_concat['C'] = index_crest3_gdfs_concat['C'].astype(str)
        index_crest3_gdfs_concat['RR'] = index_crest3_gdfs_concat['RR'].astype(str)
        index_crest3_gdfs_concat['SSS'] = index_crest3_gdfs_concat['SSS'].astype(str)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
        index_crest3_gdfs_concat = index_crest3_gdfs_concat[columns_to_keep]

        index_inflection_point_gdfs_concat['G'] = index_inflection_point_gdfs_concat['G'].astype(str)
        index_inflection_point_gdfs_concat['C'] = index_inflection_point_gdfs_concat['C'].astype(str)
        index_inflection_point_gdfs_concat['RR'] = index_inflection_point_gdfs_concat['RR'].astype(str)
        index_inflection_point_gdfs_concat['SSS'] = index_inflection_point_gdfs_concat['SSS'].astype(str)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
        index_inflection_point_gdfs_concat = index_inflection_point_gdfs_concat[columns_to_keep]

        index_toe_gdfs_concat['G'] = index_toe_gdfs_concat['G'].astype(str)
        index_toe_gdfs_concat['C'] = index_toe_gdfs_concat['C'].astype(str)
        index_toe_gdfs_concat['RR'] = index_toe_gdfs_concat['RR'].astype(str)
        index_toe_gdfs_concat['SSS'] = index_toe_gdfs_concat['SSS'].astype(str)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
        index_toe_gdfs_concat = index_toe_gdfs_concat[columns_to_keep]

        index_slopes_ArcticDEM_gdfs_concat['G'] = index_slopes_ArcticDEM_gdfs_concat['G'].astype(str)
        index_slopes_ArcticDEM_gdfs_concat['C'] = index_slopes_ArcticDEM_gdfs_concat['C'].astype(str)
        index_slopes_ArcticDEM_gdfs_concat['RR'] = index_slopes_ArcticDEM_gdfs_concat['RR'].astype(str)
        index_slopes_ArcticDEM_gdfs_concat['SSS'] = index_slopes_ArcticDEM_gdfs_concat['SSS'].astype(str)
        index_slopes_ArcticDEM_gdfs_concat['V'] = ['0']*len(index_slopes_ArcticDEM_gdfs_concat)
        index_slopes_ArcticDEM_gdfs_concat['transect_id'] = index_slopes_ArcticDEM_gdfs_concat['transect_id'].astype(str)
        index_slopes_ArcticDEM_gdfs_concat['LLLLLL'] = [l[-6:] for l in index_slopes_ArcticDEM_gdfs_concat['transect_id']]
        index_slopes_ArcticDEM_gdfs_concat['LLLLLL'] = (index_slopes_ArcticDEM_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        index_slopes_ArcticDEM_gdfs_concat['max_slope'] = index_slopes_ArcticDEM_gdfs_concat['max_slope'].astype(float)
        index_slopes_ArcticDEM_gdfs_concat['median_slope'] = index_slopes_ArcticDEM_gdfs_concat['median_slope'].astype(float)
        index_slopes_ArcticDEM_gdfs_concat['avg_slope_cleaned'] = index_slopes_ArcticDEM_gdfs_concat['avg_slope_cleaned'].astype(float)
        index_slopes_ArcticDEM_gdfs_concat['avg_slope'] = index_slopes_ArcticDEM_gdfs_concat['avg_slope'].astype(float)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL',
                        'transect_id', 
                        'max_slope', 'median_slope', 'avg_slope_cleaned', 'avg_slope', 'geometry']
        for col in index_slopes_ArcticDEM_gdfs_concat.columns:
            if col not in columns_to_keep:
                try:
                    index_slopes_ArcticDEM_gdfs_concat = index_slopes_ArcticDEM_gdfs_concat.drop(columns=[col])
                except:
                    pass
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'max_slope', 'median_slope', 'avg_slope', 
                        'avg_slope_cleaned',
                        'geometry']
        index_slopes_ArcticDEM_gdfs_concat = index_slopes_ArcticDEM_gdfs_concat[column_order]

        index_slopes_AlaskaDSM_gdfs_concat['G'] = index_slopes_AlaskaDSM_gdfs_concat['G'].astype(str)
        index_slopes_AlaskaDSM_gdfs_concat['C'] = index_slopes_AlaskaDSM_gdfs_concat['C'].astype(str)
        index_slopes_AlaskaDSM_gdfs_concat['RR'] = index_slopes_AlaskaDSM_gdfs_concat['RR'].astype(str)
        index_slopes_AlaskaDSM_gdfs_concat['SSS'] = index_slopes_AlaskaDSM_gdfs_concat['SSS'].astype(str)
        index_slopes_AlaskaDSM_gdfs_concat['V'] = ['0']*len(index_slopes_AlaskaDSM_gdfs_concat)
        index_slopes_AlaskaDSM_gdfs_concat['transect_id'] = index_slopes_AlaskaDSM_gdfs_concat['transect_id'].astype(str)
        index_slopes_AlaskaDSM_gdfs_concat['LLLLLL'] = [l[-6:] for l in index_slopes_AlaskaDSM_gdfs_concat['transect_id']]
        index_slopes_AlaskaDSM_gdfs_concat['LLLLLL'] = (index_slopes_AlaskaDSM_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        index_slopes_AlaskaDSM_gdfs_concat['max_slope'] = index_slopes_AlaskaDSM_gdfs_concat['max_slope'].astype(float)
        index_slopes_AlaskaDSM_gdfs_concat['median_slope'] = index_slopes_AlaskaDSM_gdfs_concat['median_slope'].astype(float)
        index_slopes_AlaskaDSM_gdfs_concat['avg_slope_cleaned'] = index_slopes_AlaskaDSM_gdfs_concat['avg_slope_cleaned'].astype(float)
        index_slopes_AlaskaDSM_gdfs_concat['avg_slope'] = index_slopes_AlaskaDSM_gdfs_concat['avg_slope'].astype(float)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL',
                        'transect_id', 
                        'max_slope', 'median_slope', 'avg_slope_cleaned', 'avg_slope', 'geometry']
        for col in index_slopes_AlaskaDSM_gdfs_concat.columns:
            if col not in columns_to_keep:
                try:
                    index_slopes_AlaskaDSM_gdfs_concat = index_slopes_AlaskaDSM_gdfs_concat.drop(columns=[col])
                except:
                    pass
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'max_slope', 'median_slope', 'avg_slope', 
                        'avg_slope_cleaned',
                        'geometry']
        index_slopes_AlaskaDSM_gdfs_concat = index_slopes_AlaskaDSM_gdfs_concat[column_order]

        index_slopes_TBDEM_gdfs_concat['G'] = index_slopes_TBDEM_gdfs_concat['G'].astype(str)
        index_slopes_TBDEM_gdfs_concat['C'] = index_slopes_TBDEM_gdfs_concat['C'].astype(str)
        index_slopes_TBDEM_gdfs_concat['RR'] = index_slopes_TBDEM_gdfs_concat['RR'].astype(str)
        index_slopes_TBDEM_gdfs_concat['SSS'] = index_slopes_TBDEM_gdfs_concat['SSS'].astype(str)
        index_slopes_TBDEM_gdfs_concat['V'] = ['0']*len(index_slopes_TBDEM_gdfs_concat)
        index_slopes_TBDEM_gdfs_concat['transect_id'] = index_slopes_TBDEM_gdfs_concat['transect_id'].astype(str)
        index_slopes_TBDEM_gdfs_concat['LLLLLL'] = [l[-6:] for l in index_slopes_TBDEM_gdfs_concat['transect_id']]
        index_slopes_TBDEM_gdfs_concat['LLLLLL'] = (index_slopes_TBDEM_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        index_slopes_TBDEM_gdfs_concat['max_slope'] = index_slopes_TBDEM_gdfs_concat['max_slope'].astype(float)
        index_slopes_TBDEM_gdfs_concat['median_slope'] = index_slopes_TBDEM_gdfs_concat['median_slope'].astype(float)
        index_slopes_TBDEM_gdfs_concat['avg_slope_cleaned'] = index_slopes_TBDEM_gdfs_concat['avg_slope_cleaned'].astype(float)
        index_slopes_TBDEM_gdfs_concat['avg_slope'] = index_slopes_TBDEM_gdfs_concat['avg_slope'].astype(float)
        columns_to_keep = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL',
                        'transect_id', 
                        'max_slope', 'median_slope', 'avg_slope_cleaned', 'avg_slope', 'geometry']
        for col in index_slopes_TBDEM_gdfs_concat.columns:
            if col not in columns_to_keep:
                try:
                    index_slopes_TBDEM_gdfs_concat = index_slopes_TBDEM_gdfs_concat.drop(columns=[col])
                except:
                    pass
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'max_slope', 'median_slope', 'avg_slope', 
                        'avg_slope_cleaned',
                        'geometry']
        index_slopes_TBDEM_gdfs_concat = index_slopes_TBDEM_gdfs_concat[column_order]

        index_dictionary_gdfs = {'00_transects':index_transect_gdfs_concat,
                                '01_rois':index_roi_gdfs_concat,
                                '02_reference_shorelines':index_reference_shoreline_gdfs_concat,
                                '03_reference_polygons':index_reference_polygon_gdfs_concat,
                                '04_crest1s':index_crest1_gdfs_concat,
                                '05_crest2s':index_crest2_gdfs_concat,
                                '06_crest3s':index_crest3_gdfs_concat,
                                '07_inflection_points':index_inflection_point_gdfs_concat,
                                '08_toes':index_toe_gdfs_concat,
                                '09_ArcticDEM_slopes':index_slopes_ArcticDEM_gdfs_concat,
                                '10_AlaskaDSM_slopes':index_slopes_AlaskaDSM_gdfs_concat,
                                '11_tbdem_slopes':index_slopes_TBDEM_gdfs_concat
                                }
        index_geopackage = os.path.join(home_geopackage, g+c+'_index.gpkg')

        print('writing index geopackage')
        ##writing index geopackage
        gdf_dictionary_to_geopackage(index_dictionary_gdfs, index_geopackage)
        os.system('python /mnt/f/qc_code/fixing_index.py')
        
    elif 'tier_0' in tiers:
        print('getting tier 0 files')
        ##getting tier_0 files
        tier_0_zoo_rgb_shorelines_gdfs = [None]*num_sections
        tier_0_zoo_rgb_shorelines_filter_gdfs = [None]*num_sections
        tier_0_zoo_rgb_time_series_stacked_dfs = [None]*num_sections
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs = [None]*num_sections
        tier_0_nir_thresh_shorelines_gdfs = [None]*num_sections
        tier_0_nir_thresh_shorelines_filter_gdfs = [None]*num_sections
        tier_0_nir_thresh_time_series_stacked_dfs = [None]*num_sections
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs = [None]*num_sections
        tier_0_swir_thresh_shorelines_gdfs = [None]*num_sections
        tier_0_swir_thresh_shorelines_filter_gdfs = [None]*num_sections
        tier_0_swir_thresh_time_series_stacked_dfs = [None]*num_sections
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs = [None]*num_sections
        i=0
        for rr in tqdm(rrs):
            r_home = os.path.join(home_data_service, 'Tier0', 'G'+g, 'C'+c, 'RR'+rr)
            sections = sorted(get_immediate_subdirectories(r_home))
            for section in tqdm(sections):
                section_dir = os.path.join(r_home, section)
                sss = section[3:] 
                section_string = g+c+rr+sss

                zoo_rgb_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
                fix_cols_gdf(zoo_rgb_shorelines_path, g, c, rr, sss)
                tier_0_zoo_rgb_shorelines_gdfs[i] = zoo_rgb_shorelines_path

                zoo_rgb_shorelines_filter_path = os.path.join(section_dir, section_string + '_extracted_shorelines_filter.geojson')
                fix_cols_gdf(zoo_rgb_shorelines_filter_path, g, c, rr, sss)
                tier_0_zoo_rgb_shorelines_filter_gdfs[i] = zoo_rgb_shorelines_filter_path

                zoo_time_series_tidally_corrected_path = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged.csv')
                fix_cols_df(zoo_time_series_tidally_corrected_path, g, c, rr, sss)
                zoo_time_series_tidally_corrected_path = df_to_gdf(zoo_time_series_tidally_corrected_path,'intersectxy')
                tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs[i] = zoo_time_series_tidally_corrected_path

                nir_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
                fix_cols_gdf(nir_shorelines_path, g, c, rr, sss)
                tier_0_nir_thresh_shorelines_gdfs[i] = nir_shorelines_path

                nir_shorelines_filter_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh_filter.geojson') 
                fix_cols_gdf(nir_shorelines_filter_path, g, c, rr, sss)
                tier_0_nir_thresh_shorelines_filter_gdfs[i] = nir_shorelines_filter_path

                nir_time_series_tidally_corrected_path = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_nir_thresh.csv')
                fix_cols_df(nir_time_series_tidally_corrected_path, g, c, rr, sss)
                nir_time_series_tidally_corrected_path = df_to_gdf(nir_time_series_tidally_corrected_path,'intersectxy')
                tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs[i] = nir_time_series_tidally_corrected_path 

                swir_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')
                fix_cols_gdf(swir_shorelines_path, g, c, rr, sss)
                tier_0_swir_thresh_shorelines_gdfs[i] = swir_shorelines_path

                swir_shorelines_filter_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh_filter.geojson')
                fix_cols_gdf(swir_shorelines_filter_path, g, c, rr, sss)
                tier_0_swir_thresh_shorelines_filter_gdfs[i] = swir_shorelines_filter_path

                swir_time_series_tidally_corrected_path = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_swir_thresh.csv')
                fix_cols_df(swir_time_series_tidally_corrected_path, g, c, rr, sss)
                swir_time_series_tidally_corrected_path = df_to_gdf(swir_time_series_tidally_corrected_path, 'intersectxy')
                tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs[i] = swir_time_series_tidally_corrected_path
                i=i+1

        print('concatenating tier 0 files')
        ##tier_0 files
        print('rgb waterlines')
        tier_0_zoo_rgb_shorelines_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_zoo_rgb_shorelines_gdfs) if os.path.isfile(f)==True])
        print('rgb waterlines filter')
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_zoo_rgb_shorelines_filter_gdfs) if os.path.isfile(f)==True])
        # print('rgb time series raw')
        # tier_0_zoo_rgb_time_series_stacked_dfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_zoo_rgb_time_series_stacked_dfs) if os.path.isfile(f)==True])
        print('rgb time series tidally corrected')
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs) if os.path.isfile(f)==True])
        print('nir waterlines')
        tier_0_nir_thresh_shorelines_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_nir_thresh_shorelines_gdfs) if os.path.isfile(f)==True])
        print('nir waterlines filtered')
        tier_0_nir_thresh_shorelines_filter_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_nir_thresh_shorelines_filter_gdfs) if os.path.isfile(f)==True])
        # print('nir time series raw')
        #tier_0_nir_thresh_time_series_stacked_dfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_nir_thresh_time_series_stacked_dfs) if os.path.isfile(f)==True])
        print('nir time series tidally corrected')
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs) if os.path.isfile(f)==True])
        print('swir waterlines')
        tier_0_swir_thresh_shorelines_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_swir_thresh_shorelines_gdfs) if os.path.isfile(f)==True])
        print('swir waterlines filter')
        tier_0_swir_thresh_shorelines_filter_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_swir_thresh_shorelines_filter_gdfs) if os.path.isfile(f)==True])
        # print('swir time series raw')
        # tier_0_swir_thresh_time_series_stacked_dfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_swir_thresh_time_series_stacked_dfs) if os.path.isfile(f)==True])
        print('swir time series tidally corrected')
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs) if os.path.isfile(f)==True])

        ######zoo rgb
        dt = pd.to_datetime(tier_0_zoo_rgb_shorelines_gdfs_concat['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_zoo_rgb_shorelines_gdfs_concat['dates'] = formatted
        tier_0_zoo_rgb_shorelines_gdfs_concat['dates'] = tier_0_zoo_rgb_shorelines_gdfs_concat['dates'].astype(str)
        tier_0_zoo_rgb_shorelines_gdfs_concat['image_suitability_score'] = tier_0_zoo_rgb_shorelines_gdfs_concat['image_suitability_score'].astype(float)
        tier_0_zoo_rgb_shorelines_gdfs_concat['segmentation_suitability_score'] = tier_0_zoo_rgb_shorelines_gdfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_zoo_rgb_shorelines_gdfs_concat['satname'] = tier_0_zoo_rgb_shorelines_gdfs_concat['satname'].astype(str)
        tier_0_zoo_rgb_shorelines_gdfs_concat['simplify_param'] =  tier_0_zoo_rgb_shorelines_gdfs_concat['simplify_param'].astype(float)
        tier_0_zoo_rgb_shorelines_gdfs_concat['kde_value'] =  tier_0_zoo_rgb_shorelines_gdfs_concat['kde_value'].astype(float)
        tier_0_zoo_rgb_shorelines_gdfs_concat['G'] = tier_0_zoo_rgb_shorelines_gdfs_concat['G'].astype(str)
        tier_0_zoo_rgb_shorelines_gdfs_concat['C'] = tier_0_zoo_rgb_shorelines_gdfs_concat['C'].astype(str)
        tier_0_zoo_rgb_shorelines_gdfs_concat['RR'] = tier_0_zoo_rgb_shorelines_gdfs_concat['RR'].astype(str)
        tier_0_zoo_rgb_shorelines_gdfs_concat['SSS'] = tier_0_zoo_rgb_shorelines_gdfs_concat['SSS'].astype(str)
        tier_0_zoo_rgb_shorelines_gdfs_concat['year'] = pd.to_datetime(tier_0_zoo_rgb_shorelines_gdfs_concat['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
        column_order = ['G', 'C', 'RR', 'SSS', 
                        'dates', 'satname', 'year',
                        'image_suitability_score', 'segmentation_suitability_score', 
                        'simplify_param', 'kde_value', 
                        'geometry']
        tier_0_zoo_rgb_shorelines_gdfs_concat = tier_0_zoo_rgb_shorelines_gdfs_concat[column_order]
        
        dt = pd.to_datetime(tier_0_zoo_rgb_shorelines_filter_gdfs_concat['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['dates'] = formatted
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['dates'] = tier_0_zoo_rgb_shorelines_filter_gdfs_concat['dates'].astype(str)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['image_suitability_score'] = tier_0_zoo_rgb_shorelines_filter_gdfs_concat['image_suitability_score'].astype(float)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['segmentation_suitability_score'] = tier_0_zoo_rgb_shorelines_filter_gdfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['satname'] = tier_0_zoo_rgb_shorelines_filter_gdfs_concat['satname'].astype(str)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['simplify_param'] =  tier_0_zoo_rgb_shorelines_filter_gdfs_concat['simplify_param'].astype(float)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['kde_value'] =  tier_0_zoo_rgb_shorelines_filter_gdfs_concat['kde_value'].astype(float)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['G'] = tier_0_zoo_rgb_shorelines_filter_gdfs_concat['G'].astype(str)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['C'] = tier_0_zoo_rgb_shorelines_filter_gdfs_concat['C'].astype(str)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['RR'] = tier_0_zoo_rgb_shorelines_filter_gdfs_concat['RR'].astype(str)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['SSS'] = tier_0_zoo_rgb_shorelines_filter_gdfs_concat['SSS'].astype(str)
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat['year'] = pd.to_datetime(tier_0_zoo_rgb_shorelines_filter_gdfs_concat['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
        column_order = ['G', 'C', 'RR', 'SSS', 
                        'dates', 'satname', 'year',
                        'image_suitability_score', 'segmentation_suitability_score', 
                        'simplify_param', 'kde_value', 
                        'geometry']
        tier_0_zoo_rgb_shorelines_filter_gdfs_concat = tier_0_zoo_rgb_shorelines_filter_gdfs_concat[column_order]

        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['year'] = pd.to_datetime(tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['dates'], utc=True).dt.year
        dt = pd.to_datetime(tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['dates'], utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['dates'] = formatted
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['dates'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['dates'].astype(str)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['image_suitability_score'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['image_suitability_score'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['segmentation_suitability_score'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['kde_value'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['kde_value'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['transect_id'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['transect_id'].astype(str)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['intersect_x'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['intersect_x'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['intersect_y'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['intersect_y'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['cross_distance'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['cross_distance'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['tide'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['tide'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['x'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['x'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['y'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['y'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['avg_slope_cleaned'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['avg_slope_cleaned'].astype(float)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['G'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['G'].astype(str)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['C'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['C'].astype(str)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['RR'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['RR'].astype(str)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['SSS'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['SSS'].astype(str)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['V'] = ['0']*len(tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['satname'] = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['satname'].astype(str)
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'] = [l[-6:] for l in tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['transect_id']]
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'] = (tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        keep_cols = ['dates', 'image_suitability_score', 'segmentation_suitability_score',
                    'kde_value', 'satname', 'transect_id', 'intersect_x', 'intersect_y','cross_distance_tidally_corrected', 
                    'cross_distance', 'G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'tide', 'x', 'y', 'year', 'avg_slope_cleaned', 'geometry']
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat[[c for c in keep_cols if c in tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat.columns]]
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                        'dates', 'satname', 'year',
                        'cross_distance', 'cross_distance_tidally_corrected',
                        'image_suitability_score', 'segmentation_suitability_score',
                        'kde_value', 'intersect_x', 'intersect_y', 'tide', 'x', 'y', 'avg_slope_cleaned',
                        'geometry']
        tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat = tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat[column_order]

        ###nir
        dt = pd.to_datetime(tier_0_nir_thresh_shorelines_gdfs_concat['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_nir_thresh_shorelines_gdfs_concat['dates'] = formatted
        tier_0_nir_thresh_shorelines_gdfs_concat['dates'] = tier_0_nir_thresh_shorelines_gdfs_concat['dates'].astype(str)
        tier_0_nir_thresh_shorelines_gdfs_concat['image_suitability_score'] = tier_0_nir_thresh_shorelines_gdfs_concat['image_suitability_score'].astype(float)
        tier_0_nir_thresh_shorelines_gdfs_concat['segmentation_suitability_score'] = tier_0_nir_thresh_shorelines_gdfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_nir_thresh_shorelines_gdfs_concat['satname'] = tier_0_nir_thresh_shorelines_gdfs_concat['satname'].astype(str)
        tier_0_nir_thresh_shorelines_gdfs_concat['simplify_param'] =  tier_0_nir_thresh_shorelines_gdfs_concat['simplify_param'].astype(float)
        tier_0_nir_thresh_shorelines_gdfs_concat['kde_value'] =  tier_0_nir_thresh_shorelines_gdfs_concat['kde_value'].astype(float)
        tier_0_nir_thresh_shorelines_gdfs_concat['G'] = tier_0_nir_thresh_shorelines_gdfs_concat['G'].astype(str)
        tier_0_nir_thresh_shorelines_gdfs_concat['C'] = tier_0_nir_thresh_shorelines_gdfs_concat['C'].astype(str)
        tier_0_nir_thresh_shorelines_gdfs_concat['RR'] = tier_0_nir_thresh_shorelines_gdfs_concat['RR'].astype(str)
        tier_0_nir_thresh_shorelines_gdfs_concat['SSS'] = tier_0_nir_thresh_shorelines_gdfs_concat['SSS'].astype(str)
        tier_0_nir_thresh_shorelines_gdfs_concat['year'] =pd.to_datetime(tier_0_nir_thresh_shorelines_gdfs_concat['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
        column_order = ['G', 'C', 'RR', 'SSS', 
                        'dates', 'satname', 'year',
                        'image_suitability_score', 'segmentation_suitability_score', 
                        'simplify_param', 'kde_value', 
                        'geometry']
        tier_0_nir_thresh_shorelines_gdfs_concat = tier_0_nir_thresh_shorelines_gdfs_concat[column_order]

        dt = pd.to_datetime(tier_0_nir_thresh_shorelines_filter_gdfs_concat['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['dates'] = formatted
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['dates'] = tier_0_nir_thresh_shorelines_filter_gdfs_concat['dates'].astype(str)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['image_suitability_score'] = tier_0_nir_thresh_shorelines_filter_gdfs_concat['image_suitability_score'].astype(float)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['segmentation_suitability_score'] = tier_0_nir_thresh_shorelines_filter_gdfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['satname'] = tier_0_nir_thresh_shorelines_filter_gdfs_concat['satname'].astype(str)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['simplify_param'] =  tier_0_nir_thresh_shorelines_filter_gdfs_concat['simplify_param'].astype(float)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['kde_value'] =  tier_0_nir_thresh_shorelines_filter_gdfs_concat['kde_value'].astype(float)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['G'] = tier_0_nir_thresh_shorelines_filter_gdfs_concat['G'].astype(str)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['C'] = tier_0_nir_thresh_shorelines_filter_gdfs_concat['C'].astype(str)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['RR'] = tier_0_nir_thresh_shorelines_filter_gdfs_concat['RR'].astype(str)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['SSS'] = tier_0_nir_thresh_shorelines_filter_gdfs_concat['SSS'].astype(str)
        tier_0_nir_thresh_shorelines_filter_gdfs_concat['year'] = pd.to_datetime(tier_0_nir_thresh_shorelines_filter_gdfs_concat['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
        column_order = ['G', 'C', 'RR', 'SSS', 
                        'dates', 'satname', 'year',
                        'image_suitability_score', 'segmentation_suitability_score', 
                        'simplify_param', 'kde_value', 
                        'geometry']
        tier_0_nir_thresh_shorelines_filter_gdfs_concat = tier_0_nir_thresh_shorelines_filter_gdfs_concat[column_order]      

        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['year'] = pd.to_datetime(tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'], utc=True).dt.year
        dt = pd.to_datetime(tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'], utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'] = formatted
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'].astype(str)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['image_suitability_score'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['image_suitability_score'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['segmentation_suitability_score'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['kde_value'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['kde_value'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['transect_id'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['transect_id'].astype(str)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['intersect_x'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['intersect_x'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['intersect_y'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['intersect_y'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['cross_distance'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['cross_distance'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['tide'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['tide'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['x'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['x'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['y'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['y'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['avg_slope_cleaned'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['avg_slope_cleaned'].astype(float)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['G'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['G'].astype(str)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['C'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['C'].astype(str)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['RR'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['RR'].astype(str)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['SSS'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['SSS'].astype(str)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['satname'] = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['satname'].astype(str)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['V'] = ['0']*len(tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat)
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'] = [l[-6:] for l in tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['transect_id']]
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'] = (tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        keep_cols = ['dates', 'image_suitability_score', 'segmentation_suitability_score',
                    'kde_value', 'satname', 'transect_id', 'intersect_x', 'intersect_y', 'year','cross_distance_tidally_corrected', 
                    'cross_distance', 'G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'tide', 'x', 'y', 'avg_slope_cleaned', 'geometry']
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat[[c for c in keep_cols if c in tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat.columns]]
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                        'dates', 'satname', 'year',
                        'cross_distance', 'cross_distance_tidally_corrected',
                        'image_suitability_score', 'segmentation_suitability_score',
                        'kde_value', 'intersect_x', 'intersect_y', 'tide', 'x', 'y', 'avg_slope_cleaned',
                        'geometry']
        tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat = tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat[column_order]


        ####swir
        dt = pd.to_datetime(tier_0_swir_thresh_shorelines_gdfs_concat['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_swir_thresh_shorelines_gdfs_concat['dates'] = formatted
        tier_0_swir_thresh_shorelines_gdfs_concat['dates'] = tier_0_swir_thresh_shorelines_gdfs_concat['dates'].astype(str)
        tier_0_swir_thresh_shorelines_gdfs_concat['image_suitability_score'] = tier_0_swir_thresh_shorelines_gdfs_concat['image_suitability_score'].astype(float)
        tier_0_swir_thresh_shorelines_gdfs_concat['segmentation_suitability_score'] = tier_0_swir_thresh_shorelines_gdfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_swir_thresh_shorelines_gdfs_concat['satname'] = tier_0_swir_thresh_shorelines_gdfs_concat['satname'].astype(str)
        tier_0_swir_thresh_shorelines_gdfs_concat['simplify_param'] =  tier_0_swir_thresh_shorelines_gdfs_concat['simplify_param'].astype(float)
        tier_0_swir_thresh_shorelines_gdfs_concat['kde_value'] =  tier_0_swir_thresh_shorelines_gdfs_concat['kde_value'].astype(float)
        tier_0_swir_thresh_shorelines_gdfs_concat['G'] = tier_0_swir_thresh_shorelines_gdfs_concat['G'].astype(str)
        tier_0_swir_thresh_shorelines_gdfs_concat['C'] = tier_0_swir_thresh_shorelines_gdfs_concat['C'].astype(str)
        tier_0_swir_thresh_shorelines_gdfs_concat['RR'] = tier_0_swir_thresh_shorelines_gdfs_concat['RR'].astype(str)
        tier_0_swir_thresh_shorelines_gdfs_concat['SSS'] = tier_0_swir_thresh_shorelines_gdfs_concat['SSS'].astype(str)
        tier_0_swir_thresh_shorelines_gdfs_concat['year'] =pd.to_datetime(tier_0_swir_thresh_shorelines_gdfs_concat['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
        column_order = ['G', 'C', 'RR', 'SSS', 
                        'dates', 'satname', 'year',
                        'image_suitability_score', 'segmentation_suitability_score', 
                        'simplify_param', 'kde_value', 
                        'geometry']
        tier_0_swir_thresh_shorelines_gdfs_concat = tier_0_swir_thresh_shorelines_gdfs_concat[column_order]

        dt = pd.to_datetime(tier_0_swir_thresh_shorelines_filter_gdfs_concat['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['dates'] = formatted
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['dates'] = tier_0_swir_thresh_shorelines_filter_gdfs_concat['dates'].astype(str)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['image_suitability_score'] = tier_0_swir_thresh_shorelines_filter_gdfs_concat['image_suitability_score'].astype(float)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['segmentation_suitability_score'] = tier_0_swir_thresh_shorelines_filter_gdfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['satname'] = tier_0_swir_thresh_shorelines_filter_gdfs_concat['satname'].astype(str)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['simplify_param'] =  tier_0_swir_thresh_shorelines_filter_gdfs_concat['simplify_param'].astype(float)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['kde_value'] =  tier_0_swir_thresh_shorelines_filter_gdfs_concat['kde_value'].astype(float)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['G'] = tier_0_swir_thresh_shorelines_filter_gdfs_concat['G'].astype(str)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['C'] = tier_0_swir_thresh_shorelines_filter_gdfs_concat['C'].astype(str)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['RR'] = tier_0_swir_thresh_shorelines_filter_gdfs_concat['RR'].astype(str)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['SSS'] = tier_0_swir_thresh_shorelines_filter_gdfs_concat['SSS'].astype(str)
        tier_0_swir_thresh_shorelines_filter_gdfs_concat['year'] = pd.to_datetime(tier_0_swir_thresh_shorelines_filter_gdfs_concat['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
        column_order = ['G', 'C', 'RR', 'SSS', 
                        'dates', 'satname', 'year',
                        'image_suitability_score', 'segmentation_suitability_score', 
                        'simplify_param', 'kde_value', 
                        'geometry']
        tier_0_swir_thresh_shorelines_filter_gdfs_concat = tier_0_swir_thresh_shorelines_filter_gdfs_concat[column_order]

        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['year'] = pd.to_datetime(tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'], utc=True).dt.year
        dt = pd.to_datetime(tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'], utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'] = formatted
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['dates'].astype(str)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['image_suitability_score'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['image_suitability_score'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['segmentation_suitability_score'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['segmentation_suitability_score'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['kde_value'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['kde_value'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['transect_id'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['transect_id'].astype(str)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['intersect_x'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['intersect_x'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['intersect_y'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['intersect_y'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['cross_distance'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['cross_distance'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['tide'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['tide'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['x'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['x'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['y'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['y'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['avg_slope_cleaned'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['avg_slope_cleaned'].astype(float)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['G'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['G'].astype(str)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['C'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['C'].astype(str)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['RR'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['RR'].astype(str)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['SSS'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['SSS'].astype(str)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['satname'] = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['satname'].astype(str)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['V'] = ['0']*len(tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat)
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'] = [l[-6:] for l in tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['transect_id']]
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'] = (tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        keep_cols = ['dates', 'image_suitability_score', 'segmentation_suitability_score',
                    'kde_value', 'satname', 'transect_id', 'intersect_x', 'intersect_y',
                    'cross_distance','cross_distance_tidally_corrected', 'G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'tide', 'x', 'y', 'avg_slope_cleaned', 'geometry']
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat[[c for c in keep_cols if c in tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat.columns]]
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                        'dates', 'satname', 
                        'cross_distance', 'cross_distance_tidally_corrected',
                        'image_suitability_score', 'segmentation_suitability_score',
                        'kde_value', 'intersect_x', 'intersect_y', 'tide', 'x', 'y', 'avg_slope_cleaned',
                        'geometry']
        tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat = tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat[column_order]

        ##tier_0 files
        tier_0_dictionary_gdfs = {'00_zoo_rgb_waterlines':tier_0_zoo_rgb_shorelines_gdfs_concat,
                                '01_zoo_rgb_waterlines_filter':tier_0_zoo_rgb_shorelines_filter_gdfs_concat,
                                '02_zoo_rgb_time_series':tier_0_zoo_rgb_tidally_corrected_time_series_stacked_dfs_concat,
                                '03_nir_waterlines':tier_0_nir_thresh_shorelines_gdfs_concat,
                                '04_nir_waterlines_filter':tier_0_nir_thresh_shorelines_filter_gdfs_concat,
                                '05_nir_time_series':tier_0_nir_thresh_tidally_corrected_time_series_stacked_dfs_concat,
                                '06_swir_waterlines':tier_0_swir_thresh_shorelines_gdfs_concat,
                                '07_swir_waterlines_filter':tier_0_swir_thresh_shorelines_filter_gdfs_concat,
                                '08_swir_time_series':tier_0_swir_thresh_tidally_corrected_time_series_stacked_dfs_concat
                                }
        tier_0_geopackage = os.path.join(home_geopackage, g+c+'_tier_0.gpkg')
        print('writing tier 0 geopackage')
        ##writing tier 0 geopackage
        gdf_dictionary_to_geopackage(tier_0_dictionary_gdfs, tier_0_geopackage)

    elif 'tier_1' in tiers:
        print('getting tier 1 files')
        ##getting tier 1 files
        tier_1_unfiltered_tidally_corrected_points_gdfs = [None]*num_sections
        tier_1_filtered_tidally_corrected_time_series_stacked_dfs = [None]*num_sections
        tier_1_filtered_tidally_corrected_points_gdfs = [None]*num_sections
        tier_1_spatial_kde_tifs = [None]*num_sections
        tier_1_spatial_kde_otsu_tifs = [None]*num_sections
        tier_1_spatial_kde_otsu_gdfs = [None]*num_sections
        i=0
        for rr in tqdm(rrs):
            r_home = os.path.join(home_data_service, 'Tier1', 'G'+g, 'C'+c, 'RR'+rr)
            sections = sorted(get_immediate_subdirectories(r_home))
            for section in tqdm(sections):
                section_dir = os.path.join(r_home, section)
                sss = section[3:] 
                section_string = g+c+rr+sss

                unfiltered_points_path = os.path.join(section_dir, section_string + '_unfiltered_tidally_corrected_points.geojson')
                fix_cols_gdf(unfiltered_points_path, g, c, rr, sss)
                tier_1_unfiltered_tidally_corrected_points_gdfs[i] = unfiltered_points_path

                filtered_points_path = os.path.join(section_dir, section_string + '_filtered_tidally_corrected_points.geojson')
                fix_cols_gdf(filtered_points_path, g, c, rr, sss)
                tier_1_filtered_tidally_corrected_points_gdfs[i] = filtered_points_path

                spatial_kde_path = os.path.join(section_dir, section_string + '_spatial_kde.tif')
                tier_1_spatial_kde_tifs[i] = spatial_kde_path

                spatial_kde_otsu_path = os.path.join(section_dir, section_string + '_spatial_kde_otsu.tif')
                tier_1_spatial_kde_otsu_tifs[i] = spatial_kde_otsu_path

                spatial_kde_otsu_vec_path = os.path.join(section_dir, section_string + '_spatial_kde_otsu.geojson')
                fix_cols_gdf(spatial_kde_otsu_vec_path, g, c, rr, sss)
                tier_1_spatial_kde_otsu_gdfs[i] = spatial_kde_otsu_vec_path
                i=i+1

        print('concatenating tier 1 files')
        #tier 1 files
        print('unfiltered shoreline points')
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_1_unfiltered_tidally_corrected_points_gdfs) if os.path.isfile(f)==True])
        print('shoreline points')
        tier_1_filtered_tidally_corrected_points_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_1_filtered_tidally_corrected_points_gdfs) if os.path.isfile(f)==True])
        print('spatial kde')
        tier_1_spatial_kde_tifs_concat = tier_1_spatial_kde_tifs
        print('spatial kde otsu')
        tier_1_spatial_kde_otsu_tifs_concat = tier_1_spatial_kde_otsu_tifs

        dt = pd.to_datetime(tier_1_unfiltered_tidally_corrected_points_gdfs_concat['dates'], utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['dates'] = formatted
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['cross_distance'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['cross_distance'].astype(float)    
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['cross_distance_rgb'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['cross_distance_rgb'].astype(float)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['cross_distance_nir'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['cross_distance_nir'].astype(float)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['cross_distance_swir'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['cross_distance_swir'].astype(float)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['transect_id'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['transect_id'].astype(str)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['avg_suitability'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['avg_suitability'].astype(float)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['satname'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['satname'].astype(str)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['avg_slope'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['avg_slope'].astype(float)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['tide'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['tide'].astype(float)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['ci'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['ci'].astype(float)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['year'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['year'].astype(int)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['G'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['G'].astype(str)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['C'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['C'].astype(str)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['RR'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['RR'].astype(str)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['SSS'] = tier_1_unfiltered_tidally_corrected_points_gdfs_concat['SSS'].astype(str)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['V'] = ['0']*len(tier_1_unfiltered_tidally_corrected_points_gdfs_concat)
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_1_unfiltered_tidally_corrected_points_gdfs_concat['transect_id']]
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat['LLLLLL'] = (tier_1_unfiltered_tidally_corrected_points_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                        'dates', 'satname',
                        'cross_distance', 'cross_distance_rgb', 'cross_distance_nir', 'cross_distance_swir', 'ci',
                        'avg_suitability', 
                        'avg_slope','tide',
                        'year',
                        'geometry']
        tier_1_unfiltered_tidally_corrected_points_gdfs_concat = tier_1_unfiltered_tidally_corrected_points_gdfs_concat[column_order]
        #tier 1 files
        tier_1_dictionary_gdfs = {'00_unfiltered_shoreline_points':tier_1_unfiltered_tidally_corrected_points_gdfs_concat,
                                }
        print('writing tier 1 geopackage')
        ##writing tier 1 geopackage
        tier_1_geopackage = os.path.join(home_geopackage, g+c+'_tier_1.gpkg')
        gdf_dictionary_to_geopackage(tier_1_dictionary_gdfs, tier_1_geopackage)

        dt = pd.to_datetime(tier_1_filtered_tidally_corrected_points_gdfs_concat['dates'], utc=True)
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        tier_1_filtered_tidally_corrected_points_gdfs_concat['dates'] = formatted
        tier_1_filtered_tidally_corrected_points_gdfs_concat['cross_distance'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['cross_distance'].astype(float)    
        tier_1_filtered_tidally_corrected_points_gdfs_concat['cross_distance_rgb'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['cross_distance_rgb'].astype(float)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['cross_distance_nir'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['cross_distance_nir'].astype(float)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['cross_distance_swir'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['cross_distance_swir'].astype(float)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['transect_id'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['transect_id'].astype(str)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['avg_suitability'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['avg_suitability'].astype(float)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['satname'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['satname'].astype(str)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['avg_slope'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['avg_slope'].astype(float)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['tide'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['tide'].astype(float)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['ci'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['ci'].astype(float)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['year'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['year'].astype(int)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['G'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['G'].astype(str)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['C'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['C'].astype(str)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['RR'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['RR'].astype(str)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['SSS'] = tier_1_filtered_tidally_corrected_points_gdfs_concat['SSS'].astype(str)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['V'] = ['0']*len(tier_1_filtered_tidally_corrected_points_gdfs_concat)
        tier_1_filtered_tidally_corrected_points_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_1_filtered_tidally_corrected_points_gdfs_concat['transect_id']]
        tier_1_filtered_tidally_corrected_points_gdfs_concat['LLLLLL'] = (tier_1_filtered_tidally_corrected_points_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                        'dates', 'satname',
                        'cross_distance', 'cross_distance_rgb', 'cross_distance_nir', 'cross_distance_swir', 'ci',
                        'avg_suitability', 
                        'avg_slope','tide',
                        'year',
                        'geometry']
        tier_1_filtered_tidally_corrected_points_gdfs_concat = tier_1_filtered_tidally_corrected_points_gdfs_concat[column_order]
        #tier 1 files
        tier_1_dictionary_gdfs = {
                                '01_shoreline_points':tier_1_filtered_tidally_corrected_points_gdfs_concat,
                                }
        print('writing tier 1 geopackage')
        ##writing tier 1 geopackage
        tier_1_geopackage = os.path.join(home_geopackage, g+c+'_tier_1.gpkg')
        gdf_dictionary_to_geopackage(tier_1_dictionary_gdfs, tier_1_geopackage)
        
        shoreline_lines = points_to_lines_no_gaps(tier_1_filtered_tidally_corrected_points_gdfs_concat, kind="instantaneous")
        shoreline_lines['G'] = shoreline_lines['G'].astype(str)
        shoreline_lines['C'] = shoreline_lines['C'].astype(str)
        shoreline_lines['RR'] = shoreline_lines['RR'].astype(str)
        shoreline_lines['SSS'] = shoreline_lines['SSS'].astype(str)
        dt = pd.to_datetime(shoreline_lines['dates'], format='%Y-%m-%d-%H-%M-%S')
        shoreline_lines['year'] = dt.dt.year
        formatted = dt.dt.strftime('%Y-%m-%d-%H-%M-%S')
        shoreline_lines['dates'] = formatted
        shoreline_lines['dates'] = shoreline_lines['dates'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS',
                        'dates','year',
                        'geometry']
        shoreline_lines = shoreline_lines[column_order]

        #tier 1 files
        tier_1_dictionary_gdfs = {
                                '02_shoreline_lines':shoreline_lines
                                }
        print('writing tier 1 geopackage')
        ##writing tier 1 geopackage
        tier_1_geopackage = os.path.join(home_geopackage, g+c+'_tier_1.gpkg')
        gdf_dictionary_to_geopackage(tier_1_dictionary_gdfs, tier_1_geopackage)

        tier_1_dictionary_tifs = {'spatial_kde_tifs':tier_1_spatial_kde_tifs_concat,
                                'spatial_kde_otsu_tifs':tier_1_spatial_kde_otsu_tifs_concat
                                }
        
        spatial_kde_geopackage = os.path.join(home_geopackage, g+c+'_spatial_kde.gpkg')

        #tif_dictionary_to_geopackage(tier_1_dictionary_tifs, spatial_kde_geopackage)

    elif 'tier_2' in tiers:
        print('getting tier 2 files')
        ##getting tier 2 files
        tier_2_resampled_reprojected_time_series_stacked_dfs = [None]*num_sections
        tier_2_resampled_reprojected_shorelines_gdfs = [None]*num_sections
        tier_2_trends_gdfs = [None]*num_sections
        tier_2_spatial_kde_otsu_gdfs = [None]*num_sections
        i=0
        for rr in tqdm(rrs):
            r_home = os.path.join(home_data_service, 'Tier2', 'G'+g, 'C'+c, 'RR'+rr)
            r_home_spatial_kde = os.path.join(home_data_service, 'Tier1', 'G'+g, 'C'+c, 'RR'+rr)
            sections = sorted(get_immediate_subdirectories(r_home))
            for section in tqdm(sections):
                section_dir = os.path.join(r_home, section)
                section_dir_kde = os.path.join(r_home_spatial_kde, section)
                sss = section[3:] 
                section_string = g+c+rr+sss

                resampled_time_series_path = os.path.join(section_dir, section_string + '_resampled_tidally_corrected_transect_time_series_merged.csv')
                fix_cols_df(resampled_time_series_path, g, c, rr, sss)
                tier_2_resampled_reprojected_time_series_stacked_dfs[i] = resampled_time_series_path

                resampled_points_path = os.path.join(section_dir, section_string + '_reprojected_points.geojson')
                fix_cols_gdf(resampled_points_path, g, c, rr, sss)
                tier_2_resampled_reprojected_shorelines_gdfs[i] = resampled_points_path

                trends_path = os.path.join(section_dir, section_string + '_transects_trends.geojson')
                fix_cols_gdf(trends_path, g, c, rr, sss)
                tier_2_trends_gdfs[i] = trends_path

                spatial_kde_otsu_vec_path = os.path.join(section_dir_kde, section_string + '_spatial_kde_otsu.geojson')
                fix_cols_gdf(spatial_kde_otsu_vec_path, g, c, rr, sss)
                tier_2_spatial_kde_otsu_gdfs[i] = spatial_kde_otsu_vec_path
                i=i+1
        print('concatenating tier 2 files')
        #tier 2 files
        print('annual shorelines points')
        tier_2_resampled_reprojected_shorelines_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_2_resampled_reprojected_shorelines_gdfs) if os.path.isfile(f)==True])
        print('trends')
        tier_2_trends_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_2_trends_gdfs) if os.path.isfile(f)==True])
        print('shoreline change envelopes')
        tier_2_spatial_kde_otsu_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_2_spatial_kde_otsu_gdfs) if os.path.isfile(f)==True])


        ### annual_shoreline_points
        tier_2_resampled_reprojected_shorelines_gdfs_concat['cross_distance'] = tier_2_resampled_reprojected_shorelines_gdfs_concat['cross_distance'].astype(float)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['transect_id'] = tier_2_resampled_reprojected_shorelines_gdfs_concat['transect_id'].astype(str)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['ci'] = tier_2_resampled_reprojected_shorelines_gdfs_concat['ci'].astype(float)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['year'] = tier_2_resampled_reprojected_shorelines_gdfs_concat['year'].astype(int)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['G'] = tier_2_resampled_reprojected_shorelines_gdfs_concat['G'].astype(str)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['C'] = tier_2_resampled_reprojected_shorelines_gdfs_concat['C'].astype(str)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['RR'] = tier_2_resampled_reprojected_shorelines_gdfs_concat['RR'].astype(str)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['SSS'] = tier_2_resampled_reprojected_shorelines_gdfs_concat['SSS'].astype(str)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['V'] = ['0']*len(tier_2_resampled_reprojected_shorelines_gdfs_concat)
        tier_2_resampled_reprojected_shorelines_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_2_resampled_reprojected_shorelines_gdfs_concat['transect_id']]
        tier_2_resampled_reprojected_shorelines_gdfs_concat['LLLLLL'] = (tier_2_resampled_reprojected_shorelines_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'year',
                        'cross_distance', 'ci',
                        'geometry']
        tier_2_resampled_reprojected_shorelines_gdfs_concat = tier_2_resampled_reprojected_shorelines_gdfs_concat[column_order]

        ###annual shoreline lines
        annual_shoreline_lines = points_to_lines_no_gaps(tier_2_resampled_reprojected_shorelines_gdfs_concat, kind="annual")
        annual_shoreline_lines['G'] =annual_shoreline_lines['G'].astype(str)
        annual_shoreline_lines['C'] = annual_shoreline_lines['C'].astype(str)
        annual_shoreline_lines['RR'] = annual_shoreline_lines['RR'].astype(str)
        annual_shoreline_lines['SSS'] = annual_shoreline_lines['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS',
                        'year',
                        'geometry']
        annual_shoreline_lines = annual_shoreline_lines[column_order]

        ### trends
        tier_2_trends_gdfs_concat['transect_id'] = tier_2_trends_gdfs_concat['transect_id'].astype(str)

        tier_2_trends_gdfs_concat['linear_trend'] = tier_2_trends_gdfs_concat['linear_trend'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_95_confidence'] = tier_2_trends_gdfs_concat['linear_trend_95_confidence'].astype(float)

        tier_2_trends_gdfs_concat['intercept'] = tier_2_trends_gdfs_concat['intercept'].astype(float)
        tier_2_trends_gdfs_concat['intercept_95_confidence'] = tier_2_trends_gdfs_concat['intercept_95_confidence'].astype(float)

        # windowed tier_2_trends_gdfs_concat
        tier_2_trends_gdfs_concat['linear_trend_1980s'] = tier_2_trends_gdfs_concat['linear_trend_1980s'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_1990s'] = tier_2_trends_gdfs_concat['linear_trend_1990s'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_2000s'] = tier_2_trends_gdfs_concat['linear_trend_2000s'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_2010s'] = tier_2_trends_gdfs_concat['linear_trend_2010s'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_2020s'] = tier_2_trends_gdfs_concat['linear_trend_2020s'].astype(float)

        tier_2_trends_gdfs_concat['linear_trend_1980s_95_confidence'] = tier_2_trends_gdfs_concat['linear_trend_1980s_95_confidence'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_1990s_95_confidence'] = tier_2_trends_gdfs_concat['linear_trend_1990s_95_confidence'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_2000s_95_confidence'] = tier_2_trends_gdfs_concat['linear_trend_2000s_95_confidence'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_2010s_95_confidence'] = tier_2_trends_gdfs_concat['linear_trend_2010s_95_confidence'].astype(float)
        tier_2_trends_gdfs_concat['linear_trend_2020s_95_confidence'] = tier_2_trends_gdfs_concat['linear_trend_2020s_95_confidence'].astype(float)

        # significance flag
        tier_2_trends_gdfs_concat['significant'] = tier_2_trends_gdfs_concat['significant'].astype(int)

        # region codes
        tier_2_trends_gdfs_concat['G'] = tier_2_trends_gdfs_concat['G'].astype(str)
        tier_2_trends_gdfs_concat['C'] = tier_2_trends_gdfs_concat['C'].astype(str)
        tier_2_trends_gdfs_concat['RR'] = tier_2_trends_gdfs_concat['RR'].astype(str)
        tier_2_trends_gdfs_concat['SSS'] = tier_2_trends_gdfs_concat['SSS'].astype(str)
        tier_2_trends_gdfs_concat['V'] = ['0']*len(tier_2_trends_gdfs_concat)
        tier_2_trends_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_2_trends_gdfs_concat['transect_id']]
        # drop fields explicitly marked as DROP
        drop_cols = ['snr', 'figure_path', 'csv_path']
        for col in drop_cols:
            if col in tier_2_trends_gdfs_concat.columns:
                tier_2_trends_gdfs_concat = tier_2_trends_gdfs_concat.drop(columns=[col])
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'linear_trend', 'linear_trend_95_confidence',
                        'intercept', 'intercept_95_confidence',
                        'significant',
                        'linear_trend_1980s', 'linear_trend_1980s_95_confidence',
                        'linear_trend_1990s', 'linear_trend_1990s_95_confidence',
                        'linear_trend_2000s', 'linear_trend_2000s_95_confidence',
                        'linear_trend_2010s', 'linear_trend_2010s_95_confidence',
                        'linear_trend_2020s', 'linear_trend_2020s_95_confidence',
                        'geometry'
                        ]
        tier_2_trends_gdfs_concat = tier_2_trends_gdfs_concat[column_order]

        transects = gpd.read_file(os.path.join('/', 'mnt', 'f', 'SDSDataService_c_qc', '14_index.gpkg'), layer='00_transects')
        ### decadal_shoreline_points
        decadal_shoreline_points = compute_decadal_shoreline_points_with_ci(tier_2_resampled_reprojected_shorelines_gdfs_concat,transects)
        decadal_shoreline_points['cross_distance'] = decadal_shoreline_points['cross_distance'].astype(float)
        decadal_shoreline_points['transect_id'] = decadal_shoreline_points['transect_id'].astype(str)
        decadal_shoreline_points['ci'] = decadal_shoreline_points['ci'].astype(float)
        decadal_shoreline_points['decade'] = decadal_shoreline_points['decade'].astype(int)
        decadal_shoreline_points['G'] = [l[0] for l in decadal_shoreline_points['transect_id']]
        decadal_shoreline_points['C'] = [l[1] for l in decadal_shoreline_points['transect_id']]
        decadal_shoreline_points['RR'] = [l[2:4] for l in decadal_shoreline_points['transect_id']]
        decadal_shoreline_points['SSS'] = [l[4:7] for l in decadal_shoreline_points['transect_id']]
        decadal_shoreline_points['V'] = ['0']*len(decadal_shoreline_points)
        decadal_shoreline_points['LLLLLL'] = [l[-6:] for l in decadal_shoreline_points['transect_id']]
        decadal_shoreline_points['LLLLLL'] = (decadal_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'decade',
                        'cross_distance', 'ci',
                        'geometry']
        decadal_shoreline_points = decadal_shoreline_points[column_order]

        ###decadal shoreline lines
        decadal_shoreline_lines = points_to_lines_no_gaps(decadal_shoreline_points, kind="decadal")
        decadal_shoreline_lines['G'] =decadal_shoreline_lines['G'].astype(str)
        decadal_shoreline_lines['C'] = decadal_shoreline_lines['C'].astype(str)
        decadal_shoreline_lines['RR'] = decadal_shoreline_lines['RR'].astype(str)
        decadal_shoreline_lines['SSS'] = decadal_shoreline_lines['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS',
                        'decade',
                        'geometry']
        decadal_shoreline_lines = decadal_shoreline_lines[column_order]

        tier_2_spatial_kde_otsu_gdfs_concat['G'] = tier_2_spatial_kde_otsu_gdfs_concat['G'].astype(str)
        tier_2_spatial_kde_otsu_gdfs_concat['C'] = tier_2_spatial_kde_otsu_gdfs_concat['C'].astype(str)
        tier_2_spatial_kde_otsu_gdfs_concat['RR'] = tier_2_spatial_kde_otsu_gdfs_concat['RR'].astype(str)
        tier_2_spatial_kde_otsu_gdfs_concat['SSS'] = tier_2_spatial_kde_otsu_gdfs_concat['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
        tier_2_spatial_kde_otsu_gdfs_concat = tier_2_spatial_kde_otsu_gdfs_concat[column_order]
        tier_2_geopackage = os.path.join(home_geopackage, g+c+'_tier_2.gpkg')

        print('writing tier 2 geopackage')
        ##tier 2 files
        tier_2_dictionary_gdfs = {
                                '00_annual_shoreline_points':tier_2_resampled_reprojected_shorelines_gdfs_concat,
                                '01_annual_shoreline_lines':annual_shoreline_lines,
                                '02_annual_trends':tier_2_trends_gdfs_concat,
                                '03_decadal_shoreline_points': decadal_shoreline_points,
                                '04_decadal_shoreline_lines': decadal_shoreline_lines,
                                '05_shoreline_change_envelopes':tier_2_spatial_kde_otsu_gdfs_concat
                                }
        ##writing tier 2 geopackage
        gdf_dictionary_to_geopackage(tier_2_dictionary_gdfs, tier_2_geopackage)

    elif 'tier_3' in tiers:
        print('getting tier 3 files')
        ##getting tier 3 files
        tier_3_mean_shorelines_points_gdfs = [None]*num_sections
        tier_3_mean_shorelines_line_gdfs = [None]*num_sections
        tier_3_median_shorelines_points_gdfs = [None]*num_sections
        tier_3_median_shorelines_line_gdfs = [None]*num_sections
        tier_3_min_shorelines_points_gdfs = [None]*num_sections
        tier_3_min_shorelines_line_gdfs = [None]*num_sections
        tier_3_max_shorelines_points_gdfs = [None]*num_sections
        tier_3_max_shorelines_line_gdfs = [None]*num_sections
        tier_3_q1_shorelines_points_gdfs = [None]*num_sections
        tier_3_q1_shorelines_line_gdfs = [None]*num_sections
        tier_3_q3_shorelines_points_gdfs = [None]*num_sections
        tier_3_q3_shorelines_line_gdfs = [None]*num_sections
        i=0
        for rr in tqdm(rrs):
            r_home = os.path.join(home_data_service, 'Tier3', 'G'+g, 'C'+c, 'RR'+rr)
            sections = sorted(get_immediate_subdirectories(r_home))
            for section in tqdm(sections):
                section_dir = os.path.join(r_home, section)
                sss = section[3:] 
                section_string = g+c+rr+sss

                ##mean
                mean_shorelines_points_path = os.path.join(section_dir, section_string + '_mean_shoreline_points.geojson')
                fix_cols_gdf(mean_shorelines_points_path, g, c, rr, sss)
                tier_3_mean_shorelines_points_gdfs[i] = mean_shorelines_points_path
                
                mean_shorelines_path = os.path.join(section_dir, section_string + '_mean_shoreline.geojson')
                fix_cols_gdf(mean_shorelines_path, g, c, rr, sss)
                tier_3_mean_shorelines_line_gdfs[i] = mean_shorelines_path
                ##median
                median_shorelines_points_path = os.path.join(section_dir, section_string + '_median_shoreline_points.geojson')
                fix_cols_gdf(median_shorelines_points_path, g, c, rr, sss)
                tier_3_median_shorelines_points_gdfs[i] = median_shorelines_points_path
                
                median_shorelines_path = os.path.join(section_dir, section_string + '_median_shoreline.geojson')
                fix_cols_gdf(median_shorelines_path, g, c, rr, sss)
                tier_3_median_shorelines_line_gdfs[i] = median_shorelines_path 
                ##min
                min_shorelines_points_path = os.path.join(section_dir, section_string + '_min_shoreline_points.geojson')
                fix_cols_gdf(min_shorelines_points_path, g, c, rr, sss)
                tier_3_min_shorelines_points_gdfs[i] = min_shorelines_points_path
        
                min_shorelines_path = os.path.join(section_dir, section_string + '_min_shoreline.geojson')
                fix_cols_gdf(min_shorelines_path, g, c, rr, sss)
                tier_3_min_shorelines_line_gdfs[i] = min_shorelines_path 
                ##max
                max_shorelines_points_path = os.path.join(section_dir, section_string + '_max_shoreline_points.geojson')
                fix_cols_gdf(max_shorelines_points_path, g, c, rr, sss)
                tier_3_max_shorelines_points_gdfs[i] = max_shorelines_points_path
        
                max_shorelines_path = os.path.join(section_dir, section_string + '_max_shoreline.geojson')
                fix_cols_gdf(max_shorelines_path, g, c, rr, sss)
                tier_3_max_shorelines_line_gdfs[i] = max_shorelines_path 
                ##q1
                q1_shorelines_points_path = os.path.join(section_dir, section_string + '_q1_shoreline_points.geojson')
                fix_cols_gdf(q1_shorelines_points_path, g, c, rr, sss)
                tier_3_q1_shorelines_points_gdfs[i] = q1_shorelines_points_path
        
                q1_shorelines_path = os.path.join(section_dir, section_string + '_q1_shoreline.geojson')
                fix_cols_gdf(q1_shorelines_path, g, c, rr, sss)
                tier_3_q1_shorelines_line_gdfs[i] = q1_shorelines_path 
                ##q3
                q3_shorelines_points_path = os.path.join(section_dir, section_string + '_q3_shoreline_points.geojson')
                fix_cols_gdf(q3_shorelines_points_path, g, c, rr, sss)
                tier_3_q3_shorelines_points_gdfs[i] = q3_shorelines_points_path
        
                q3_shorelines_path = os.path.join(section_dir, section_string + '_q3_shoreline.geojson')
                fix_cols_gdf(q3_shorelines_path, g, c, rr, sss)
                tier_3_q3_shorelines_line_gdfs[i] = q3_shorelines_path 

                i=i+1

        print('concatenating tier 3 files')
        ##tier 3 files
        print('mean shorelines points')
        tier_3_mean_shorelines_points_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_mean_shorelines_points_gdfs) if os.path.isfile(f)==True])
        print('mean shorelines lines')
        tier_3_mean_shorelines_line_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_mean_shorelines_line_gdfs) if os.path.isfile(f)==True])
        print('median shorelines points')
        tier_3_median_shorelines_points_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_median_shorelines_points_gdfs) if os.path.isfile(f)==True])
        print('median shorelines lines')
        tier_3_median_shorelines_line_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_median_shorelines_line_gdfs) if os.path.isfile(f)==True])
        print('min shorelines points')
        tier_3_min_shorelines_points_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_min_shorelines_points_gdfs) if os.path.isfile(f)==True])
        print('min shorelines lines')
        tier_3_min_shorelines_line_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_min_shorelines_line_gdfs) if os.path.isfile(f)==True])
        print('max shorelines points')
        tier_3_max_shorelines_points_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_max_shorelines_points_gdfs) if os.path.isfile(f)==True])
        print('max shorelines lines')
        tier_3_max_shorelines_line_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_max_shorelines_line_gdfs) if os.path.isfile(f)==True])
        print('q1 shorelines points')
        tier_3_q1_shorelines_points_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_q1_shorelines_points_gdfs) if os.path.isfile(f)==True])
        print('q1 shorelines lines')
        tier_3_q1_shorelines_line_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_q1_shorelines_line_gdfs) if os.path.isfile(f)==True])
        print('q3 shorelines points')
        tier_3_q3_shorelines_points_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_q3_shorelines_points_gdfs) if os.path.isfile(f)==True])
        print('q3 shorelines lines')
        tier_3_q3_shorelines_line_gdfs_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(tier_3_q3_shorelines_line_gdfs) if os.path.isfile(f)==True])


        ### min_shoreline_points
        tier_3_min_shorelines_points_gdfs_concat['transect_id'] = tier_3_min_shorelines_points_gdfs_concat['transect_id'].astype(str)
        tier_3_min_shorelines_points_gdfs_concat['cross_distance_min'] = tier_3_min_shorelines_points_gdfs_concat['cross_distance_min'].astype(float)
        if 'geometry_from_centroids' in tier_3_min_shorelines_points_gdfs_concat.columns:
            tier_3_min_shorelines_points_gdfs_concat = tier_3_min_shorelines_points_gdfs_concat.drop(columns=['geometry_from_centroids'])
        tier_3_min_shorelines_points_gdfs_concat['G'] = tier_3_min_shorelines_points_gdfs_concat['G'].astype(str)
        tier_3_min_shorelines_points_gdfs_concat['C'] = tier_3_min_shorelines_points_gdfs_concat['C'].astype(str)
        tier_3_min_shorelines_points_gdfs_concat['RR'] = tier_3_min_shorelines_points_gdfs_concat['RR'].astype(str)
        tier_3_min_shorelines_points_gdfs_concat['SSS'] = tier_3_min_shorelines_points_gdfs_concat['SSS'].astype(str)
        tier_3_min_shorelines_points_gdfs_concat['V'] = ['0'] * len(tier_3_min_shorelines_points_gdfs_concat)
        tier_3_min_shorelines_points_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_3_min_shorelines_points_gdfs_concat['transect_id']]
        tier_3_min_shorelines_points_gdfs_concat['LLLLLL'] = (tier_3_min_shorelines_points_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'cross_distance_min',
                        'geometry']
        tier_3_min_shorelines_points_gdfs_concat = tier_3_min_shorelines_points_gdfs_concat[column_order]


        ### min_shoreline_lines
        tier_3_min_shorelines_line_gdfs_concat['G'] = tier_3_min_shorelines_line_gdfs_concat['G'].astype(str)
        tier_3_min_shorelines_line_gdfs_concat['C'] = tier_3_min_shorelines_line_gdfs_concat['C'].astype(str)
        tier_3_min_shorelines_line_gdfs_concat['RR'] = tier_3_min_shorelines_line_gdfs_concat['RR'].astype(str)
        tier_3_min_shorelines_line_gdfs_concat['SSS'] = tier_3_min_shorelines_line_gdfs_concat['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
        tier_3_min_shorelines_line_gdfs_concat = tier_3_min_shorelines_line_gdfs_concat[column_order]

    
        ### q1_shoreline_points
        tier_3_q1_shorelines_points_gdfs_concat['transect_id'] = tier_3_q1_shorelines_points_gdfs_concat['transect_id'].astype(str)
        tier_3_q1_shorelines_points_gdfs_concat['cross_distance_q1'] = tier_3_q1_shorelines_points_gdfs_concat['cross_distance_q1'].astype(float)
        if 'geometry_from_centroids' in tier_3_q1_shorelines_points_gdfs_concat.columns:
            tier_3_q1_shorelines_points_gdfs_concat = tier_3_q1_shorelines_points_gdfs_concat.drop(columns=['geometry_from_centroids'])
        tier_3_q1_shorelines_points_gdfs_concat['G'] = tier_3_q1_shorelines_points_gdfs_concat['G'].astype(str)
        tier_3_q1_shorelines_points_gdfs_concat['C'] = tier_3_q1_shorelines_points_gdfs_concat['C'].astype(str)
        tier_3_q1_shorelines_points_gdfs_concat['RR'] = tier_3_q1_shorelines_points_gdfs_concat['RR'].astype(str)
        tier_3_q1_shorelines_points_gdfs_concat['SSS'] = tier_3_q1_shorelines_points_gdfs_concat['SSS'].astype(str)
        tier_3_q1_shorelines_points_gdfs_concat['V'] = ['0'] * len(tier_3_q1_shorelines_points_gdfs_concat)
        tier_3_q1_shorelines_points_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_3_q1_shorelines_points_gdfs_concat['transect_id']]
        tier_3_q1_shorelines_points_gdfs_concat['LLLLLL'] = (tier_3_q1_shorelines_points_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'cross_distance_q1',
                        'geometry']
        tier_3_q1_shorelines_points_gdfs_concat = tier_3_q1_shorelines_points_gdfs_concat[column_order]

    
        ### q1_shoreline_lines
        tier_3_q1_shorelines_line_gdfs_concat['G'] = tier_3_q1_shorelines_line_gdfs_concat['G'].astype(str)
        tier_3_q1_shorelines_line_gdfs_concat['C'] = tier_3_q1_shorelines_line_gdfs_concat['C'].astype(str)
        tier_3_q1_shorelines_line_gdfs_concat['RR'] = tier_3_q1_shorelines_line_gdfs_concat['RR'].astype(str)
        tier_3_q1_shorelines_line_gdfs_concat['SSS'] = tier_3_q1_shorelines_line_gdfs_concat['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
        tier_3_q1_shorelines_line_gdfs_concat = tier_3_q1_shorelines_line_gdfs_concat[column_order]


        ### mean_shoreline_points
        tier_3_mean_shorelines_points_gdfs_concat['transect_id'] = tier_3_mean_shorelines_points_gdfs_concat['transect_id'].astype(str)
        tier_3_mean_shorelines_points_gdfs_concat['cross_distance_mean'] = tier_3_mean_shorelines_points_gdfs_concat['cross_distance_mean'].astype(float)
        if 'geometry_from_centroids' in tier_3_mean_shorelines_points_gdfs_concat.columns:
            tier_3_mean_shorelines_points_gdfs_concat = tier_3_mean_shorelines_points_gdfs_concat.drop(columns=['geometry_from_centroids'])
        tier_3_mean_shorelines_points_gdfs_concat['G'] = tier_3_mean_shorelines_points_gdfs_concat['G'].astype(str)
        tier_3_mean_shorelines_points_gdfs_concat['C'] = tier_3_mean_shorelines_points_gdfs_concat['C'].astype(str)
        tier_3_mean_shorelines_points_gdfs_concat['RR'] = tier_3_mean_shorelines_points_gdfs_concat['RR'].astype(str)
        tier_3_mean_shorelines_points_gdfs_concat['SSS'] = tier_3_mean_shorelines_points_gdfs_concat['SSS'].astype(str)
        tier_3_mean_shorelines_points_gdfs_concat['V'] = ['0'] * len(tier_3_mean_shorelines_points_gdfs_concat)
        tier_3_mean_shorelines_points_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_3_mean_shorelines_points_gdfs_concat['transect_id']]
        tier_3_mean_shorelines_points_gdfs_concat['LLLLLL'] = (tier_3_mean_shorelines_points_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'cross_distance_mean',
                        'geometry']
        tier_3_mean_shorelines_points_gdfs_concat = tier_3_mean_shorelines_points_gdfs_concat[column_order]


        ### mean_shoreline_lines
        tier_3_mean_shorelines_line_gdfs_concat['G'] = tier_3_mean_shorelines_line_gdfs_concat['G'].astype(str)
        tier_3_mean_shorelines_line_gdfs_concat['C'] = tier_3_mean_shorelines_line_gdfs_concat['C'].astype(str)
        tier_3_mean_shorelines_line_gdfs_concat['RR'] = tier_3_mean_shorelines_line_gdfs_concat['RR'].astype(str)
        tier_3_mean_shorelines_line_gdfs_concat['SSS'] = tier_3_mean_shorelines_line_gdfs_concat['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
        tier_3_mean_shorelines_line_gdfs_concat = tier_3_mean_shorelines_line_gdfs_concat[column_order]


        ### median_shoreline_points
        tier_3_median_shorelines_points_gdfs_concat['transect_id'] = tier_3_median_shorelines_points_gdfs_concat['transect_id'].astype(str)
        tier_3_median_shorelines_points_gdfs_concat['cross_distance_median'] = tier_3_median_shorelines_points_gdfs_concat['cross_distance_median'].astype(float)
        if 'geometry_from_centroids' in tier_3_median_shorelines_points_gdfs_concat.columns:
            tier_3_median_shorelines_points_gdfs_concat = tier_3_median_shorelines_points_gdfs_concat.drop(columns=['geometry_from_centroids'])
        tier_3_median_shorelines_points_gdfs_concat['G'] = tier_3_median_shorelines_points_gdfs_concat['G'].astype(str)
        tier_3_median_shorelines_points_gdfs_concat['C'] = tier_3_median_shorelines_points_gdfs_concat['C'].astype(str)
        tier_3_median_shorelines_points_gdfs_concat['RR'] = tier_3_median_shorelines_points_gdfs_concat['RR'].astype(str)
        tier_3_median_shorelines_points_gdfs_concat['SSS'] = tier_3_median_shorelines_points_gdfs_concat['SSS'].astype(str)
        tier_3_median_shorelines_points_gdfs_concat['V'] = ['0'] * len(tier_3_median_shorelines_points_gdfs_concat)
        tier_3_median_shorelines_points_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_3_median_shorelines_points_gdfs_concat['transect_id']]
        tier_3_median_shorelines_points_gdfs_concat['LLLLLL'] = (tier_3_median_shorelines_points_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'cross_distance_median', 
                        'iqr', 'q1', 'q3', 'mad', 'std', 'mean', 'cv', 'skewness','kurtosis',
                        'geometry']
        tier_3_median_shorelines_points_gdfs_concat = tier_3_median_shorelines_points_gdfs_concat[column_order]


        ### median_shoreline_lines
        tier_3_median_shorelines_line_gdfs_concat['G'] = tier_3_median_shorelines_line_gdfs_concat['G'].astype(str)
        tier_3_median_shorelines_line_gdfs_concat['C'] = tier_3_median_shorelines_line_gdfs_concat['C'].astype(str)
        tier_3_median_shorelines_line_gdfs_concat['RR'] = tier_3_median_shorelines_line_gdfs_concat['RR'].astype(str)
        tier_3_median_shorelines_line_gdfs_concat['SSS'] = tier_3_median_shorelines_line_gdfs_concat['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
        tier_3_median_shorelines_line_gdfs_concat = tier_3_median_shorelines_line_gdfs_concat[column_order]


        ### q3_shoreline_points
        tier_3_q3_shorelines_points_gdfs_concat['transect_id'] = tier_3_q3_shorelines_points_gdfs_concat['transect_id'].astype(str)
        tier_3_q3_shorelines_points_gdfs_concat['cross_distance_q3'] = tier_3_q3_shorelines_points_gdfs_concat['cross_distance_q3'].astype(float)
        if 'geometry_from_centroids' in tier_3_q3_shorelines_points_gdfs_concat.columns:
            tier_3_q3_shorelines_points_gdfs_concat = tier_3_q3_shorelines_points_gdfs_concat.drop(columns=['geometry_from_centroids'])
        tier_3_q3_shorelines_points_gdfs_concat['G'] = tier_3_q3_shorelines_points_gdfs_concat['G'].astype(str)
        tier_3_q3_shorelines_points_gdfs_concat['C'] = tier_3_q3_shorelines_points_gdfs_concat['C'].astype(str)
        tier_3_q3_shorelines_points_gdfs_concat['RR'] = tier_3_q3_shorelines_points_gdfs_concat['RR'].astype(str)
        tier_3_q3_shorelines_points_gdfs_concat['SSS'] = tier_3_q3_shorelines_points_gdfs_concat['SSS'].astype(str)
        tier_3_q3_shorelines_points_gdfs_concat['V'] = ['0'] * len(tier_3_q3_shorelines_points_gdfs_concat)
        tier_3_q3_shorelines_points_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_3_q3_shorelines_points_gdfs_concat['transect_id']]
        tier_3_q3_shorelines_points_gdfs_concat['LLLLLL'] = (tier_3_q3_shorelines_points_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',

                        'cross_distance_q3',
                        'geometry']
        tier_3_q3_shorelines_points_gdfs_concat = tier_3_q3_shorelines_points_gdfs_concat[column_order]


        ### q3_shoreline_lines
        tier_3_q3_shorelines_line_gdfs_concat['G'] = tier_3_q3_shorelines_line_gdfs_concat['G'].astype(str)
        tier_3_q3_shorelines_line_gdfs_concat['C'] = tier_3_q3_shorelines_line_gdfs_concat['C'].astype(str)
        tier_3_q3_shorelines_line_gdfs_concat['RR'] = tier_3_q3_shorelines_line_gdfs_concat['RR'].astype(str)
        tier_3_q3_shorelines_line_gdfs_concat['SSS'] = tier_3_q3_shorelines_line_gdfs_concat['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
        tier_3_q3_shorelines_line_gdfs_concat = tier_3_q3_shorelines_line_gdfs_concat[column_order]


        ### max_shoreline_points
        tier_3_max_shorelines_points_gdfs_concat['transect_id'] = tier_3_max_shorelines_points_gdfs_concat['transect_id'].astype(str)
        tier_3_max_shorelines_points_gdfs_concat['cross_distance_max'] = tier_3_max_shorelines_points_gdfs_concat['cross_distance_max'].astype(float)
        if 'geometry_from_centroids' in tier_3_max_shorelines_points_gdfs_concat.columns:
            tier_3_max_shorelines_points_gdfs_concat = tier_3_max_shorelines_points_gdfs_concat.drop(columns=['geometry_from_centroids'])
        tier_3_max_shorelines_points_gdfs_concat['G'] = tier_3_max_shorelines_points_gdfs_concat['G'].astype(str)
        tier_3_max_shorelines_points_gdfs_concat['C'] = tier_3_max_shorelines_points_gdfs_concat['C'].astype(str)
        tier_3_max_shorelines_points_gdfs_concat['RR'] = tier_3_max_shorelines_points_gdfs_concat['RR'].astype(str)
        tier_3_max_shorelines_points_gdfs_concat['SSS'] = tier_3_max_shorelines_points_gdfs_concat['SSS'].astype(str)
        tier_3_max_shorelines_points_gdfs_concat['V'] = ['0']*len(tier_3_max_shorelines_points_gdfs_concat)
        tier_3_max_shorelines_points_gdfs_concat['LLLLLL'] = [l[-6:] for l in tier_3_max_shorelines_points_gdfs_concat['transect_id']]
        tier_3_max_shorelines_points_gdfs_concat['LLLLLL'] = (tier_3_max_shorelines_points_gdfs_concat['LLLLLL'].astype(int).astype(str).str.zfill(6))
        column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                        'cross_distance_max',
                        'geometry']
        tier_3_max_shorelines_points_gdfs_concat = tier_3_max_shorelines_points_gdfs_concat[column_order]


        ### max_shoreline_lines
        tier_3_max_shorelines_line_gdfs_concat['G'] = tier_3_max_shorelines_line_gdfs_concat['G'].astype(str)
        tier_3_max_shorelines_line_gdfs_concat['C'] = tier_3_max_shorelines_line_gdfs_concat['C'].astype(str)
        tier_3_max_shorelines_line_gdfs_concat['RR'] = tier_3_max_shorelines_line_gdfs_concat['RR'].astype(str)
        tier_3_max_shorelines_line_gdfs_concat['SSS'] = tier_3_max_shorelines_line_gdfs_concat['SSS'].astype(str)
        column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
        tier_3_max_shorelines_line_gdfs_concat = tier_3_max_shorelines_line_gdfs_concat[column_order]

        ##tier 3 files
        tier_3_dictionary_gdfs = {'00_min_shoreline_points':tier_3_min_shorelines_points_gdfs_concat,
                                '01_min_shoreline_lines':tier_3_min_shorelines_line_gdfs_concat,
                                '02_q1_shoreline_points':tier_3_q1_shorelines_points_gdfs_concat,
                                '03_q1_shoreline_lines':tier_3_q1_shorelines_line_gdfs_concat,
                                '04_mean_shoreline_points':tier_3_mean_shorelines_points_gdfs_concat,
                                '05_mean_shoreline_lines':tier_3_mean_shorelines_line_gdfs_concat,
                                '06_median_shoreline_points':tier_3_median_shorelines_points_gdfs_concat,
                                '07_median_shoreline_lines':tier_3_median_shorelines_line_gdfs_concat,
                                '08_q3_shoreline_points':tier_3_q3_shorelines_points_gdfs_concat,
                                '09_q3_shoreline_lines':tier_3_q3_shorelines_line_gdfs_concat,
                                '10_max_shoreline_points':tier_3_max_shorelines_points_gdfs_concat,
                                '11_max_shoreline_lines':tier_3_max_shorelines_line_gdfs_concat
                                }
        tier_3_geopackage = os.path.join(home_geopackage, g+c+'_tier_3.gpkg')
        print('writing tier 3 geopackage')
        ##writing tier 3 geopackage
        gdf_dictionary_to_geopackage(tier_3_dictionary_gdfs, tier_3_geopackage)


"""
Here we are creating two different directory structures for the data we want.
    G
        C
            RR
                SSS
                    Index
                        xxx.geojson
                    Tier0
                        xxx.geojson
                    Tier1
                        xxx.geojson
                    Tier2
                        xxx.geojson
                    Tier3
                        xxx.geojson
Index
    G
        C
            RR
                SSS
                    xxx.geojson
Tier0
    G
        C
            RR
                SSS
                    xxx.geojson
Tier1
    G
        C
            RR
                SSS
                    xxx.geojson
Tier2
    G
        C
            RR
                SSS
                    xxx.geojson
Tier3
    G
        C
            RR
                SSS
                    xxx.geojson
"""
g = '1'
c = '4'
rrs = ['00', '01', '02', '03']
home = os.path.join('/', 'mnt', 'f', 'SDSDataService_b')
print('tiering data')
for rr in tqdm(rrs):
    print(rr)
    r_home_analysis = os.path.join('/', 'mnt', 'hdd_6tb', 'Alaska_Analysis_Images', 'G'+g, 'C'+c, 'RR'+rr)
    r_home_data_service = os.path.join('/', 'mnt', 'f', 'SDSDataService_a', 'G'+g, 'C'+c, 'RR'+rr)
    try:
        os.makedirs(r_home_data_service)
    except:
        pass
    sections = sorted(get_immediate_subdirectories(r_home_analysis))
    print(sections)
    for section in tqdm(sections):
        print(section)
        sss = section[3:]
        tier_last_section(g, c, rr, sss, r_home_analysis, r_home_data_service)
        tier_first_section(g, c, rr, sss, home, r_home_analysis)



"""
Here we are concacenating geojsons and then writing to geopackage layers
GC_index.gpkg
GC_tier_0.gpkg
GC_tier_1.gpkg
GC_tier_2.gpkg
GC_tier_3.gpkg
"""
g = '1'
c = '4'
rrs = ['00', '01', '02', '03']
home_data_service = os.path.join('/', 'mnt', 'f', 'SDSDataService_b')
home_geopackage = os.path.join('/', 'mnt', 'f', 'SDSDataService_c')
tier_to_gpkg(g, c, rrs, home_data_service, home_geopackage, ['index', 'tier_0', 'tier_1', 'tier_2', 'tier_3'])

