import geopandas as gpd
import pandas as pd
import os
from tqdm import tqdm

def get_immediate_subdirectories(a_dir):
    """
    gets immediate subdirectories, only returns name not full path
    inputs:
    a_dir (str): path to that directory

    outputs:
    a list of the immediate subdirectory names
    """
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

g='1'
c='2'
rrs = ['04', '05', '06']
roi_list = []
reference_shoreline_list = []
reference_polygon_list = []
transect_list = []
for rr in rrs:
    r_home = os.path.join('/', 'mnt','hdd_6tb', 'Alaska_Analysis_Images', 'G'+g, 'C'+c, 'RR'+rr)
    sections = sorted(get_immediate_subdirectories(r_home))
    for section in sections:
        section_dir = os.path.join(r_home, section)
        sss = section[3:]
        print(section)
        transects = os.path.join(section_dir, g+c+rr+sss+'_transects.geojson')
        transects_gdf = gpd.read_file(transects)
        transects_columns = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 'geometry']
        transects_gdf['G'] = ['1']*len(transects_gdf)
        transects_gdf['C'] = ['2']*len(transects_gdf)
        transects_gdf['RR'] = [rr]*len(transects_gdf)
        transects_gdf['V'] = ['0']*len(transects_gdf)
        transects_gdf['SSS'] = [sss]*len(transects_gdf)
        # longshore distance in 50 m increments, zero‑padded to 6 digits
        transects_gdf['LLLLLL'] = (
            (transects_gdf.index * 50)
            .astype(int)
            .astype(str)
            .str.zfill(6)
        )

        # Build transect_id = g + c + rr + sss + v + llllll
        transects_gdf['transect_id'] = (
            transects_gdf['G'].astype(str)
            + transects_gdf['C'].astype(str)
            + transects_gdf['RR'].astype(str)
            + transects_gdf['SSS'].astype(str)
            + transects_gdf['V'].astype(str)
            + transects_gdf['LLLLLL'].astype(str)
        )
        transects_gdf = transects_gdf[transects_columns]
        print('transects')
        print(transects_gdf.head())
        transects_gdf.to_file(transects)

        roi = os.path.join(section_dir, g+c+rr+sss+'_rois.geojson')
        roi_gdf = gpd.read_file(roi)
        roi_columns = ['G', 'C', 'RR', 'SSS',  'geometry']
        roi_gdf['G'] = ['1']*len(roi_gdf)
        roi_gdf['C'] = ['2']*len(roi_gdf)
        roi_gdf['RR'] = [rr]*len(roi_gdf)
        roi_gdf['SSS'] = [sss]*len(roi_gdf)
        roi_gdf = roi_gdf[roi_columns]
        print('rois')
        print(roi_gdf.head())
        roi_gdf.to_file(roi)

        reference_shoreline = os.path.join(section_dir, g+c+rr+sss+'_reference_shoreline.geojson')
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
        reference_shoreline_columns = ['G', 'C', 'RR', 'SSS', 'geometry']
        reference_shoreline_gdf['G'] = ['1']*len(reference_shoreline_gdf)
        reference_shoreline_gdf['C'] = ['2']*len(reference_shoreline_gdf)
        reference_shoreline_gdf['RR'] = [rr]*len(reference_shoreline_gdf)
        reference_shoreline_gdf['SSS'] = [sss]*len(reference_shoreline_gdf)
        reference_shoreline_gdf = reference_shoreline_gdf[reference_shoreline_columns]
        print('reference shoreline')
        print(reference_shoreline_gdf.head())
        reference_shoreline_gdf.to_file(reference_shoreline)

        reference_polygon = os.path.join(section_dir, g+c+rr+sss+'_reference_polygon.geojson')
        reference_polygon_gdf = gpd.read_file(reference_polygon)
        reference_polygon_columns = ['G', 'C', 'RR', 'SSS', 'geometry']
        reference_polygon_gdf['G'] = ['1']*len(reference_polygon_gdf)
        reference_polygon_gdf['C'] = ['2']*len(reference_polygon_gdf)
        reference_polygon_gdf['RR'] = [rr]*len(reference_polygon_gdf)
        reference_polygon_gdf['SSS'] = [sss]*len(reference_polygon_gdf)
        reference_polygon_gdf = reference_polygon_gdf[reference_polygon_columns]
        print('reference polygon')
        print(reference_polygon_gdf.head())
        reference_polygon_gdf.to_file(reference_polygon)
        roi_list.append(roi)
        reference_shoreline_list.append(reference_shoreline)
        reference_polygon_list.append(reference_polygon)
        transect_list.append(transects)

transects_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(transect_list) if os.path.isfile(f)==True])
rois_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(roi_list) if os.path.isfile(f)==True])
reference_polygons_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(reference_shoreline_list) if os.path.isfile(f)==True])
reference_shorelines_concat = pd.concat([gpd.read_file(f).to_crs(4326) for f in tqdm(reference_polygon_list) if os.path.isfile(f)==True])

gpkg_path = os.path.join('/', 'mnt', 'f', '12_index.gpkg')
transects_concat.to_file(gpkg_path, layer='00_transects')
rois_concat.to_file(gpkg_path, layer='01_rois')
reference_shorelines_concat.to_file(gpkg_path, layer='02_reference_shorelines')
reference_polygons_concat.to_file(gpkg_path, layer='03_reference_polygons')

