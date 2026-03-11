import geopandas as gpd
import pandas as pd
import numpy as np
import os



def qc_index_columns(index, index_qc):
    """
    Strictly enforcing attribute column data types and dropping unnecessary columns
    """
    print('fixing transects')
    transects = gpd.read_file(index, layer='00_transects')
    print(transects.columns)
    transects['G'] = transects['G'].astype(str)
    transects['C'] = transects['C'].astype(str)
    transects['RR'] = transects['RR'].astype(str)
    transects['SSS'] = transects['SSS'].astype(str)
    transects['V'] = transects['V'].astype(str)
    transects['LLLLLL'] = [l[-6:] for l in transects['transect_id']]
    transects['LLLLLL'] = (transects['LLLLLL'].astype(int).astype(str).str.zfill(6))
    transects['transect_id'] = transects['transect_id'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 'geometry']
    transects = transects[column_order]
    transects.to_file(index_qc, layer='00_transects')
    transects = None
    print('transects fixed')

    print('fixing rois')
    rois = gpd.read_file(index, layer='01_rois')
    print(rois.columns)
    rois['G'] = rois['G'].astype(str)
    rois['C'] = rois['C'].astype(str)
    rois['RR'] = rois['RR'].astype(str)
    rois['SSS'] = rois['SSS'].astype(str)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
    rois = rois[columns_to_keep]
    rois.to_file(index_qc, layer='01_rois')
    rois = None
    print('rois fixed')

    print('fixing reference_shorelines')
    reference_shorelines = gpd.read_file(index, layer='02_reference_shorelines')
    print(reference_shorelines.columns)
    reference_shorelines['G'] = reference_shorelines['G'].astype(str)
    reference_shorelines['C'] = reference_shorelines['C'].astype(str)
    reference_shorelines['RR'] = reference_shorelines['RR'].astype(str)
    reference_shorelines['SSS'] = reference_shorelines['SSS'].astype(str)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
    reference_shorelines = reference_shorelines[columns_to_keep]
    reference_shorelines.to_file(index_qc, layer='02_reference_shorelines')
    reference_shorelines = None
    print('reference shorelines fixed')

    print('fixing reference polygons')
    reference_polygons = gpd.read_file(index, layer='03_reference_polygons')
    print(reference_polygons.columns)
    reference_polygons['G'] = reference_polygons['G'].astype(str)
    reference_polygons['C'] = reference_polygons['C'].astype(str)
    reference_polygons['RR'] = reference_polygons['RR'].astype(str)
    reference_polygons['SSS'] = reference_polygons['SSS'].astype(str)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
    reference_polygons = reference_polygons[columns_to_keep]
    reference_polygons.to_file(index_qc, layer='03_reference_polygons')
    reference_polygons = None
    print('reference polygons fixed')

    print('fixing crest1s')
    crest1s = gpd.read_file(index, layer='04_crest1s')
    print(crest1s.columns)
    crest1s['G'] = crest1s['G'].astype(str)
    crest1s['C'] = crest1s['C'].astype(str)
    crest1s['RR'] = crest1s['RR'].astype(str)
    crest1s['SSS'] = crest1s['SSS'].astype(str)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
    crest1s = crest1s[columns_to_keep]
    crest1s.to_file(index_qc, layer='04_crest1s')
    crest1s = None
    print('crest1s fixed')

    print('fixing crest2s')
    crest2s = gpd.read_file(index, layer='05_crest2s')
    print(crest2s.columns)
    crest2s['G'] = crest2s['G'].astype(str)
    crest2s['C'] = crest2s['C'].astype(str)
    crest2s['RR'] = crest2s['RR'].astype(str)
    crest2s['SSS'] = crest2s['SSS'].astype(str)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
    crest2s = crest2s[columns_to_keep]
    crest2s.to_file(index_qc, layer='05_crest2s')
    crest2s = None
    print('crest2s fixed')

    print('fixing crest3s')
    crest3s = gpd.read_file(index, layer='06_crest3s')
    print(crest3s.columns)
    crest3s['G'] = crest3s['G'].astype(str)
    crest3s['C'] = crest3s['C'].astype(str)
    crest3s['RR'] = crest3s['RR'].astype(str)
    crest3s['SSS'] = crest3s['SSS'].astype(str)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
    crest3s = crest3s[columns_to_keep]
    crest3s.to_file(index_qc, layer='06_crest3s')
    crest3s = None
    print('crest3s fixed')

    print('fixing inflection points')
    inflection_points = gpd.read_file(index, layer='07_inflection_points')
    print(inflection_points.columns)
    inflection_points['G'] = inflection_points['G'].astype(str)
    inflection_points['C'] = inflection_points['C'].astype(str)
    inflection_points['RR'] = inflection_points['RR'].astype(str)
    inflection_points['SSS'] = inflection_points['SSS'].astype(str)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
    inflection_points = inflection_points[columns_to_keep]
    inflection_points.to_file(index_qc, layer='07_inflection_points')
    inflection_points = None
    print('inflection points fixed')

    print('fixing toes')
    toes = gpd.read_file(index, layer='08_toes')
    print(toes.columns)
    toes['G'] = toes['G'].astype(str)
    toes['C'] = toes['C'].astype(str)
    toes['RR'] = toes['RR'].astype(str)
    toes['SSS'] = toes['SSS'].astype(str)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'geometry']
    toes = toes[columns_to_keep]
    toes.to_file(index_qc, layer='08_toes')
    toes = None
    print('toes fixed')

    print('fixing ArcticDEM_slopes')
    ArcticDEM_slopes = gpd.read_file(index, layer='09_ArcticDEM_slopes')
    print(ArcticDEM_slopes.columns)
    ArcticDEM_slopes['G'] = ArcticDEM_slopes['G'].astype(str)
    ArcticDEM_slopes['C'] = ArcticDEM_slopes['C'].astype(str)
    ArcticDEM_slopes['RR'] = ArcticDEM_slopes['RR'].astype(str)
    ArcticDEM_slopes['SSS'] = ArcticDEM_slopes['SSS'].astype(str)
    ArcticDEM_slopes['V'] = ['0']*len(ArcticDEM_slopes)
    ArcticDEM_slopes['transect_id'] = ArcticDEM_slopes['transect_id'].astype(str)
    ArcticDEM_slopes['LLLLLL'] = [l[-6:] for l in ArcticDEM_slopes['transect_id']]
    ArcticDEM_slopes['LLLLLL'] = (ArcticDEM_slopes['LLLLLL'].astype(int).astype(str).str.zfill(6))
    ArcticDEM_slopes['max_slope'] = ArcticDEM_slopes['max_slope'].astype(float)
    ArcticDEM_slopes['median_slope'] = ArcticDEM_slopes['median_slope'].astype(float)
    ArcticDEM_slopes['avg_slope_cleaned'] = ArcticDEM_slopes['avg_slope_cleaned'].astype(float)
    ArcticDEM_slopes['avg_slope'] = ArcticDEM_slopes['avg_slope'].astype(float)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL',
                       'transect_id', 
                       'max_slope', 'median_slope', 'avg_slope_cleaned', 'avg_slope', 'geometry']
    for col in ArcticDEM_slopes.columns:
        if col not in columns_to_keep:
            try:
                ArcticDEM_slopes = ArcticDEM_slopes.drop(columns=[col])
            except:
                pass
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'max_slope', 'median_slope', 'avg_slope', 
                    'avg_slope_cleaned',
                    'geometry']
    ArcticDEM_slopes = ArcticDEM_slopes[column_order]
    ArcticDEM_slopes.to_file(index_qc, layer='09_ArcticDEM_slopes')
    ArcticDEM_slopes = None
    print('ArcticDEM_slopes fixed')

    print('fixing tbdem_slopes')
    tbdem_slopes = gpd.read_file(index, layer='10_tbdem_slopes')
    print(tbdem_slopes.columns)
    tbdem_slopes['G'] = tbdem_slopes['G'].astype(str)
    tbdem_slopes['C'] = tbdem_slopes['C'].astype(str)
    tbdem_slopes['RR'] = tbdem_slopes['RR'].astype(str)
    tbdem_slopes['SSS'] = tbdem_slopes['SSS'].astype(str)
    tbdem_slopes['V'] = ['0']*len(tbdem_slopes)
    tbdem_slopes['transect_id'] = tbdem_slopes['transect_id'].astype(str)
    tbdem_slopes['LLLLLL'] = [l[-6:] for l in tbdem_slopes['transect_id']]
    tbdem_slopes['LLLLLL'] = (tbdem_slopes['LLLLLL'].astype(int).astype(str).str.zfill(6))
    tbdem_slopes['max_slope'] = tbdem_slopes['max_slope'].astype(float)
    tbdem_slopes['median_slope'] = tbdem_slopes['median_slope'].astype(float)
    tbdem_slopes['avg_slope_cleaned'] = tbdem_slopes['avg_slope_cleaned'].astype(float)
    tbdem_slopes['avg_slope'] = tbdem_slopes['avg_slope'].astype(float)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL',
                       'transect_id', 
                       'max_slope', 'median_slope', 'avg_slope_cleaned', 'avg_slope', 'geometry']
    for col in tbdem_slopes.columns:
        if col not in columns_to_keep:
            try:
                tbdem_slopes = tbdem_slopes.drop(columns=[col])
            except:
                pass
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'max_slope', 'median_slope', 'avg_slope', 
                    'avg_slope_cleaned',
                    'geometry']
    tbdem_slopes = tbdem_slopes[column_order]
    tbdem_slopes.to_file(index_qc, layer='10_tbdem_slopes')
    tbdem_slopes = None
    print('tbdem_slopes fixed')

    print('fixing AlaskaDSM_slopes')
    AlaskaDSM_slopes = gpd.read_file(index, layer='11_AlaskaDSM_slopes')
    print(AlaskaDSM_slopes.columns)
    AlaskaDSM_slopes['G'] = AlaskaDSM_slopes['G'].astype(str)
    AlaskaDSM_slopes['C'] = AlaskaDSM_slopes['C'].astype(str)
    AlaskaDSM_slopes['RR'] = AlaskaDSM_slopes['RR'].astype(str)
    AlaskaDSM_slopes['SSS'] = AlaskaDSM_slopes['SSS'].astype(str)
    AlaskaDSM_slopes['V'] = ['0']*len(AlaskaDSM_slopes)
    AlaskaDSM_slopes['transect_id'] = AlaskaDSM_slopes['transect_id'].astype(str)
    AlaskaDSM_slopes['LLLLLL'] = [l[-6:] for l in AlaskaDSM_slopes['transect_id']]
    AlaskaDSM_slopes['LLLLLL'] = (AlaskaDSM_slopes['LLLLLL'].astype(int).astype(str).str.zfill(6))
    AlaskaDSM_slopes['max_slope'] = AlaskaDSM_slopes['max_slope'].astype(float)
    AlaskaDSM_slopes['median_slope'] = AlaskaDSM_slopes['median_slope'].astype(float)
    AlaskaDSM_slopes['avg_slope_cleaned'] = AlaskaDSM_slopes['avg_slope_cleaned'].astype(float)
    AlaskaDSM_slopes['avg_slope'] = AlaskaDSM_slopes['avg_slope'].astype(float)
    columns_to_keep = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL',
                       'transect_id', 
                       'max_slope', 'median_slope', 'avg_slope_cleaned', 'avg_slope', 'geometry']
    for col in AlaskaDSM_slopes.columns:
        if col not in columns_to_keep:
            try:
                AlaskaDSM_slopes = AlaskaDSM_slopes.drop(columns=[col])
            except:
                pass
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'max_slope', 'median_slope', 'avg_slope', 
                    'avg_slope_cleaned',
                    'geometry']
    AlaskaDSM_slopes = AlaskaDSM_slopes[column_order]
    AlaskaDSM_slopes.to_file(index_qc, layer='11_AlaskaDSM_slopes')
    AlaskaDSM_slopes = None
    print('AlaskaDSM_slopes fixed')

def qc_tier_0_columns(tier_0, tier_0_qc):
    ######zoo rgb
    zoo_rgb_waterlines = gpd.read_file(tier_0, layer='00_zoo_rgb_waterlines')
    zoo_rgb_waterlines['dates'] = zoo_rgb_waterlines['dates'].astype(str)
    zoo_rgb_waterlines['image_suitability_score'] = zoo_rgb_waterlines['image_suitability_score'].astype(float)
    zoo_rgb_waterlines['segmentation_suitability_score'] = zoo_rgb_waterlines['segmentation_suitability_score'].astype(float)
    zoo_rgb_waterlines['satname'] = zoo_rgb_waterlines['satname'].astype(str)
    zoo_rgb_waterlines['simplify_param'] =  zoo_rgb_waterlines['simplify_param'].astype(float)
    zoo_rgb_waterlines['kde_value'] =  zoo_rgb_waterlines['kde_value'].astype(float)
    zoo_rgb_waterlines['G'] = zoo_rgb_waterlines['G'].astype(str)
    zoo_rgb_waterlines['C'] = zoo_rgb_waterlines['C'].astype(str)
    zoo_rgb_waterlines['RR'] = zoo_rgb_waterlines['RR'].astype(str)
    zoo_rgb_waterlines['SSS'] = zoo_rgb_waterlines['SSS'].astype(str)
    zoo_rgb_waterlines['year'] = pd.to_datetime(zoo_rgb_waterlines['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
    column_order = ['G', 'C', 'RR', 'SSS', 
                    'dates', 'satname', 'year',
                    'image_suitability_score', 'segmentation_suitability_score', 
                    'simplify_param', 'kde_value', 
                    'geometry']
    zoo_rgb_waterlines = zoo_rgb_waterlines[column_order]
    zoo_rgb_waterlines.to_file(tier_0_qc, layer='00_zoo_rgb_waterlines')
    zoo_rgb_waterlines = None
    
    zoo_rgb_waterlines_filter = gpd.read_file(tier_0, layer='01_zoo_rgb_waterlines_filter')
    zoo_rgb_waterlines_filter['dates'] = zoo_rgb_waterlines_filter['dates'].astype(str)
    zoo_rgb_waterlines_filter['image_suitability_score'] = zoo_rgb_waterlines_filter['image_suitability_score'].astype(float)
    zoo_rgb_waterlines_filter['segmentation_suitability_score'] = zoo_rgb_waterlines_filter['segmentation_suitability_score'].astype(float)
    zoo_rgb_waterlines_filter['satname'] = zoo_rgb_waterlines_filter['satname'].astype(str)
    zoo_rgb_waterlines_filter['simplify_param'] =  zoo_rgb_waterlines_filter['simplify_param'].astype(float)
    zoo_rgb_waterlines_filter['kde_value'] =  zoo_rgb_waterlines_filter['kde_value'].astype(float)
    zoo_rgb_waterlines_filter['G'] = zoo_rgb_waterlines_filter['G'].astype(str)
    zoo_rgb_waterlines_filter['C'] = zoo_rgb_waterlines_filter['C'].astype(str)
    zoo_rgb_waterlines_filter['RR'] = zoo_rgb_waterlines_filter['RR'].astype(str)
    zoo_rgb_waterlines_filter['SSS'] = zoo_rgb_waterlines_filter['SSS'].astype(str)
    zoo_rgb_waterlines_filter['year'] = pd.to_datetime(zoo_rgb_waterlines_filter['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
    column_order = ['G', 'C', 'RR', 'SSS', 
                    'dates', 'satname', 'year',
                    'image_suitability_score', 'segmentation_suitability_score', 
                    'simplify_param', 'kde_value', 
                    'geometry']
    zoo_rgb_waterlines_filter = zoo_rgb_waterlines_filter[column_order]
    zoo_rgb_waterlines_filter.to_file(tier_0_qc, layer='01_zoo_rgb_waterlines_filter')
    zoo_rgb_waterlines_filter = None
    
    zoo_rgb_time_series_tidally_corrected = gpd.read_file(tier_0, layer='02_zoo_rgb_time_series')
    zoo_rgb_time_series_tidally_corrected['dates'] =zoo_rgb_time_series_tidally_corrected['dates'].astype(str)
    zoo_rgb_time_series_tidally_corrected['image_suitability_score'] = zoo_rgb_time_series_tidally_corrected['image_suitability_score'].astype(float)
    zoo_rgb_time_series_tidally_corrected['segmentation_suitability_score'] = zoo_rgb_time_series_tidally_corrected['segmentation_suitability_score'].astype(float)
    zoo_rgb_time_series_tidally_corrected['kde_value'] = zoo_rgb_time_series_tidally_corrected['kde_value'].astype(float)
    zoo_rgb_time_series_tidally_corrected['transect_id'] = zoo_rgb_time_series_tidally_corrected['transect_id'].astype(str)
    zoo_rgb_time_series_tidally_corrected['intersect_x'] = zoo_rgb_time_series_tidally_corrected['intersect_x'].astype(float)
    zoo_rgb_time_series_tidally_corrected['intersect_y'] = zoo_rgb_time_series_tidally_corrected['intersect_y'].astype(float)
    zoo_rgb_time_series_tidally_corrected['cross_distance'] = zoo_rgb_time_series_tidally_corrected['cross_distance'].astype(float)
    zoo_rgb_time_series_tidally_corrected['tide'] = zoo_rgb_time_series_tidally_corrected['tide'].astype(float)
    zoo_rgb_time_series_tidally_corrected['x'] = zoo_rgb_time_series_tidally_corrected['x'].astype(float)
    zoo_rgb_time_series_tidally_corrected['y'] = zoo_rgb_time_series_tidally_corrected['y'].astype(float)
    zoo_rgb_time_series_tidally_corrected['avg_slope_cleaned'] = zoo_rgb_time_series_tidally_corrected['avg_slope_cleaned'].astype(float)
    zoo_rgb_time_series_tidally_corrected['G'] = zoo_rgb_time_series_tidally_corrected['G'].astype(str)
    zoo_rgb_time_series_tidally_corrected['C'] = zoo_rgb_time_series_tidally_corrected['C'].astype(str)
    zoo_rgb_time_series_tidally_corrected['RR'] = zoo_rgb_time_series_tidally_corrected['RR'].astype(str)
    zoo_rgb_time_series_tidally_corrected['SSS'] = zoo_rgb_time_series_tidally_corrected['SSS'].astype(str)
    zoo_rgb_time_series_tidally_corrected['V'] = ['0']*len(zoo_rgb_time_series_tidally_corrected)
    zoo_rgb_time_series_tidally_corrected['satname'] = zoo_rgb_time_series_tidally_corrected['satname'].astype(str)
    zoo_rgb_time_series_tidally_corrected['LLLLLL'] = [l[-6:] for l in zoo_rgb_time_series_tidally_corrected['transect_id']]
    zoo_rgb_time_series_tidally_corrected['LLLLLL'] = (zoo_rgb_time_series_tidally_corrected['LLLLLL'].astype(int).astype(str).str.zfill(6))
    zoo_rgb_time_series_tidally_corrected['year'] = pd.to_datetime(zoo_rgb_time_series_tidally_corrected['dates'], utc=True, format='ISO8601').dt.year
    keep_cols = ['dates', 'image_suitability_score', 'segmentation_suitability_score',
                 'kde_value', 'satname', 'transect_id', 'intersect_x', 'intersect_y','cross_distance_tidally_corrected', 
                 'cross_distance', 'G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'tide', 'x', 'y', 'year', 'avg_slope_cleaned', 'geometry']
    zoo_rgb_time_series_tidally_corrected = zoo_rgb_time_series_tidally_corrected[[c for c in keep_cols if c in zoo_rgb_time_series_tidally_corrected.columns]]
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                    'dates', 'satname', 'year',
                    'cross_distance', 'cross_distance_tidally_corrected',
                    'image_suitability_score', 'segmentation_suitability_score',
                    'kde_value', 'intersect_x', 'intersect_y', 'tide', 'x', 'y', 'avg_slope_cleaned',
                    'geometry']
    zoo_rgb_time_series_tidally_corrected = zoo_rgb_time_series_tidally_corrected[column_order]
    zoo_rgb_time_series_tidally_corrected.to_file(tier_0_qc, layer='02_zoo_rgb_time_series')
    zoo_rgb_time_series_tidally_corrected = None

    ###nir
    nir_waterlines = gpd.read_file(tier_0, layer='03_nir_waterlines')
    nir_waterlines['dates'] = nir_waterlines['dates'].astype(str)
    nir_waterlines['image_suitability_score'] = nir_waterlines['image_suitability_score'].astype(float)
    nir_waterlines['segmentation_suitability_score'] = nir_waterlines['segmentation_suitability_score'].astype(float)
    nir_waterlines['satname'] = nir_waterlines['satname'].astype(str)
    nir_waterlines['simplify_param'] =  nir_waterlines['simplify_param'].astype(float)
    nir_waterlines['kde_value'] =  nir_waterlines['kde_value'].astype(float)
    nir_waterlines['G'] = nir_waterlines['G'].astype(str)
    nir_waterlines['C'] = nir_waterlines['C'].astype(str)
    nir_waterlines['RR'] = nir_waterlines['RR'].astype(str)
    nir_waterlines['SSS'] = nir_waterlines['SSS'].astype(str)
    nir_waterlines['year'] =pd.to_datetime(nir_waterlines['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
    column_order = ['G', 'C', 'RR', 'SSS', 
                    'dates', 'satname', 'year',
                    'image_suitability_score', 'segmentation_suitability_score', 
                    'simplify_param', 'kde_value', 
                    'geometry']
    nir_waterlines = nir_waterlines[column_order]
    nir_waterlines.to_file(tier_0_qc, layer='03_nir_waterlines')
    nir_waterlines = None
    
    nir_waterlines_filter = gpd.read_file(tier_0, layer='04_nir_waterlines_filter')
    nir_waterlines_filter['dates'] = nir_waterlines_filter['dates'].astype(str)
    nir_waterlines_filter['image_suitability_score'] = nir_waterlines_filter['image_suitability_score'].astype(float)
    nir_waterlines_filter['segmentation_suitability_score'] = nir_waterlines_filter['segmentation_suitability_score'].astype(float)
    nir_waterlines_filter['satname'] = nir_waterlines_filter['satname'].astype(str)
    nir_waterlines_filter['simplify_param'] =  nir_waterlines_filter['simplify_param'].astype(float)
    nir_waterlines_filter['kde_value'] =  nir_waterlines_filter['kde_value'].astype(float)
    nir_waterlines_filter['G'] = nir_waterlines_filter['G'].astype(str)
    nir_waterlines_filter['C'] = nir_waterlines_filter['C'].astype(str)
    nir_waterlines_filter['RR'] = nir_waterlines_filter['RR'].astype(str)
    nir_waterlines_filter['SSS'] = nir_waterlines_filter['SSS'].astype(str)
    nir_waterlines_filter['year'] = pd.to_datetime(nir_waterlines_filter['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
    column_order = ['G', 'C', 'RR', 'SSS', 
                    'dates', 'satname', 'year',
                    'image_suitability_score', 'segmentation_suitability_score', 
                    'simplify_param', 'kde_value', 
                    'geometry']
    nir_waterlines_filter = nir_waterlines_filter[column_order]                   
    nir_waterlines_filter.to_file(tier_0_qc, layer='04_nir_waterlines_filter')
    nir_waterlines_filter = None
    
    nir_time_series_tidally_corrected = gpd.read_file(tier_0, layer='05_nir_time_series')
    nir_time_series_tidally_corrected['dates'] =nir_time_series_tidally_corrected['dates'].astype(str)
    nir_time_series_tidally_corrected['image_suitability_score'] = nir_time_series_tidally_corrected['image_suitability_score'].astype(float)
    nir_time_series_tidally_corrected['segmentation_suitability_score'] = nir_time_series_tidally_corrected['segmentation_suitability_score'].astype(float)
    nir_time_series_tidally_corrected['kde_value'] = nir_time_series_tidally_corrected['kde_value'].astype(float)
    nir_time_series_tidally_corrected['transect_id'] = nir_time_series_tidally_corrected['transect_id'].astype(str)
    nir_time_series_tidally_corrected['intersect_x'] = nir_time_series_tidally_corrected['intersect_x'].astype(float)
    nir_time_series_tidally_corrected['intersect_y'] = nir_time_series_tidally_corrected['intersect_y'].astype(float)
    nir_time_series_tidally_corrected['cross_distance'] = nir_time_series_tidally_corrected['cross_distance'].astype(float)
    nir_time_series_tidally_corrected['tide'] = nir_time_series_tidally_corrected['tide'].astype(float)
    nir_time_series_tidally_corrected['x'] = nir_time_series_tidally_corrected['x'].astype(float)
    nir_time_series_tidally_corrected['y'] = nir_time_series_tidally_corrected['y'].astype(float)
    nir_time_series_tidally_corrected['avg_slope_cleaned'] = nir_time_series_tidally_corrected['avg_slope_cleaned'].astype(float)
    nir_time_series_tidally_corrected['G'] = nir_time_series_tidally_corrected['G'].astype(str)
    nir_time_series_tidally_corrected['C'] = nir_time_series_tidally_corrected['C'].astype(str)
    nir_time_series_tidally_corrected['RR'] = nir_time_series_tidally_corrected['RR'].astype(str)
    nir_time_series_tidally_corrected['SSS'] = nir_time_series_tidally_corrected['SSS'].astype(str)
    nir_time_series_tidally_corrected['satname'] = nir_time_series_tidally_corrected['satname'].astype(str)
    nir_time_series_tidally_corrected['V'] = ['0']*len(nir_time_series_tidally_corrected)
    nir_time_series_tidally_corrected['LLLLLL'] = [l[-6:] for l in nir_time_series_tidally_corrected['transect_id']]
    nir_time_series_tidally_corrected['LLLLLL'] = (nir_time_series_tidally_corrected['LLLLLL'].astype(int).astype(str).str.zfill(6))
    nir_time_series_tidally_corrected['year'] = pd.to_datetime(nir_time_series_tidally_corrected['dates'], utc=True, format='ISO8601').dt.year
    keep_cols = ['dates', 'image_suitability_score', 'segmentation_suitability_score',
                 'kde_value', 'satname', 'transect_id', 'intersect_x', 'intersect_y', 'year','cross_distance_tidally_corrected', 
                 'cross_distance', 'G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'tide', 'x', 'y', 'avg_slope_cleaned', 'geometry']
    nir_time_series_tidally_corrected = nir_time_series_tidally_corrected[[c for c in keep_cols if c in nir_time_series_tidally_corrected.columns]]
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                    'dates', 'satname', 'year',
                    'cross_distance', 'cross_distance_tidally_corrected',
                    'image_suitability_score', 'segmentation_suitability_score',
                    'kde_value', 'intersect_x', 'intersect_y', 'tide', 'x', 'y', 'avg_slope_cleaned',
                    'geometry']
    nir_time_series_tidally_corrected = nir_time_series_tidally_corrected[column_order]
    nir_time_series_tidally_corrected.to_file(tier_0_qc, layer='05_nir_time_series')
    nir_time_series_tidally_corrected = None


    ####swir
    swir_waterlines = gpd.read_file(tier_0, layer='06_swir_waterlines')
    swir_waterlines['dates'] = swir_waterlines['dates'].astype(str)
    swir_waterlines['image_suitability_score'] = swir_waterlines['image_suitability_score'].astype(float)
    swir_waterlines['segmentation_suitability_score'] = swir_waterlines['segmentation_suitability_score'].astype(float)
    swir_waterlines['satname'] = swir_waterlines['satname'].astype(str)
    swir_waterlines['simplify_param'] =  swir_waterlines['simplify_param'].astype(float)
    swir_waterlines['kde_value'] =  swir_waterlines['kde_value'].astype(float)
    swir_waterlines['G'] = swir_waterlines['G'].astype(str)
    swir_waterlines['C'] = swir_waterlines['C'].astype(str)
    swir_waterlines['RR'] = swir_waterlines['RR'].astype(str)
    swir_waterlines['SSS'] = swir_waterlines['SSS'].astype(str)
    swir_waterlines['year'] =pd.to_datetime(swir_waterlines['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
    column_order = ['G', 'C', 'RR', 'SSS', 
                    'dates', 'satname', 'year',
                    'image_suitability_score', 'segmentation_suitability_score', 
                    'simplify_param', 'kde_value', 
                    'geometry']
    swir_waterlines = swir_waterlines[column_order]
    swir_waterlines.to_file(tier_0_qc, layer='06_swir_waterlines')
    swir_waterlines = None
    
    swir_waterlines_filter = gpd.read_file(tier_0, layer='07_swir_waterlines_filter')
    swir_waterlines_filter['dates'] = swir_waterlines_filter['dates'].astype(str)
    swir_waterlines_filter['image_suitability_score'] = swir_waterlines_filter['image_suitability_score'].astype(float)
    swir_waterlines_filter['segmentation_suitability_score'] = swir_waterlines_filter['segmentation_suitability_score'].astype(float)
    swir_waterlines_filter['satname'] = swir_waterlines_filter['satname'].astype(str)
    swir_waterlines_filter['simplify_param'] =  swir_waterlines_filter['simplify_param'].astype(float)
    swir_waterlines_filter['kde_value'] =  swir_waterlines_filter['kde_value'].astype(float)
    swir_waterlines_filter['G'] = swir_waterlines_filter['G'].astype(str)
    swir_waterlines_filter['C'] = swir_waterlines_filter['C'].astype(str)
    swir_waterlines_filter['RR'] = swir_waterlines_filter['RR'].astype(str)
    swir_waterlines_filter['SSS'] = swir_waterlines_filter['SSS'].astype(str)
    swir_waterlines_filter['year'] = pd.to_datetime(swir_waterlines_filter['dates'], utc=True, format='%Y-%m-%d-%H-%M-%S').dt.year
    column_order = ['G', 'C', 'RR', 'SSS', 
                    'dates', 'satname', 'year',
                    'image_suitability_score', 'segmentation_suitability_score', 
                    'simplify_param', 'kde_value', 
                    'geometry']
    swir_waterlines_filter = swir_waterlines_filter[column_order]
    swir_waterlines_filter.to_file(tier_0_qc, layer='07_swir_waterlines_filter')
    swir_waterlines_filter = None
    
    swir_time_series_tidally_corrected = gpd.read_file(tier_0, layer='08_swir_time_series')
    swir_time_series_tidally_corrected['dates'] =swir_time_series_tidally_corrected['dates'].astype(str)
    swir_time_series_tidally_corrected['image_suitability_score'] = swir_time_series_tidally_corrected['image_suitability_score'].astype(float)
    swir_time_series_tidally_corrected['segmentation_suitability_score'] = swir_time_series_tidally_corrected['segmentation_suitability_score'].astype(float)
    swir_time_series_tidally_corrected['kde_value'] = swir_time_series_tidally_corrected['kde_value'].astype(float)
    swir_time_series_tidally_corrected['transect_id'] = swir_time_series_tidally_corrected['transect_id'].astype(str)
    swir_time_series_tidally_corrected['intersect_x'] = swir_time_series_tidally_corrected['intersect_x'].astype(float)
    swir_time_series_tidally_corrected['intersect_y'] = swir_time_series_tidally_corrected['intersect_y'].astype(float)
    swir_time_series_tidally_corrected['cross_distance'] = swir_time_series_tidally_corrected['cross_distance'].astype(float)
    swir_time_series_tidally_corrected['tide'] = swir_time_series_tidally_corrected['tide'].astype(float)
    swir_time_series_tidally_corrected['x'] = swir_time_series_tidally_corrected['x'].astype(float)
    swir_time_series_tidally_corrected['y'] = swir_time_series_tidally_corrected['y'].astype(float)
    swir_time_series_tidally_corrected['avg_slope_cleaned'] = swir_time_series_tidally_corrected['avg_slope_cleaned'].astype(float)
    swir_time_series_tidally_corrected['G'] = swir_time_series_tidally_corrected['G'].astype(str)
    swir_time_series_tidally_corrected['C'] = swir_time_series_tidally_corrected['C'].astype(str)
    swir_time_series_tidally_corrected['RR'] = swir_time_series_tidally_corrected['RR'].astype(str)
    swir_time_series_tidally_corrected['SSS'] = swir_time_series_tidally_corrected['SSS'].astype(str)
    swir_time_series_tidally_corrected['satname'] = swir_time_series_tidally_corrected['satname'].astype(str)
    swir_time_series_tidally_corrected['V'] = ['0']*len(swir_time_series_tidally_corrected)
    swir_time_series_tidally_corrected['LLLLLL'] = [l[-6:] for l in swir_time_series_tidally_corrected['transect_id']]
    swir_time_series_tidally_corrected['LLLLLL'] = (swir_time_series_tidally_corrected['LLLLLL'].astype(int).astype(str).str.zfill(6))
    swir_time_series_tidally_corrected['year'] = pd.to_datetime(swir_time_series_tidally_corrected['dates'], utc=True, format='ISO8601').dt.year
    keep_cols = ['dates', 'image_suitability_score', 'segmentation_suitability_score',
                 'kde_value', 'satname', 'transect_id', 'intersect_x', 'intersect_y',
                 'cross_distance','cross_distance_tidally_corrected', 'G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'tide', 'x', 'y', 'avg_slope_cleaned', 'geometry']
    swir_time_series_tidally_corrected = swir_time_series_tidally_corrected[[c for c in keep_cols if c in swir_time_series_tidally_corrected.columns]]
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                    'dates', 'satname', 
                    'cross_distance', 'cross_distance_tidally_corrected',
                    'image_suitability_score', 'segmentation_suitability_score',
                    'kde_value', 'intersect_x', 'intersect_y', 'tide', 'x', 'y', 'avg_slope_cleaned',
                    'geometry']
    swir_time_series_tidally_corrected = swir_time_series_tidally_corrected[column_order]
    swir_time_series_tidally_corrected.to_file(tier_0_qc, layer='08_swir_time_series')
    swir_time_series_tidally_corrected = None
    
def qc_tier_1_columns(tier_1, tier_1_qc):
    unfiltered_shoreline_points = gpd.read_file(tier_1, layer='00_unfiltered_shoreline_points')
    unfiltered_shoreline_points['cross_distance'] = unfiltered_shoreline_points['cross_distance'].astype(float)    
    unfiltered_shoreline_points['cross_distance_rgb'] = unfiltered_shoreline_points['cross_distance_rgb'].astype(float)
    unfiltered_shoreline_points['cross_distance_nir'] = unfiltered_shoreline_points['cross_distance_nir'].astype(float)
    unfiltered_shoreline_points['cross_distance_swir'] = unfiltered_shoreline_points['cross_distance_swir'].astype(float)
    unfiltered_shoreline_points['transect_id'] = unfiltered_shoreline_points['transect_id'].astype(str)
    unfiltered_shoreline_points['avg_suitability'] = unfiltered_shoreline_points['avg_suitability'].astype(float)
    unfiltered_shoreline_points['satname'] = unfiltered_shoreline_points['satname'].astype(str)
    unfiltered_shoreline_points['avg_slope'] = unfiltered_shoreline_points['avg_slope'].astype(float)
    unfiltered_shoreline_points['tide'] = unfiltered_shoreline_points['tide'].astype(float)
    unfiltered_shoreline_points['ci'] = unfiltered_shoreline_points['ci'].astype(float)
    unfiltered_shoreline_points['year'] = unfiltered_shoreline_points['year'].astype(int)
    unfiltered_shoreline_points['G'] = unfiltered_shoreline_points['G'].astype(str)
    unfiltered_shoreline_points['C'] = unfiltered_shoreline_points['C'].astype(str)
    unfiltered_shoreline_points['RR'] = unfiltered_shoreline_points['RR'].astype(str)
    unfiltered_shoreline_points['SSS'] = unfiltered_shoreline_points['SSS'].astype(str)
    unfiltered_shoreline_points['V'] = ['0']*len(unfiltered_shoreline_points)
    unfiltered_shoreline_points['LLLLLL'] = [l[-6:] for l in unfiltered_shoreline_points['transect_id']]
    unfiltered_shoreline_points['LLLLLL'] = (unfiltered_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                    'dates', 'satname',
                    'cross_distance', 'cross_distance_rgb', 'cross_distance_nir', 'cross_distance_swir', 'ci',
                    'avg_suitability', 
                    'avg_slope','tide',
                    'year',
                    'geometry']
    unfiltered_shoreline_points = unfiltered_shoreline_points[column_order]
    unfiltered_shoreline_points.to_file(tier_1_qc, layer='00_unfiltered_shoreline_points')
    unfiltered_shoreline_points = None
    
    shoreline_points = gpd.read_file(tier_1, layer='01_shoreline_points')
    shoreline_points['cross_distance'] = shoreline_points['cross_distance'].astype(float)    
    shoreline_points['cross_distance_rgb'] = shoreline_points['cross_distance_rgb'].astype(float)
    shoreline_points['cross_distance_nir'] = shoreline_points['cross_distance_nir'].astype(float)
    shoreline_points['cross_distance_swir'] = shoreline_points['cross_distance_swir'].astype(float)
    shoreline_points['transect_id'] = shoreline_points['transect_id'].astype(str)
    shoreline_points['avg_suitability'] = shoreline_points['avg_suitability'].astype(float)
    shoreline_points['satname'] = shoreline_points['satname'].astype(str)
    shoreline_points['avg_slope'] = shoreline_points['avg_slope'].astype(float)
    shoreline_points['tide'] = shoreline_points['tide'].astype(float)
    shoreline_points['ci'] = shoreline_points['ci'].astype(float)
    shoreline_points['year'] = shoreline_points['year'].astype(int)
    shoreline_points['G'] = shoreline_points['G'].astype(str)
    shoreline_points['C'] = shoreline_points['C'].astype(str)
    shoreline_points['RR'] = shoreline_points['RR'].astype(str)
    shoreline_points['SSS'] = shoreline_points['SSS'].astype(str)
    shoreline_points['V'] = ['0']*len(shoreline_points)
    shoreline_points['LLLLLL'] = [l[-6:] for l in shoreline_points['transect_id']]
    shoreline_points['LLLLLL'] = (shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id', 
                    'dates', 'satname',
                    'cross_distance', 'cross_distance_rgb', 'cross_distance_nir', 'cross_distance_swir', 'ci',
                    'avg_suitability', 
                    'avg_slope','tide',
                    'year',
                    'geometry']
    shoreline_points = shoreline_points[column_order]
    shoreline_points.to_file(tier_1_qc, layer='01_shoreline_points')
    shoreline_points = None
    
    shoreline_lines = gpd.read_file(tier_1, layer='02_shoreline_lines')
    shoreline_lines['G'] = shoreline_lines['G'].astype(str)
    shoreline_lines['C'] = shoreline_lines['C'].astype(str)
    shoreline_lines['RR'] = shoreline_lines['RR'].astype(str)
    shoreline_lines['SSS'] = shoreline_lines['SSS'].astype(str)
    shoreline_lines['dates'] = pd.to_datetime(shoreline_lines['dates'])
    shoreline_lines['year'] = shoreline_lines['dates'].dt.year
    shoreline_lines['dates'] = shoreline_lines['dates'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS',
                    'dates','year',
                    'geometry']
    shoreline_lines = shoreline_lines[column_order]
    shoreline_lines.to_file(tier_1_qc, layer='02_shoreline_lines')
    shoreline_lines = None

def qc_tier_2_columns(tier_2, tier_2_qc):

    ### annual_shoreline_points
    annual_shoreline_points = gpd.read_file(tier_2, layer='00_annual_shoreline_points')
    annual_shoreline_points['cross_distance'] = annual_shoreline_points['cross_distance'].astype(float)
    annual_shoreline_points['transect_id'] = annual_shoreline_points['transect_id'].astype(str)
    annual_shoreline_points['ci'] = annual_shoreline_points['ci'].astype(float)
    annual_shoreline_points['year'] = annual_shoreline_points['year'].astype(int)
    annual_shoreline_points['G'] = annual_shoreline_points['G'].astype(str)
    annual_shoreline_points['C'] = annual_shoreline_points['C'].astype(str)
    annual_shoreline_points['RR'] = annual_shoreline_points['RR'].astype(str)
    annual_shoreline_points['SSS'] = annual_shoreline_points['SSS'].astype(str)
    annual_shoreline_points['V'] = ['0']*len(annual_shoreline_points)
    annual_shoreline_points['LLLLLL'] = [l[-6:] for l in annual_shoreline_points['transect_id']]
    annual_shoreline_points['LLLLLL'] = (annual_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'year',
                    'cross_distance', 'ci',
                    'geometry']
    annual_shoreline_points = annual_shoreline_points[column_order]
    annual_shoreline_points.to_file(tier_2_qc, layer='00_annual_shoreline_points')
    annual_shoreline_points = None

    ###annual shoreline lines
    annual_shoreline_lines = gpd.read_file(tier_2, layer='01_annual_shoreline_lines')
    annual_shoreline_lines['G'] =annual_shoreline_lines['G'].astype(str)
    annual_shoreline_lines['C'] = annual_shoreline_lines['C'].astype(str)
    annual_shoreline_lines['RR'] = annual_shoreline_lines['RR'].astype(str)
    annual_shoreline_lines['SSS'] = annual_shoreline_lines['SSS'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS',
                    'year',
                    'geometry']
    annual_shoreline_lines = annual_shoreline_lines[column_order]
    annual_shoreline_lines.to_file(tier_2_qc, layer='01_annual_shoreline_lines')
    annual_shoreline_lines = None

    ### trends
    trends = gpd.read_file(tier_2, layer='02_trends')

    trends['transect_id'] = trends['transect_id'].astype(str)

    trends['linear_trend'] = trends['linear_trend'].astype(float)
    trends['linear_trend_95_confidence'] = trends['linear_trend_95_confidence'].astype(float)

    trends['intercept'] = trends['intercept'].astype(float)
    trends['intercept_95_confidence'] = trends['intercept_95_confidence'].astype(float)

    # windowed trends
    trends['linear_trend_1980s'] = trends['linear_trend_1980s'].astype(float)
    trends['linear_trend_1990s'] = trends['linear_trend_1990s'].astype(float)
    trends['linear_trend_2000s'] = trends['linear_trend_2000s'].astype(float)
    trends['linear_trend_2010s'] = trends['linear_trend_2010s'].astype(float)
    trends['linear_trend_2020s'] = trends['linear_trend_2020s'].astype(float)

    trends['linear_trend_1980s_95_confidence'] = trends['linear_trend_1980s_95_confidence'].astype(float)
    trends['linear_trend_1990s_95_confidence'] = trends['linear_trend_1990s_95_confidence'].astype(float)
    trends['linear_trend_2000s_95_confidence'] = trends['linear_trend_2000s_95_confidence'].astype(float)
    trends['linear_trend_2010s_95_confidence'] = trends['linear_trend_2010s_95_confidence'].astype(float)
    trends['linear_trend_2020s_95_confidence'] = trends['linear_trend_2020s_95_confidence'].astype(float)

    # significance flag
    trends['significant'] = trends['significant'].astype(int)

    # region codes
    trends['G'] = trends['G'].astype(str)
    trends['C'] = trends['C'].astype(str)
    trends['RR'] = trends['RR'].astype(str)
    trends['SSS'] = trends['SSS'].astype(str)
    trends['V'] = ['0']*len(trends)
    trends['LLLLLL'] = [l[-6:] for l in trends['transect_id']]
    # drop fields explicitly marked as DROP
    drop_cols = ['snr', 'figure_path', 'csv_path']
    for col in drop_cols:
        if col in trends.columns:
            trends = trends.drop(columns=[col])
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
    trends = trends[column_order]
    trends.to_file(tier_2_qc, layer='02_annual_trends')
    trends = None

    ### decadal_shoreline_points
    decadal_shoreline_points = gpd.read_file(tier_2, layer='03_decadal_shoreline_points')
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
    decadal_shoreline_points.to_file(tier_2_qc, layer='03_decadal_shoreline_points')
    decadal_shoreline_points = None

    ###decadal shoreline lines
    decadal_shoreline_lines = gpd.read_file(tier_2, layer='04_decadal_shoreline_lines')
    decadal_shoreline_lines['G'] =decadal_shoreline_lines['G'].astype(str)
    decadal_shoreline_lines['C'] = decadal_shoreline_lines['C'].astype(str)
    decadal_shoreline_lines['RR'] = decadal_shoreline_lines['RR'].astype(str)
    decadal_shoreline_lines['SSS'] = decadal_shoreline_lines['SSS'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS',
                    'decade',
                    'geometry']
    decadal_shoreline_lines = decadal_shoreline_lines[column_order]
    decadal_shoreline_lines.to_file(tier_2_qc, layer='04_decadal_shoreline_lines')
    decadal_shoreline_lines = None

    shoreline_change_envelopes = gpd.read_file(tier_2, layer='05_shoreline_change_envelopes')
    shoreline_change_envelopes['G'] = shoreline_change_envelopes['G'].astype(str)
    shoreline_change_envelopes['C'] = shoreline_change_envelopes['C'].astype(str)
    shoreline_change_envelopes['RR'] = shoreline_change_envelopes['RR'].astype(str)
    shoreline_change_envelopes['SSS'] = shoreline_change_envelopes['SSS'].astype(str)
    shoreline_change_envelopes.to_file(tier_2_qc, layer='05_shoreline_change_envelopes')
    shoreline_change_envelopes = None

def qc_tier_3_columns(tier_3, tier_3_qc):
    ### min_shoreline_points
    min_shoreline_points = gpd.read_file(tier_3, layer='00_min_shoreline_points')
    min_shoreline_points['transect_id'] = min_shoreline_points['transect_id'].astype(str)
    min_shoreline_points['cross_distance_min'] = min_shoreline_points['cross_distance_min'].astype(float)
    if 'geometry_from_centroids' in min_shoreline_points.columns:
        min_shoreline_points = min_shoreline_points.drop(columns=['geometry_from_centroids'])
    min_shoreline_points['G'] = min_shoreline_points['G'].astype(str)
    min_shoreline_points['C'] = min_shoreline_points['C'].astype(str)
    min_shoreline_points['RR'] = min_shoreline_points['RR'].astype(str)
    min_shoreline_points['SSS'] = min_shoreline_points['SSS'].astype(str)
    min_shoreline_points['V'] = ['0'] * len(min_shoreline_points)
    min_shoreline_points['LLLLLL'] = [l[-6:] for l in min_shoreline_points['transect_id']]
    min_shoreline_points['LLLLLL'] = (min_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'cross_distance_min',
                    'geometry']
    min_shoreline_points = min_shoreline_points[column_order]
    min_shoreline_points.to_file(tier_3_qc, layer='00_min_shoreline_points')
    min_shoreline_points = None

    ### min_shoreline_lines
    min_shoreline_lines = gpd.read_file(tier_3, layer='01_min_shoreline_lines')
    min_shoreline_lines['G'] = min_shoreline_lines['G'].astype(str)
    min_shoreline_lines['C'] = min_shoreline_lines['C'].astype(str)
    min_shoreline_lines['RR'] = min_shoreline_lines['RR'].astype(str)
    min_shoreline_lines['SSS'] = min_shoreline_lines['SSS'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
    min_shoreline_lines = min_shoreline_lines[column_order]
    min_shoreline_lines.to_file(tier_3_qc, layer='01_min_shoreline_lines')
    min_shoreline_lines = None
   
    ### q1_shoreline_points
    q1_shoreline_points = gpd.read_file(tier_3, layer='02_q1_shoreline_points')
    q1_shoreline_points['transect_id'] = q1_shoreline_points['transect_id'].astype(str)
    q1_shoreline_points['cross_distance_q1'] = q1_shoreline_points['cross_distance_q1'].astype(float)
    if 'geometry_from_centroids' in q1_shoreline_points.columns:
        q1_shoreline_points = q1_shoreline_points.drop(columns=['geometry_from_centroids'])
    q1_shoreline_points['G'] = q1_shoreline_points['G'].astype(str)
    q1_shoreline_points['C'] = q1_shoreline_points['C'].astype(str)
    q1_shoreline_points['RR'] = q1_shoreline_points['RR'].astype(str)
    q1_shoreline_points['SSS'] = q1_shoreline_points['SSS'].astype(str)
    q1_shoreline_points['V'] = ['0'] * len(q1_shoreline_points)
    q1_shoreline_points['LLLLLL'] = [l[-6:] for l in q1_shoreline_points['transect_id']]
    q1_shoreline_points['LLLLLL'] = (q1_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'cross_distance_q1',
                    'geometry']
    q1_shoreline_points = q1_shoreline_points[column_order]
    q1_shoreline_points.to_file(tier_3_qc, layer='02_q1_shoreline_points')
    q1_shoreline_points = None
   
    ### q1_shoreline_lines
    q1_shoreline_lines = gpd.read_file(tier_3, layer='03_q1_shoreline_lines')
    q1_shoreline_lines['G'] = q1_shoreline_lines['G'].astype(str)
    q1_shoreline_lines['C'] = q1_shoreline_lines['C'].astype(str)
    q1_shoreline_lines['RR'] = q1_shoreline_lines['RR'].astype(str)
    q1_shoreline_lines['SSS'] = q1_shoreline_lines['SSS'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
    q1_shoreline_lines = q1_shoreline_lines[column_order]
    q1_shoreline_lines.to_file(tier_3_qc, layer='03_q1_shoreline_lines')
    q1_shoreline_lines = None

    ### mean_shoreline_points
    mean_shoreline_points = gpd.read_file(tier_3, layer='04_mean_shoreline_points')
    mean_shoreline_points['transect_id'] = mean_shoreline_points['transect_id'].astype(str)
    mean_shoreline_points['cross_distance_mean'] = mean_shoreline_points['cross_distance_mean'].astype(float)
    if 'geometry_from_centroids' in mean_shoreline_points.columns:
        mean_shoreline_points = mean_shoreline_points.drop(columns=['geometry_from_centroids'])
    mean_shoreline_points['G'] = mean_shoreline_points['G'].astype(str)
    mean_shoreline_points['C'] = mean_shoreline_points['C'].astype(str)
    mean_shoreline_points['RR'] = mean_shoreline_points['RR'].astype(str)
    mean_shoreline_points['SSS'] = mean_shoreline_points['SSS'].astype(str)
    mean_shoreline_points['V'] = ['0'] * len(mean_shoreline_points)
    mean_shoreline_points['LLLLLL'] = [l[-6:] for l in mean_shoreline_points['transect_id']]
    mean_shoreline_points['LLLLLL'] = (mean_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'cross_distance_mean',
                    'geometry']
    mean_shoreline_points = mean_shoreline_points[column_order]
    mean_shoreline_points.to_file(tier_3_qc, layer='04_mean_shoreline_points')
    mean_shoreline_points = None

    ### mean_shoreline_lines
    mean_shoreline_lines = gpd.read_file(tier_3, layer='05_mean_shoreline_lines')
    mean_shoreline_lines['G'] = mean_shoreline_lines['G'].astype(str)
    mean_shoreline_lines['C'] = mean_shoreline_lines['C'].astype(str)
    mean_shoreline_lines['RR'] = mean_shoreline_lines['RR'].astype(str)
    mean_shoreline_lines['SSS'] = mean_shoreline_lines['SSS'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
    mean_shoreline_lines = mean_shoreline_lines[column_order]
    mean_shoreline_lines.to_file(tier_3_qc, layer='05_mean_shoreline_lines')
    mean_shoreline_lines = None

    ### median_shoreline_points
    median_shoreline_points = gpd.read_file(tier_3, layer='06_median_shoreline_points')
    median_shoreline_points['transect_id'] = median_shoreline_points['transect_id'].astype(str)
    median_shoreline_points['cross_distance_median'] = median_shoreline_points['cross_distance_median'].astype(float)
    if 'geometry_from_centroids' in median_shoreline_points.columns:
        median_shoreline_points = median_shoreline_points.drop(columns=['geometry_from_centroids'])
    median_shoreline_points['G'] = median_shoreline_points['G'].astype(str)
    median_shoreline_points['C'] = median_shoreline_points['C'].astype(str)
    median_shoreline_points['RR'] = median_shoreline_points['RR'].astype(str)
    median_shoreline_points['SSS'] = median_shoreline_points['SSS'].astype(str)
    median_shoreline_points['V'] = ['0'] * len(median_shoreline_points)
    median_shoreline_points['LLLLLL'] = [l[-6:] for l in median_shoreline_points['transect_id']]
    median_shoreline_points['LLLLLL'] = (median_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'cross_distance_median', 
                    'iqr', 'q1', 'q3', 'mad', 'std', 'mean', 'cv', 'skewness','kurtosis',
                    'geometry']
    median_shoreline_points = median_shoreline_points[column_order]
    median_shoreline_points.to_file(tier_3_qc, layer='06_median_shoreline_points')
    median_shoreline_points = None

    ### median_shoreline_lines
    median_shoreline_lines = gpd.read_file(tier_3, layer='07_median_shoreline_lines')
    median_shoreline_lines['G'] = median_shoreline_lines['G'].astype(str)
    median_shoreline_lines['C'] = median_shoreline_lines['C'].astype(str)
    median_shoreline_lines['RR'] = median_shoreline_lines['RR'].astype(str)
    median_shoreline_lines['SSS'] = median_shoreline_lines['SSS'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
    median_shoreline_lines = median_shoreline_lines[column_order]
    median_shoreline_lines.to_file(tier_3_qc, layer='07_median_shoreline_lines')
    median_shoreline_lines = None

    ### q3_shoreline_points
    q3_shoreline_points = gpd.read_file(tier_3, layer='08_q3_shoreline_points')
    q3_shoreline_points['transect_id'] = q3_shoreline_points['transect_id'].astype(str)
    q3_shoreline_points['cross_distance_q3'] = q3_shoreline_points['cross_distance_q3'].astype(float)
    if 'geometry_from_centroids' in q3_shoreline_points.columns:
        q3_shoreline_points = q3_shoreline_points.drop(columns=['geometry_from_centroids'])
    q3_shoreline_points['G'] = q3_shoreline_points['G'].astype(str)
    q3_shoreline_points['C'] = q3_shoreline_points['C'].astype(str)
    q3_shoreline_points['RR'] = q3_shoreline_points['RR'].astype(str)
    q3_shoreline_points['SSS'] = q3_shoreline_points['SSS'].astype(str)
    q3_shoreline_points['V'] = ['0'] * len(q3_shoreline_points)
    q3_shoreline_points['LLLLLL'] = [l[-6:] for l in q3_shoreline_points['transect_id']]
    q3_shoreline_points['LLLLLL'] = (q3_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',

                    'cross_distance_q3',
                    'geometry']
    q3_shoreline_points = q3_shoreline_points[column_order]
    q3_shoreline_points.to_file(tier_3_qc, layer='08_q3_shoreline_points')
    q3_shoreline_points = None

    ### q3_shoreline_lines
    q3_shoreline_lines = gpd.read_file(tier_3, layer='09_q3_shoreline_lines')
    q3_shoreline_lines['G'] = q3_shoreline_lines['G'].astype(str)
    q3_shoreline_lines['C'] = q3_shoreline_lines['C'].astype(str)
    q3_shoreline_lines['RR'] = q3_shoreline_lines['RR'].astype(str)
    q3_shoreline_lines['SSS'] = q3_shoreline_lines['SSS'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
    q3_shoreline_lines = q3_shoreline_lines[column_order]
    q3_shoreline_lines.to_file(tier_3_qc, layer='09_q3_shoreline_lines')
    q3_shoreline_lines = None

    ### max_shoreline_points
    max_shoreline_points = gpd.read_file(tier_3, layer='10_max_shoreline_points')
    max_shoreline_points['transect_id'] = max_shoreline_points['transect_id'].astype(str)
    max_shoreline_points['cross_distance_max'] = max_shoreline_points['cross_distance_max'].astype(float)
    if 'geometry_from_centroids' in max_shoreline_points.columns:
        max_shoreline_points = max_shoreline_points.drop(columns=['geometry_from_centroids'])
    max_shoreline_points['G'] = max_shoreline_points['G'].astype(str)
    max_shoreline_points['C'] = max_shoreline_points['C'].astype(str)
    max_shoreline_points['RR'] = max_shoreline_points['RR'].astype(str)
    max_shoreline_points['SSS'] = max_shoreline_points['SSS'].astype(str)
    max_shoreline_points['V'] = ['0']*len(max_shoreline_points)
    max_shoreline_points['LLLLLL'] = [l[-6:] for l in max_shoreline_points['transect_id']]
    max_shoreline_points['LLLLLL'] = (max_shoreline_points['LLLLLL'].astype(int).astype(str).str.zfill(6))
    column_order = ['G', 'C', 'RR', 'SSS', 'V', 'LLLLLL', 'transect_id',
                    'cross_distance_max',
                    'geometry']
    max_shoreline_points = max_shoreline_points[column_order]
    max_shoreline_points.to_file(tier_3_qc, layer='10_max_shoreline_points')
    max_shoreline_points = None

    ### max_shoreline_lines
    max_shoreline_lines = gpd.read_file(tier_3, layer='11_max_shoreline_lines')
    max_shoreline_lines['G'] = max_shoreline_lines['G'].astype(str)
    max_shoreline_lines['C'] = max_shoreline_lines['C'].astype(str)
    max_shoreline_lines['RR'] = max_shoreline_lines['RR'].astype(str)
    max_shoreline_lines['SSS'] = max_shoreline_lines['SSS'].astype(str)
    column_order = ['G', 'C', 'RR', 'SSS', 'geometry']
    max_shoreline_lines = max_shoreline_lines[column_order]
    max_shoreline_lines.to_file(tier_3_qc, layer='11_max_shoreline_lines')
    max_shoreline_lines = None





# folder = os.path.join('/', 'mnt', 'f', 'SDSDataService_c')
# qc_folder = os.path.join('/', 'mnt', 'f', 'SDSDataService_c_qc1')
# index = os.path.join(folder, '14_index.gpkg')
# tier_0 = os.path.join(folder, '14_tier_0.gpkg')
# tier_1 = os.path.join(folder, '14_tier_1.gpkg')
# tier_2 = os.path.join(folder, '14_tier_2.gpkg')
# tier_3 = os.path.join(folder, '14_tier_3.gpkg')
# index_qc = os.path.join(qc_folder, '14_index.gpkg')
# tier_0_qc = os.path.join(qc_folder, '14_tier_0.gpkg')
# tier_1_qc = os.path.join(qc_folder, '14_tier_1.gpkg')
# tier_2_qc = os.path.join(qc_folder, '14_tier_2.gpkg')
# tier_3_qc = os.path.join(qc_folder, '14_tier_3.gpkg')
# qc_index_columns(index, index_qc)
# qc_tier_0_columns(tier_0, tier_0_qc)
# qc_tier_1_columns(tier_1, tier_1_qc)
# qc_tier_2_columns(tier_2, tier_2_qc)
# qc_tier_3_columns(tier_3, tier_3_qc)
