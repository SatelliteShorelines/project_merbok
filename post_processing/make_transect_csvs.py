
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from typing import Union, List, Tuple
import geopandas as gpd
import datetime
import shapely
import traceback
import tqdm
import itertools
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# USGS-style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "0.85",
    "grid.linestyle": "--"
})

def sign(num):
    if num>0:
        return 'positive'
    if num<0:
        return 'negative'
    if num==0:
        return 'no change'
    
def snr_median(t, y, eps=1e-12):
    t=t.dropna()
    y=y.dropna()
    y = np.asarray(y)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return np.abs(med) / (mad + eps)


def shannon_entropy(t, y, bins=30):
    # drop NaNs
    y = np.asarray(y.dropna())

    # empty or all-NaN
    if y.size == 0 or np.all(np.isnan(y)):
        return np.nan

    # histogram
    hist, _ = np.histogram(y, bins=bins, density=False)

    # empirical probabilities
    p = hist / hist.sum()
    p = p[p > 0]  # remove zero-probability bins

    # Shannon entropy
    return -np.sum(p * np.log(p))

def split_list_at_none(lst):
    # Initialize variables
    result = []
    temp = []

    # Iterate through the list
    for item in lst:
        if item is None:
            # Append the current sublist to the result and reset temp
            result.append(temp)
            temp = []
        else:
            # Add item to the current sublist
            temp.append(item)

    # Append the last sublist if not empty
    if temp:
        result.append(temp)

    return result

def remove_nones(my_list):
    new_list = [x for x in my_list if x is not None]
    return new_list

def arr_to_LineString(coords):
    """
    Makes a line feature from a list of xy tuples
    inputs: coords
    outputs: line
    """
    points = [None]*len(coords)
    i=0
    for xy in coords:
        points[i] = shapely.geometry.Point(xy)
        i=i+1
    line = shapely.geometry.LineString(points)
    return line

def simplify_lines(shorelines_path, tolerance=1):
    """
    Uses shapely simplify function to smooth out the extracted shorelines
    inputs:
    shapefile: path to extracted shorelines
    tolerance (optional): simplification tolerance
    outputs:
    save_path: path to simplified shorelines
    """

    save_path = os.path.splitext(shorelines_path)[0]+'_simplify'+str(tolerance)+'.geojson'
    lines = gpd.read_file(shorelines_path)
    lines['geometry'] = lines['geometry'].simplify(tolerance)
    lines.to_file(save_path)
    return save_path

def LineString_to_arr(line):
    """
    Makes an array from linestring
    inputs: line
    outputs: array of xy tuples
    """
    listarray = []
    for pp in line.coords:
        listarray.append(pp)
    nparray = np.array(listarray)
    return nparray

def chaikins_corner_cutting(coords, refinements=5):
    """
    Smooths out lines or polygons with Chaikin's method
    """
    i=0
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25
        i=i+1
    return coords

def smooth_lines(shorelines,refinements=5):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM
    saves output with '_smooth' appended to original filename in same directory

    inputs:
    shorelines (str): path to extracted shorelines in UTM
    refinements (int): number of refinemnets for Chaikin's smoothing algorithm
    outputs:
    save_path (str): path of output file in UTM
    """
    dirname = os.path.dirname(shorelines)
    save_path = os.path.join(dirname,os.path.splitext(os.path.basename(shorelines))[0]+'_smooth.geojson')
    lines = gpd.read_file(shorelines)
    new_lines = lines.copy()
    for i in range(len(lines)):
        line = lines.iloc[i]
        coords = LineString_to_arr(line.geometry)
        refined = chaikins_corner_cutting(coords, refinements=refinements)
        refined_geom = arr_to_LineString(refined)
        new_lines['geometry'][i] = refined_geom
    new_lines.to_file(save_path)
    return save_path

def wgs84_to_utm_file(geojson_file):
    """
    Converts wgs84 to UTM
    inputs:
    geojson_file (path): path to a geojson in wgs84
    outputs:
    geojson_file_utm (path): path to a geojson in utm
    """

    geojson_file_utm = os.path.splitext(geojson_file)[0]+'_utm.geojson'

    gdf_wgs84 = gpd.read_file(geojson_file)
    utm_crs = gdf_wgs84.estimate_utm_crs()

    gdf_utm = gdf_wgs84.to_crs(utm_crs)
    gdf_utm.to_file(geojson_file_utm)
    return geojson_file_utm

def utm_to_wgs84_df(geo_df):
    """
    Converts utm to wgs84
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in utm
    outputs:
    geo_df_wgs84 (geopandas  dataframe): a geopandas dataframe in wgs84
    """
    wgs84_crs = 'epsg:4326'
    gdf_wgs84 = geo_df.to_crs(wgs84_crs)
    return gdf_wgs84
def transect_timeseries_to_wgs84(transect_timeseries_merged_path,
                                 transects_path,
                                 savename_lines,
                                 savename_points):
    """
    Takes merged transect timeseries path and outputs new shoreline lines and points files
    inputs:
    transect_timeseries_merged_path (str): path to the transect_timeseries_merged.csv
    config_gdf_path (str): path to the the config_gdf as geojson
    savename_lines (str): basename of the output lines ('..._lines.geojson')
    savename_points (str): basename of the output points ('...._points.geojson')
    """
    ##Load in data, make some new paths
    timeseries_data = pd.read_csv(transect_timeseries_merged_path).dropna(subset='cross_distance').reset_index(drop=True)
    timeseries_data = timeseries_data.sort_values('transect_id', ascending=True).reset_index(drop=True)
    timeseries_data['dates'] = pd.to_datetime(timeseries_data['dates'], utc=True)
    timeseries_data['transect_id'] = timeseries_data['transect_id'].astype(int)
    transects = gpd.read_file(transects_path)
    transects['transect_id'] = transects['transect_id'].astype(int)

    transects = transects.sort_values('transect_id', ascending=True).reset_index(drop=True)

    ##save paths
    new_gdf_shorelines_wgs84_path = os.path.join(os.path.dirname(transect_timeseries_merged_path), savename_lines)    
    points_wgs84_path = os.path.join(os.path.dirname(transect_timeseries_merged_path), savename_points)
    
    ##Gonna do this in UTM to keep the math simple...problems when we get to longer distances (10s of km)
    org_crs = transects.crs
    utm_crs = transects.estimate_utm_crs()
    transects_utm = transects.to_crs(utm_crs)

    ##need some placeholders
    shore_x_vals = [None]*len(timeseries_data)
    shore_y_vals = [None]*len(timeseries_data)
    timeseries_data['shore_x'] = shore_x_vals
    timeseries_data['shore_y'] = shore_y_vals
    
    ##make an empty gdf to hold points
    size = len(timeseries_data)
    transect_ids = [None]*size
    dates = [None]*size
    points = [None]*size
    ci = [None]*size
    cross_distances = [None]*size
    cross_distances_rgb = [None]*size
    cross_distances_nir = [None]*size
    cross_distances_swir = [None]*size
    avg_suitabilities = [None]*size
    satnames = [None]*size
    avg_slopes = [None]*size
    tides = [None]*size
    points_gdf_utm = gpd.GeoDataFrame({'geometry':points,
                                       'cross_distance':cross_distances,
                                       'cross_distance_rgb':cross_distances_rgb,
                                       'cross_distance_nir':cross_distances_nir,
                                       'cross_distance_swir':cross_distances_swir,
                                       'dates':dates,
                                       'transect_id':transect_ids,
                                       'avg_suitability':avg_suitabilities,
                                       'satname':satnames,
                                       'avg_slope':avg_slopes,
                                       'tide':tides,
                                       'ci':ci},
                                      crs=utm_crs)
    
    ##loop over all transects
    for i in range(len(transects_utm)):
        transect = transects_utm.iloc[i]
        transect_id = transect['transect_id']
        first = transect.geometry.coords[0]
        last = transect.geometry.coords[1]
        idx = timeseries_data.index[timeseries_data['transect_id'] == transect_id].tolist()
        ##in case there is a transect in the config_gdf that doesn't have any intersections
        ##skip that transect
        if np.any(idx):
            timeseries_data_filter = timeseries_data.iloc[idx]
        else:
            continue

        idxes = timeseries_data_filter.index
        distances = timeseries_data_filter['cross_distance']
        angle = np.arctan2(last[1] - first[1], last[0] - first[0])

        shore_x_utm = first[0]+distances*np.cos(angle)
        shore_y_utm = first[1]+distances*np.sin(angle)
        points_utm = [shapely.Point(xy) for xy in zip(shore_x_utm, shore_y_utm)]

        #conversion from utm to wgs84, put them in the transect_timeseries csv and utm gdf
        dummy_gdf_utm = gpd.GeoDataFrame({'geometry':points_utm},
                                         crs=utm_crs)
        dummy_gdf_wgs84 = dummy_gdf_utm.to_crs(org_crs)

        points_wgs84 = [shapely.get_coordinates(p) for p in dummy_gdf_wgs84.geometry]
        points_wgs84 = np.array(points_wgs84)
        points_wgs84 = points_wgs84.reshape(len(points_wgs84),2)
        x_wgs84 = points_wgs84[:,0]
        y_wgs84 = points_wgs84[:,1]
        timeseries_data.loc[idxes,'shore_x'] = x_wgs84
        timeseries_data.loc[idxes,'shore_y'] = y_wgs84
        cis = timeseries_data['ci'].loc[idxes]
        dates = timeseries_data['dates'].loc[idxes]
        cross_distance_rgb_vals = timeseries_data['cross_distance_rgb'].loc[idxes]
        cross_distance_swir_vals = timeseries_data['cross_distance_swir'].loc[idxes]
        cross_distance_nir_vals = timeseries_data['cross_distance_nir'].loc[idxes]
        avg_suitability_vals = timeseries_data['avg_suitability'].loc[idxes]
        satname_vals = timeseries_data['satname'].loc[idxes]
        avg_slope_vals = timeseries_data['avg_slope'].loc[idxes]
        tide_vals = timeseries_data['tide'].loc[idxes]

        points_gdf_utm.loc[idxes,'geometry'] = points_utm
        points_gdf_utm.loc[idxes,'dates'] = dates
        points_gdf_utm.loc[idxes,'transect_id'] = [transect_id]*len(dates)
        points_gdf_utm.loc[idxes,'ci'] = cis
        points_gdf_utm.loc[idxes,'cross_distance'] = distances
        points_gdf_utm.loc[idxes, 'cross_distance_rgb'] = cross_distance_rgb_vals
        points_gdf_utm.loc[idxes, 'cross_distance_nir'] = cross_distance_nir_vals
        points_gdf_utm.loc[idxes, 'cross_distance_swir'] = cross_distance_swir_vals
        points_gdf_utm.loc[idxes, 'avg_suitability'] = avg_suitability_vals
        points_gdf_utm.loc[idxes, 'satname'] = satname_vals
        points_gdf_utm.loc[idxes, 'avg_slope'] = avg_slope_vals
        points_gdf_utm.loc[idxes, 'tide'] = tide_vals


    ##get points as wgs84 gdf
    points_gdf_wgs84 = points_gdf_utm.to_crs(org_crs)
    points_gdf_wgs84 = points_gdf_wgs84.mask(points_gdf_wgs84.eq('None')).dropna(subset=['geometry']).reset_index(drop=True)

    points_gdf_wgs84['dates'] = pd.to_datetime(points_gdf_wgs84['dates'], utc=True)
    points_gdf_wgs84['year'] = points_gdf_wgs84['dates'].dt.year
    points_gdf_wgs84['ci'] = points_gdf_wgs84['ci'].astype(float)

    ##keep this commented out
    # ##Need to loop over unique dates to make shoreline gdf from points
    # dates = points_gdf_wgs84['dates']
    # new_dates = []
    # new_lines = []
    # for i in range(len(dates)):
    #     date = dates.iloc[i]
    #     points_filter = points_gdf_wgs84[points_gdf_wgs84['dates']==date].reset_index(drop=True)
    #     if len(points_filter)<2:
    #         continue
    #     ps = [None]*len(transects_utm)
    #     ps_ids = [None]*len(transects_utm)
    #     for k in range(len(transects_utm)):
    #         transect = transects.iloc[k]
    #         transect_id = transect['transect_id']
    #         points_filter_filter = points_filter[points_filter['id']==transect_id].reset_index(drop=True)
    #         if len(points_filter_filter)>0:
    #             ps[k] = points_filter_filter['geometry'].iloc[0]
    #             ps_ids[k] = points_filter_filter['id'].iloc[0]
    #     split_lists = split_list_at_none(ps)
    #     for spl in split_lists:
    #         if len(spl)>1:
    #             new_lines.append(shapely.LineString(spl))
    #             new_dates.append(date)
    #         elif len(spl) == 1:
    #             # create a new point by adding a small amount to the x value and y value of the point
    #             point = spl[0]
    #             x = point.x + 0.00001
    #             y = point.y + 0.00001
    #             new_point = (x,y)
    #             new_lines.append(shapely.LineString([point, new_point]))
    #             new_dates.append(date)
    # new_gdf_shorelines_wgs84 = gpd.GeoDataFrame({'dates':new_dates,
    #                                              'geometry':new_lines},
    #                                             crs=org_crs)
    # new_gdf_shorelines_wgs84['dates'] = pd.to_datetime(new_gdf_shorelines_wgs84['dates'], utc=True)
    # new_gdf_shorelines_wgs84['year'] = new_gdf_shorelines_wgs84['dates'].dt.year
    # new_gdf_shorelines_wgs84 = new_gdf_shorelines_wgs84.mask(new_gdf_shorelines_wgs84.eq('None')).dropna().reset_index(drop=True)

    ##convert to utm, save wgs84 and utm geojsons
    #new_gdf_shorelines_wgs84.to_file(new_gdf_shorelines_wgs84_path)
    points_gdf_wgs84.to_file(points_wgs84_path)

def transect_timeseries_to_wgs84_resampled(transect_timeseries_merged_path,
                                 transects_path,
                                 savename_lines,
                                 savename_points):
    """
    Takes merged transect timeseries path and outputs new shoreline lines and points files
    inputs:
    transect_timeseries_merged_path (str): path to the transect_timeseries_merged.csv
    config_gdf_path (str): path to the the config_gdf as geojson
    savename_lines (str): basename of the output lines ('..._lines.geojson')
    savename_points (str): basename of the output points ('...._points.geojson')
    """
    ##Load in data, make some new paths
    timeseries_data = pd.read_csv(transect_timeseries_merged_path).dropna().reset_index(drop=True)
    timeseries_data = timeseries_data.sort_values('transect_id', ascending=True).reset_index(drop=True)
    timeseries_data['dates'] = pd.to_datetime(timeseries_data['dates'], utc=True)
    timeseries_data['transect_id'] = timeseries_data['transect_id'].astype(int)
    transects = gpd.read_file(transects_path)
    transects['transect_id'] = transects['transect_id'].astype(int)

    transects = transects.sort_values('transect_id', ascending=True).reset_index(drop=True)

    ##save paths
    new_gdf_shorelines_wgs84_path = os.path.join(os.path.dirname(transect_timeseries_merged_path), savename_lines)    
    points_wgs84_path = os.path.join(os.path.dirname(transect_timeseries_merged_path), savename_points)
    
    ##Gonna do this in UTM to keep the math simple...problems when we get to longer distances (10s of km)
    org_crs = transects.crs
    utm_crs = transects.estimate_utm_crs()
    transects_utm = transects.to_crs(utm_crs)

    ##need some placeholders
    shore_x_vals = [None]*len(timeseries_data)
    shore_y_vals = [None]*len(timeseries_data)
    timeseries_data['shore_x'] = shore_x_vals
    timeseries_data['shore_y'] = shore_y_vals
    
    ##make an empty gdf to hold points
    size = len(timeseries_data)
    transect_ids = [None]*size
    dates = [None]*size
    points = [None]*size
    ci = [None]*size
    cross_distances = [None]*size
    points_gdf_utm = gpd.GeoDataFrame({'geometry':points,
                                       'cross_distance':cross_distances,
                                       'dates':dates,
                                       'transect_id':transect_ids,
                                       'ci':ci},
                                      crs=utm_crs)
    
    ##loop over all transects
    for i in range(len(transects_utm)):
        transect = transects_utm.iloc[i]
        transect_id = transect['transect_id']
        first = transect.geometry.coords[0]
        last = transect.geometry.coords[1]
        idx = timeseries_data.index[timeseries_data['transect_id'] == transect_id].tolist()
        ##in case there is a transect in the config_gdf that doesn't have any intersections
        ##skip that transect
        if np.any(idx):
            timeseries_data_filter = timeseries_data.iloc[idx]
        else:
            continue

        idxes = timeseries_data_filter.index
        distances = timeseries_data_filter['cross_distance']
        angle = np.arctan2(last[1] - first[1], last[0] - first[0])

        shore_x_utm = first[0]+distances*np.cos(angle)
        shore_y_utm = first[1]+distances*np.sin(angle)
        points_utm = [shapely.Point(xy) for xy in zip(shore_x_utm, shore_y_utm)]

        #conversion from utm to wgs84, put them in the transect_timeseries csv and utm gdf
        dummy_gdf_utm = gpd.GeoDataFrame({'geometry':points_utm},
                                         crs=utm_crs)
        dummy_gdf_wgs84 = dummy_gdf_utm.to_crs(org_crs)

        points_wgs84 = [shapely.get_coordinates(p) for p in dummy_gdf_wgs84.geometry]
        points_wgs84 = np.array(points_wgs84)
        points_wgs84 = points_wgs84.reshape(len(points_wgs84),2)
        x_wgs84 = points_wgs84[:,0]
        y_wgs84 = points_wgs84[:,1]
        timeseries_data.loc[idxes,'shore_x'] = x_wgs84
        timeseries_data.loc[idxes,'shore_y'] = y_wgs84
        cis = timeseries_data['ci'].loc[idxes]
        dates = timeseries_data['dates'].loc[idxes]
        points_gdf_utm.loc[idxes,'geometry'] = points_utm
        points_gdf_utm.loc[idxes,'dates'] = dates
        points_gdf_utm.loc[idxes,'transect_id'] = [transect_id]*len(dates)
        points_gdf_utm.loc[idxes,'ci'] = cis
        points_gdf_utm.loc[idxes,'cross_distance'] = distances
        
    ##get points as wgs84 gdf
    points_gdf_wgs84 = points_gdf_utm.to_crs(org_crs)
    points_gdf_wgs84 = points_gdf_wgs84.mask(points_gdf_wgs84.eq('None')).dropna().reset_index(drop=True)

    points_gdf_wgs84['dates'] = pd.to_datetime(points_gdf_wgs84['dates'], utc=True)
    points_gdf_wgs84['year'] = points_gdf_wgs84['dates'].dt.year
    points_gdf_wgs84['ci'] = points_gdf_wgs84['ci'].astype(float)
    # ##Need to loop over unique dates to make shoreline gdf from points
    # dates = points_gdf_wgs84['dates']
    # new_dates = []
    # new_lines = []
    # for i in range(len(dates)):
    #     date = dates.iloc[i]
    #     points_filter = points_gdf_wgs84[points_gdf_wgs84['dates']==date].reset_index(drop=True)
    #     if len(points_filter)<2:
    #         continue
    #     ps = [None]*len(transects_utm)
    #     ps_ids = [None]*len(transects_utm)
    #     for k in range(len(transects_utm)):
    #         transect = transects.iloc[k]
    #         transect_id = transect['transect_id']
    #         points_filter_filter = points_filter[points_filter['id']==transect_id].reset_index(drop=True)
    #         if len(points_filter_filter)>0:
    #             ps[k] = points_filter_filter['geometry'].iloc[0]
    #             ps_ids[k] = points_filter_filter['id'].iloc[0]
    #     split_lists = split_list_at_none(ps)
    #     for spl in split_lists:
    #         if len(spl)>1:
    #             new_lines.append(shapely.LineString(spl))
    #             new_dates.append(date)
    #         elif len(spl) == 1:
    #             # create a new point by adding a small amount to the x value and y value of the point
    #             point = spl[0]
    #             x = point.x + 0.00001
    #             y = point.y + 0.00001
    #             new_point = (x,y)
    #             new_lines.append(shapely.LineString([point, new_point]))
    #             new_dates.append(date)
    # new_gdf_shorelines_wgs84 = gpd.GeoDataFrame({'dates':new_dates,
    #                                              'geometry':new_lines},
    #                                             crs=org_crs)
    # new_gdf_shorelines_wgs84['dates'] = pd.to_datetime(new_gdf_shorelines_wgs84['dates'], utc=True)
    # new_gdf_shorelines_wgs84['year'] = new_gdf_shorelines_wgs84['dates'].dt.year
    # new_gdf_shorelines_wgs84 = new_gdf_shorelines_wgs84.mask(new_gdf_shorelines_wgs84.eq('None')).dropna().reset_index(drop=True)

    ##convert to utm, save wgs84 and utm geojsons
    #new_gdf_shorelines_wgs84.to_file(new_gdf_shorelines_wgs84_path)
    points_gdf_wgs84.to_file(points_wgs84_path)

def resample_timeseries(df, timedelta):
    """
    Resamples the timeseries according to the provided timedelta, mean
    """
    old_df = df
    old_df.index = df['dates']
    new_df = old_df.resample(timedelta).mean()
    new_df['ci'] = old_df['cross_distance'].resample(timedelta).sem()*1.96
    return new_df

def fill_nans(df):
    """
    Fills nans in timeseries with linear interpolation
    """
    old_df = df
    old_df.index = df['dates']
    new_df = old_df.interpolate(method='linear', limit=None, limit_direction='both')
    return new_df

def resample_and_reproject_shorelines(transect_timeseries_path,
         transects_path,
         output_folder,
         transect_spacing,
         timedelta='365D'):
    """
    Performs timeseries and spatial series analysis cookbook on each
    transect in the transect_time_series matrix from CoastSeg
    inputs:
    transect_timeseries_path (str): path to the transect_time_series.csv
    config_gdf_path (str): path to the config_gdf.geojson
    output_folder (str): path to save outputs to

    """

    ##Load in data
    timeseries_data = pd.read_csv(transect_timeseries_path)
    timeseries_data['dates'] = pd.to_datetime(timeseries_data['dates'],
                                              utc=True,
                                              )
    transects = gpd.read_file(transects_path)

    ##Loop over transects (space)
    transect_ids = [None]*len(transects)
    timeseries_dfs = [None]*len(transects)
    timedeltas = [None]*len(transects)
    for i in range(len(transects)):
        transect_id = str(transects['transect_id'].iloc[i])
        dates = timeseries_data['dates']
        try:
            select_timeseries = np.array(timeseries_data[transect_id])
        except:
            continue
        
        transect_ids[i] = transect_id
        
        ##Timeseries processing
        data = pd.DataFrame({'dates':dates,
                             'cross_distance':select_timeseries})
        filtered_data = resample_timeseries(filtered_data, timedelta)
        filtered_data = fill_nans(filtered_data)
        output_df = output_df.rename(columns = {'cross_distance':transect_id})
        timeseries_dfs[i] = output_df
        timedeltas[i] = timedelta

    ##Remove Nones in case there were transects in config_gdf with no timeseries data
    transect_ids = [ele for ele in transect_ids if ele is not None]
    timeseries_dfs = [ele for ele in timeseries_dfs if ele is not None]
    timedeltas = [ele for ele in timedeltas if ele is not None]

    ##Make new matrix 
    new_matrix = pd.concat(timeseries_dfs)

    ##Stacked csv
    stacked = new_matrix.melt(id_vars=['date'],
                              var_name='transect_id',
                              value_name='cross_distance')


class HampelFilter:
    """
    HampelFilter class for providing additional functionality such as checking the upper/lower boundaries for paramter tuning.
    """

    def __init__(self, window_size: int = 5, n_sigma: int = 3, c: float = 1.4826):
        """ Initialize HampelFilter object. Rolling median and rolling sigma are calculated here.

        :param window_size: length of the sliding window, a positive odd integer.
            (`window_size` - 1) // 2 adjacent samples on each side of the current sample are used for calculating median.
        :param n_sigma: threshold for outlier detection, a real scalar greater than or equal to 0. default is 3.
        :param c: consistency constant. default is 1.4826, supposing the given timeseries values are normally distributed.
        :return: the outlier indices
        """

        if not (type(window_size) == int and window_size % 2 == 1 and window_size > 0):
            raise ValueError("window_size must be a positive odd integer greater than 0.")

        if not (type(n_sigma) == int and n_sigma >= 0):
            raise ValueError("n_sigma must be a positive integer greater than or equal to 0.")

        self.window_size = window_size
        self.n_sigma = n_sigma
        self.c = c

        # These values will be set after executing apply()
        self._outlier_indices = None
        self._upper_bound = None
        self._lower_bound = None

    def apply(self, x: Union[List, pd.Series, np.ndarray]):
        """ Return the indices of the detected outliers by the filter.

        :param x: timeseries values of type List, numpy.ndarray, or pandas.Series

        :return: indices of the outliers
        """
        # Check given arguments
        if not (type(x) == list or type(x) == np.ndarray or type(x) == pd.Series):
            raise ValueError("x must be either of type List, numpy.ndarray, or pandas.Series.")

        # calculate rolling_median and rolling_sigma using the given parameters.
        x_window_view = sliding_window_view(np.array(x), window_shape=self.window_size)
        rolling_median = np.median(x_window_view, axis=1)
        rolling_sigma = self.c * np.median(np.abs(x_window_view - rolling_median.reshape(-1, 1)), axis=1)

        self._upper_bound = rolling_median + (self.n_sigma * rolling_sigma)
        self._lower_bound = rolling_median - (self.n_sigma * rolling_sigma)

        outlier_indices = np.nonzero(
            np.abs(np.array(x)[(self.window_size - 1) // 2:-(self.window_size - 1) // 2] - rolling_median)
            >= (self.n_sigma * rolling_sigma)
        )[0] + (self.window_size - 1) // 2

        if type(x) == list:
            # When x is of List[float | int], return the indices in List.
            self._outlier_indices = list(outlier_indices)
        elif type(x) == pd.Series:
            # When x is of pd.Series, return the indices of the Series object.
            self._outlier_indices = x.index[outlier_indices]
        else:
            self._outlier_indices = outlier_indices

        return self

    def get_indices(self) -> Union[List, pd.Series, np.ndarray]:
        """
        """
        if self._outlier_indices is None:
            raise AttributeError("Outlier indices have not been set. Execute hampel_filter_object.apply(x) first.")
        return self._outlier_indices

    def get_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the upper and lower boundaries of the filter. Note that the values are `window_size - 1` shorter than the given timeseries x.

        :return: a tuple of the lower bound values and the upper bound values. i.e. (lower_bound_values, upper_bound_values)
        """
        if self._upper_bound is None or self._lower_bound is None:
            raise AttributeError("Boundary values have not been set. Execute hampel_filter_object.apply() first.")

        return self._lower_bound, self._upper_bound


def hampel_filter(x: Union[List, pd.Series, np.ndarray], window_size: int = 5, n_sigma: int = 3, c: float = 1.4826) \
        -> Union[List, pd.Series, np.ndarray]:
    """ Outlier detection using the Hampel identifier

    :param x: timeseries values of type List, numpy.ndarray, or pandas.Series
    :param window_size: length of the sliding window, a positive odd integer.
        (`window_size` - 1) // 2 adjacent samples on each side of the current sample are used for calculating median.
    :param n_sigma: threshold for outlier detection, a real scalar greater than or equal to 0. default is 3.
    :param c: consistency constant. default is 1.4826, supposing the given timeseries values are normally distributed.
    :return: the outlier indices
    """

    return HampelFilter(window_size=window_size, n_sigma=n_sigma, c=c).apply(x).get_indices()



def hampel_filter_df(df, hampel_window=3, hampel_sigma=2):
    """
    Applies a Hampel Filter

    inputs:
    df (pandas dataframe): must have column 'cross_distance' and 'dates'
    hampel_window (int): odd integer that is less than len(df)
    hampel_sigma (float): sigma for hampel filter
    
    outputs:
    df (pandas dataframe): filtered dataframe
    """
  
    vals = df['median_pos'].values
    outlier_idxes = hampel_filter(vals, hampel_window, hampel_sigma)
    vals[outlier_idxes] = np.nan
    df['median_pos'] = vals
    return df

def hampel_filter_loop(df, hampel_window=3, hampel_sigma=2):
    """
    Applies a Hampel Filter recursively, that is over and over again until no
    more outliers are computed and removed.

    inputs:
    df (pandas dataframe): must have column 'cross_distance' and 'dates'
    hampel_window (int): odd integer that is less than len(df)
    hampel_sigma (float): sigma for hampel filter

    outputs:
    df (pandas dataframe): filtered dataframe
    """

    df['date'] = df.index
    num_nans = df['median_pos'].isna().sum()
    new_num_nans = None
    h=0
    while (num_nans != new_num_nans) and (len(df)>hampel_window):
        num_nans = df['median_pos'].isna().sum()
        df = hampel_filter_df(df, hampel_window=hampel_window, hampel_sigma=hampel_sigma)
        new_num_nans = df['median_pos'].isna().sum()
        df = df.dropna(subset=['median_pos'])
        h=h+1
    return df

def change_filter(df, q=0.75):
    """
    Applies a filter on shoreline change
    
    inputs:
    df (pandas dataframe): must have column 'cross_distance' and 'dates'
    q (float): quantile to filter daily change below

    outputs:
    df (pandas dataframe): filtered dataframe
    """
    vals = df['median_pos'].values
    time = df['dates'] - df['dates'].iloc[0]
    time = time.dt.days
    change_y = np.abs(np.diff(vals))
    change_t = np.diff(time)
    dy_dt = change_y/change_t
    dy_dt = np.concatenate([[0],dy_dt])
    max_val = np.nanquantile(dy_dt,q)
    outlier_idxes = dy_dt>max_val
    vals[outlier_idxes] = np.nan
    df['median_pos'] = vals
    return df

def change_filter_loop(df, iterations=1, q=0.75):
    """
    Loops the filter on shoreline change
    
    inputs:
    df (pandas dataframe): must have column 'cross_distance' and 'dates'
    iterations (int): number of iterations
    q (float): quantile to filter daily change below

    outputs:
    df (pandas dataframe): filtered dataframe
    """
    
    h=0
    for i in range(iterations):
        df = change_filter(df, q=q)
        df = df.dropna().reset_index(drop=True)
        h=h+1
    return df

def min_max_normalize(arr):
    """
    Normalizes array to minimum and maximum values

    inputs:
    arr (np.ndarray): input numpy array
    
    outputs:
    new_arr (np.ndarray): the output array
    """
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    new_arr = (arr - min_val) / (max_val - min_val)
    return new_arr

############################################################
#  ALL FILTER FUNCTIONS + ALL DEPENDENCIES (SELF-CONTAINED)
############################################################

import numpy as np
import pandas as pd

############################################################
#  BASIC UTILITIES
############################################################

def safe_minmax(x, invert=False):
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax > xmin:
        z = (x - xmin) / (xmax - xmin)
    else:
        z = np.zeros_like(x)
    return 1.0 - z if invert else z


def mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))


def local_slopes(time, vals):
    time = np.asarray(time, dtype=float)
    vals = np.asarray(vals, dtype=float)
    dt = np.diff(time)
    dv = np.diff(vals)
    D = dv / dt
    slopes = np.empty(len(vals), dtype=float)
    slopes[0] = 0.0
    slopes[1:] = D
    return slopes


############################################################
#  LOESS SUITABILITY
############################################################

def loess_suitability_from_dates(df, target_col, date_col,
                                 frac=0.12, robust_iters=3, normalize=True):
    dates = pd.to_datetime(df[date_col], errors='coerce')
    y = df[target_col].to_numpy(float)
    valid = dates.notna().to_numpy() & np.isfinite(y)
    out = pd.Series(index=df.index, dtype=float)

    if valid.sum() == 0:
        out[:] = np.nan
        return out

    x_all = dates.astype('int64').to_numpy(float)
    x_valid = x_all[valid]
    y_valid = y[valid]

    order = np.argsort(x_valid)
    xs = x_valid[order]
    ys = y_valid[order]

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        fitted_sorted = lowess(ys, xs, frac=frac, it=robust_iters, return_sorted=False)
    except Exception:
        n = max(3, int(len(xs) * frac))
        if n % 2 == 0:
            n += 1
        pad = n // 2
        padded = np.pad(ys, pad_width=pad, mode='edge')
        kernel = np.arange(1, pad + 2)
        kernel = np.concatenate([kernel, kernel[-2::-1]])
        kernel = kernel / kernel.sum()
        fitted_sorted = np.convolve(padded, kernel, mode='valid')

    fitted_unsorted = np.empty_like(fitted_sorted)
    fitted_unsorted[order.argsort()] = fitted_sorted

    fitted_full = np.full_like(y, np.nan, dtype=float)
    fitted_full[valid] = fitted_unsorted

    resid = np.abs(y - fitted_full)
    r_valid = resid[valid]

    if normalize and r_valid.size > 0:
        rmin, rmax = np.nanmin(r_valid), np.nanmax(r_valid)
        if np.isfinite(rmin) and np.isfinite(rmax) and rmax > rmin:
            suit_valid = 1.0 - (r_valid - rmin) / (rmax - rmin)
        else:
            suit_valid = np.ones_like(r_valid) * 0.5
    else:
        suit_valid = 1.0 - r_valid

    out[:] = np.nan
    out.loc[valid] = suit_valid
    return out.clip(0.0, 1.0)


############################################################
#  CLUSTER SUITABILITY
############################################################

def cluster_suitability(time, pos, k=20, r_min=1e-3):
    """
    Joint time–position suitability based on local density scale.
    Suitability = 1 / max(k-th nearest-neighbor distance, r_min).
    A minimum radius prevents inflated scores in extremely tight clusters.
    """

    # Normalize inputs
    t = safe_minmax(time)
    p = safe_minmax(pos)

    # Pairwise distances
    dt = np.abs(t[:, None] - t[None, :])
    dp = np.abs(p[:, None] - p[None, :])
    d = np.sqrt(dt**2 + dp**2)

    # Local scale from k-th nearest neighbor
    kth = np.partition(d, k, axis=1)[:, k]

    # Enforce minimum radius
    kth = np.maximum(kth, r_min)

    # Rank-based suitability
    scores = 1.0 / kth

    return scores / scores.max()


############################################################
#  ROC SUITABILITY
############################################################

def roc_suitability(dy_dt, t0=3):
    x = np.asarray(dy_dt, dtype=float)
    med = np.nanmedian(x)
    mad_val = np.nanmedian(np.abs(x - med)) + 1e-8
    z = np.abs(x - med) / (t0 * mad_val)
    return np.maximum(1.0 - z, 0.0)


############################################################
#  HAMPEL SUITABILITY
############################################################

def hampel_score(x, k=7, t0=3):
    x = np.asarray(x, dtype=float)
    n = len(x)
    scores = np.ones(n, float)
    for i in range(n):
        lo = max(i - k, 0)
        hi = min(i + k, n)
        x0 = x[lo:hi]
        med = np.nanmedian(x0)
        mad_val = mad(x0)
        if mad_val == 0:
            continue
        deviation = np.abs(x[i] - med) / (t0 * mad_val)
        scores[i] = max(0.0, 1.0 - deviation)
    return scores


############################################################
#  FILTER FUNCTIONS (CLEAN, MODULAR)
############################################################

def compute_loess_suitability(df):
    return loess_suitability_from_dates(
        df,
        target_col='median_pos',
        date_col='dates',
        frac=0.12,
        robust_iters=3,
        normalize=True
    )


def compute_cluster_suitability_df(df):
    valid = df['dates'].notna() & df['median_pos'].notna()
    time = (df.loc[valid, 'dates'] - df.loc[valid, 'dates'].iloc[0]).dt.days.to_numpy(float)
    pos  = df.loc[valid, 'median_pos'].to_numpy(float)
    df.loc[valid, 'cluster_suitability'] = cluster_suitability(time,pos)
    return df

def compute_ensemble_suitability(df):
    vals_stack = np.vstack([
        df['cross_distance_tidally_corrected_rgb'].to_numpy(),
        df['cross_distance_tidally_corrected_nir'].to_numpy(),
        df['cross_distance_tidally_corrected_swir'].to_numpy()
    ])

    robust_range = np.nanstd(vals_stack, axis=0)
    n_est = np.sum(~np.isnan(vals_stack), axis=0)

    mask = (robust_range == 0) | (n_est <= 1)

    lo, hi = np.nanpercentile(robust_range, [1, 99])
    den = max(hi - lo, 1e-8)

    scaled = 1 - np.clip((robust_range - lo) / den, 0, 1)
    scaled[mask] = np.nan

    return scaled


def compute_median_deviation_suitability(df):
    vals = (
        pd.Series(df['median_pos'])
        .rolling(10, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    Y_med = np.nanmedian(vals)
    M = np.abs(vals - Y_med) / (np.abs(Y_med) + 1e-6)
    return safe_minmax(M, invert=True)


def compute_roc_suitability_df(df):
    time = (df['dates'] - df['dates'].iloc[0]).dt.days.to_numpy(float)
    vals_smooth = (
        pd.Series(df['median_pos'])
        .rolling(10, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    slopes = local_slopes(time, vals_smooth)
    return roc_suitability(slopes, t0=3)


def compute_hampel_suitability(df):
    return hampel_score(df['median_pos'].to_numpy())


def apply_all_suitability_filters_for_transect(df):
    df['loess_suitability'] = compute_loess_suitability(df)
    df = compute_cluster_suitability_df(df)
    df['ensemble_suitability'] = compute_ensemble_suitability(df)
    df['median_deviation_suitability'] = compute_median_deviation_suitability(df)
    df['roc_suitability'] = compute_roc_suitability_df(df)
    df['hampel_suitability'] = compute_hampel_suitability(df)
    return df

def remap_bimodal_rank_push_array(arr, gamma=3.0):
    a = np.asarray(arr, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    valid = np.isfinite(a)
    if valid.sum() == 0:
        return out
    v = a[valid].copy()
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if vmax > 1.0 or vmin < 0.0:
        if vmax > vmin:
            v = (v - vmin) / (vmax - vmin + 1e-12)
        else:
            v = np.zeros_like(v)
    ranks = np.argsort(np.argsort(v)).astype(float)
    ranks = (ranks + 1.0) / (len(ranks) + 1.0)
    d = np.abs(ranks - 0.5)
    sdist = 2.0 * d
    pushed = 1.0 - (1.0 - sdist) ** gamma
    sign = np.sign(ranks - 0.5)
    mapped = 0.5 + 0.5 * sign * pushed
    out[valid] = np.clip(mapped, 0.0, 1.0)
    return out

def snr(y, eps=1e-8):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    med = np.nanmedian(y)
    mad_val = mad(y)
    if mad_val < eps:
        return 0.0
    return abs(med) / (mad_val + eps)

def retention_objective(rets, r_target=0.4, r_width=0.1):
    lower, upper = r_target - r_width, r_target + r_width
    dev = np.where(
        rets < lower, lower - rets,
        np.where(rets > upper, rets - upper, 0.0)
    )
    obj = 1.0 - dev**2
    return np.clip(obj, 0.0, 1.0)

def best_threshold(a, y, alpha=0.5, beta=0.5, grid_size=401,
                   r_target=0.4, r_width=0.1, lambda_t=0.05, t_min=0.05):
    ts = np.linspace(t_min, 1.0, grid_size)
    snrs, rets = [], []
    for t in ts:
        mask = a >= t
        if mask.sum() == 0:
            snrs.append(0.0)
            rets.append(0.0)
        else:
            snrs.append(snr(y[mask]))
            rets.append(mask.mean())
    snrs = np.asarray(snrs, dtype=float)
    rets = np.asarray(rets, dtype=float)
    rmin, rmax = np.nanmin(rets), np.nanmax(rets)
    if rmax > rmin:
        rets_norm = (rets - rmin) / (rmax - rmin)
    else:
        rets_norm = np.zeros_like(rets)
    ret_obj = retention_objective(rets_norm, r_target=r_target, r_width=r_width)
    t_reg = (ts - ts.min()) / (ts.max() - ts.min())
    scores = alpha * snrs + beta * ret_obj + lambda_t * t_reg

    # S-curve knee on normalized retention
    rets_full = rets_norm
    ts_full = ts
    dret = np.diff(rets_full) / np.diff(ts_full)
    mid_ret = 0.5 * (rets_full[:-1] + rets_full[1:])
    low, high = 0.3, 0.7
    mask_mid = (mid_ret >= low) & (mid_ret <= high)
    idxs = np.arange(len(dret))
    if mask_mid.any():
        cand_idxs = idxs[mask_mid]
        best_local = cand_idxs[np.argmin(dret[mask_mid])]
        idx = int(best_local)
    else:
        idx = int(np.argmin(dret))
    return ts[idx], snrs[idx], rets_norm[idx], scores[idx]

def optimize_with_fixed_weights(df, feature_cols, fixed_w,
                                alpha=0.5, beta=0.5,
                                grid_size=401, r_target=0.3,
                                r_width=0.05, lambda_t=0.01,
                                t_min=0.05,
                                min_filter_floor=0.00):
    fixed_w = np.asarray(fixed_w, dtype=float)
    if fixed_w.shape[0] != len(feature_cols):
        raise ValueError("fixed_w length does not match feature_cols.")
    w_sum = fixed_w.sum()
    if w_sum <= 0:
        raise ValueError("fixed_w must have positive sum.")
    fixed_w = fixed_w / w_sum

    X = df[feature_cols].to_numpy(dtype=float)
    Y = df['median_pos'].to_numpy(dtype=float)
    a = X @ fixed_w

    t, snr_val, ret, score = best_threshold(
        a, Y,
        alpha=alpha, beta=beta,
        grid_size=grid_size,
        r_target=r_target, r_width=r_width,
        lambda_t=lambda_t, t_min=t_min
    )

    df_full = df.copy()
    df_full['avg_suitability'] = a

    # ---------------------------------------------------------
    # Robust filtering (Option 1)
    # A point is only kept if:
    #   avg_suitability >= threshold  AND
    #   min individual suitability >= min_filter_floor
    # ---------------------------------------------------------
    min_suit = df_full[feature_cols].min(axis=1).to_numpy(float)

    robust_mask = (df_full['avg_suitability'] >= t)# & (min_suit >= min_filter_floor)

    df_filtered = df_full[robust_mask].reset_index(drop=True)

    info = {
        'threshold': t,
        'snr': snr_val,
        'retention': ret,
        'score': score,
        'weights': fixed_w,
        'min_filter_floor': min_filter_floor
    }
    return df_full, df_filtered, info




def plot_diagnostics(
    df_full,
    df_filtered,
    feature_cols,
    threshold,
    transect_id,
    output_folder
):
    """
    Produces TWO separate diagnostic figures:

    1. <transect_id>_threshold_diagnostic.png
         - 2×2 grid:
             * Retention vs SNR
             * Threshold vs Retention
             * Threshold vs SNR
             * Threshold vs Score

    2. <transect_id>_suitability_diagnostic.png
         - One subplot per suitability metric + avg_suitability
         - Green = suitable   (avg_suitability >= threshold)
         - Red   = unsuitable (avg_suitability < threshold)
         - One global legend
         - Inset histogram in each subplot
         - Clean time labels
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    from matplotlib import gridspec
    import matplotlib.dates as mdates

    os.makedirs(output_folder, exist_ok=True)

    # ---------------------------------------------------------
    # Shared data
    # ---------------------------------------------------------
    dates = pd.to_datetime(df_full['dates'])
    a = df_full['avg_suitability'].to_numpy(float)

    # Correct filtered mask: threshold on avg_suitability
    mask_suitable = a >= threshold
    mask_unsuitable = ~mask_suitable

    ts = np.linspace(0.0, 1.0, 401)

    # Retention
    rets = np.array([(a >= t).mean() for t in ts])

    # SNR
    def snr_local(vals):
        vals = vals[np.isfinite(vals)]
        if len(vals) < 3:
            return 0.0
        med = np.nanmedian(vals)
        mad_val = np.nanmedian(np.abs(vals - med)) + 1e-8
        return abs(med) / mad_val

    snrs = np.array([snr_local(a[a >= t]) if (a >= t).sum() > 0 else 0.0 for t in ts])

    # Score = retention × SNR
    scores = rets * snrs

    # =========================================================
    # FIGURE 1 — THRESHOLD DIAGNOSTICS (2×2 grid)
    # =========================================================
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Retention vs SNR
    axes[0, 0].plot(rets, snrs, color='black')
    axes[0, 0].set_xlabel("Retention")
    axes[0, 0].set_ylabel("SNR")
    axes[0, 0].set_title("Retention vs SNR")
    axes[0, 0].grid(True)

    # Threshold vs Retention
    axes[0, 1].plot(ts, rets, color='blue')
    axes[0, 1].axvline(threshold, color='red', linestyle='--')
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Retention")
    axes[0, 1].set_title("Threshold vs Retention")
    axes[0, 1].grid(True)

    # Threshold vs SNR
    axes[1, 0].plot(ts, snrs, color='green')
    axes[1, 0].axvline(threshold, color='red', linestyle='--')
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("SNR")
    axes[1, 0].set_title("Threshold vs SNR")
    axes[1, 0].grid(True)

    # Threshold vs Score
    axes[1, 1].plot(ts, scores, color='purple')
    axes[1, 1].axvline(threshold, color='red', linestyle='--')
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Threshold vs Score")
    axes[1, 1].grid(True)

    fig1.suptitle(f"Threshold Diagnostics — Transect {transect_id}", fontsize=14)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_folder, f"{transect_id}_threshold_diagnostic.png"), dpi=150)
    plt.close(fig1)

    # =========================================================
    # FIGURE 2 — SUITABILITY DIAGNOSTICS
    # =========================================================

    # Add avg_suitability as its own subplot
    feature_cols_extended = feature_cols + ["avg_suitability"]

    n = len(feature_cols_extended)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig2 = plt.figure(figsize=(5 * ncols, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows, ncols)

    date_fmt = mdates.DateFormatter("%Y")
    locator = mdates.AutoDateLocator()

    # For global legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='green', linestyle='None', markersize=6, label='suitable'),
        plt.Line2D([0], [0], marker='o', color='red',   linestyle='None', markersize=6, label='unsuitable')
    ]

    for i, col in enumerate(feature_cols_extended):
        ax = fig2.add_subplot(gs[i])

        vals = df_full[col].to_numpy(float)

        # Time series: green = suitable, red = unsuitable
        ax.scatter(dates[mask_suitable],   vals[mask_suitable],   s=22, color='green', alpha=0.9)
        ax.scatter(dates[mask_unsuitable], vals[mask_unsuitable], s=18, color='red',   alpha=0.6)

        ax.set_title(col)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Suitability")
        ax.grid(True)

        # Clean time labels
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(date_fmt)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

        # Inset histogram
        inset_ax = ax.inset_axes([0.65, 0.55, 0.3, 0.4])
        vals_hist = vals[np.isfinite(vals)]
        inset_ax.hist(vals_hist, bins=30, color='steelblue', alpha=0.7)
        inset_ax.set_xlim(0, 1)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title("Dist", fontsize=8)

    # Add one global legend
    fig2.legend(handles=legend_handles, loc='upper right', fontsize=10)

    fig2.suptitle(f"Suitability Diagnostics — Transect {transect_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig2.savefig(os.path.join(output_folder, f"{transect_id}_suitability_diagnostic.png"), dpi=150)
    plt.close(fig2)

def save_csv_per_id(
    input_file_rgb,
    input_file_nir,
    input_file_swir,
    timeseries_type,
    input_file_transects,
    section_string,
    resample_freq='365D'
):
    csvname = "timeseries.csv"
    figname = "timeseries.png"
    id_column_name = "transect_id"

    save_location = os.path.join(os.path.dirname(input_file_rgb), timeseries_type+'_timeseries_csvs')
    try:
        os.mkdir(save_location)
    except:
        pass
    try:
        os.mkdir(save_location_fig)
    except:
        pass

    # read the csv file
    df_rgb = pd.read_csv(input_file_rgb)
    df_nir = pd.read_csv(input_file_nir)
    df_swir = pd.read_csv(input_file_swir)
    if "Unnamed: 0" in df_rgb.columns:
        df_rgb.drop(columns=["Unnamed: 0"], inplace=True)
    if "Unnamed: 0" in df_nir.columns:
        df_nir.drop(columns=["Unnamed: 0"], inplace=True)
    if "Unnamed: 0" in df_swir.columns:
        df_swir.drop(columns=["Unnamed: 0"], inplace=True)

    new_df_rgb = pd.DataFrame()
    new_df_nir = pd.DataFrame()
    new_df_swir = pd.DataFrame()
    unique_ids = sorted(df_rgb[id_column_name].unique())
    dfs = [None]*len(unique_ids)
    dfs2 = [None]*len(unique_ids)
    dfs_resample_filter = [None]*len(unique_ids)
    i=0
    for uid in unique_ids:
        try:
            new_filename = str(uid)+'_'+csvname
            new_figure = str(uid)+'_'+figname
            new_optimum = str(uid)+'_optimization.png'
            new_diagnostic = str(uid)+'_weights_diagnostic.png'
            new_hist = str(uid)+'_suitability_hist.png'
            csv_save = os.path.join(save_location, new_filename)
            fig_save = os.path.join(save_location, new_figure)
            optimum_save = os.path.join(save_location, new_optimum)
            diagnostic_save = os.path.join(save_location, new_diagnostic)
            hist_save = os.path.join(save_location,new_hist)
            new_df_rgb = df_rgb[df_rgb[id_column_name] == uid].reset_index(drop=True)
            new_df_nir = df_nir[df_nir[id_column_name] == uid].reset_index(drop=True)
            new_df_swir = df_swir[df_swir[id_column_name] == uid].reset_index(drop=True)
            new_df_all = pd.merge(new_df_rgb, new_df_nir, on='dates', suffixes=['_rgb', '_nir'])

            new_df_all = pd.merge(new_df_all, new_df_swir, how='outer', on='dates', suffixes=['','_swir'])
            new_df_all = new_df_all.rename(columns={'image_suitability_score':'image_suitability_score_swir',
                                                    'segmentation_suitability_score':'segmentation_suitability_score_swir', 
                                                    'image_suitability_score_rgb':'image_suitability_score',
                                                    'satname':'satname_swir', 
                                                    'transect_id':'transect_id_swir',
                                                    'intersect_x':'intersect_x_swir', 
                                                    'intersect_y':'intersect_y_swir', 
                                                    'cross_distance':'cross_distance_swir', 
                                                    'cross_distance_tidally_corrected':'cross_distance_tidally_corrected_swir',
                                                    'x':'x_swir', 
                                                    'y':'y_swir', 
                                                    'tide':'tide_swir',
                                                    'avg_slope':'avg_slope_swir', 
                                                    }
                                        )
            new_df_all = new_df_all.rename(columns={'image_suitability_score_rgb':'image_suitability_score',
                                                    'segmentation_suitability_score_swir':'segmentaion_suitability_score'
                                                    }
                                                    )
            # ------------------------------
            # Prepare dataframe
            # ------------------------------
            new_df_all['dates'] = pd.to_datetime(new_df_all['dates'], utc=True)
            new_df_all = new_df_all.sort_values(by='dates')

            # Median, min, max across sensors
            vals_stack = np.vstack([
                new_df_all['cross_distance_tidally_corrected_rgb'],
                new_df_all['cross_distance_tidally_corrected_nir'],
                new_df_all['cross_distance_tidally_corrected_swir']
            ])

            new_df_all['median_pos'] = np.nanmedian(vals_stack, axis=0)
            new_df_all['max_pos']    = np.nanquantile(vals_stack, q=0.90, axis=0)
            new_df_all['min_pos']    = np.nanquantile(vals_stack, q=0.10, axis=0)

            ####### Skipping if timeseries is only one point long
            if len(new_df_all) < 2:
                i += 1
                continue

            # Compute all feature-level suitabilities
            new_df_all = apply_all_suitability_filters_for_transect(new_df_all)

            FEATURE_COLS = [
                'ensemble_suitability',
                'loess_suitability',
                'roc_suitability',
                'median_deviation_suitability',
                'cluster_suitability',
                'hampel_suitability',
            ]

            FILTER_MODE = 'all'

            # Weight selection
            if FILTER_MODE != 'all':
                fixed_w = np.zeros(len(FEATURE_COLS))
                idx = FEATURE_COLS.index(FILTER_MODE + '_suitability')
                fixed_w[idx] = 1.0
            else:
                fixed_w = np.array([1.0 / len(FEATURE_COLS)] * len(FEATURE_COLS))

            # Bimodal remap per feature
            for col in FEATURE_COLS:
                new_df_all[col] = remap_bimodal_rank_push_array(
                    new_df_all[col].to_numpy(float), gamma=3.0
                )

            # Threshold optimization
            new_df_all, new_df_all_filter, info = optimize_with_fixed_weights(
                new_df_all,
                FEATURE_COLS,
                fixed_w,
                alpha=0.5,
                beta=0.5,
                grid_size=401,
                r_target=0.3,
                r_width=0.05,
                lambda_t=0.01,
                t_min=0.05,
                min_filter_floor=0.01
            )

            plot_diagnostics(
                df_full=new_df_all,
                df_filtered=new_df_all_filter,
                feature_cols=FEATURE_COLS,
                threshold=info['threshold'],
                transect_id=uid,
                output_folder=save_location
            )


            """Old way"""
            # new_df_all_filter now contains the filtered shoreline positions
            # new_df_all['avg_suitability'] = new_df_all['range_scaled']*w_best[0]+new_df_all['median_normalized']*w_best[1]+new_df_all['dy_dt_scaled']*w_best[2]+new_df_all['image_suitability_score_rgb']*w_best[3]+new_df_all['segmentation_suitability_score_swir']*w_best[4]+new_df_all['kde_value']*w_best[5]
            # optimum_thresh = optimize_threshold(new_df_all, w_best, optimum_save)
            #new_df_all_filter = new_df_all[new_df_all['avg_suitability']>=optimum_thresh].reset_index(drop=True)
            #new_df_all_filter = hampel_filter_loop(new_df_all_filter, hampel_window=15, hampel_sigma=2).reset_index(drop=True)
            #new_df_all_filter = hampel_filter_loop(new_df_all_filter, hampel_window=3, hampel_sigma=2).reset_index(drop=True)

            print("Saving CSV to:", csv_save)
            new_df_all.to_csv(os.path.splitext(csv_save)[0]+'_unfiltered.csv')
            new_df_all_filter.to_csv(csv_save)

            ####Prepping dataframes for resampling
            data = pd.DataFrame({'dates':new_df_all_filter['dates'],
                                 'cross_distance':new_df_all_filter['median_pos'],
                                 'transect_id':[uid]*len(new_df_all_filter)
                                 }
                                )
            data1 =  pd.DataFrame({'dates':new_df_all_filter['dates'],
                                 'cross_distance':new_df_all_filter['median_pos'],
                                 'cross_distance_min':new_df_all_filter['min_pos'],
                                 'cross_distance_max':new_df_all_filter['max_pos'],
                                 'cross_distance_rgb':new_df_all_filter['cross_distance_tidally_corrected_rgb'],
                                 'cross_distance_nir':new_df_all_filter['cross_distance_tidally_corrected_nir'],
                                 'cross_distance_swir':new_df_all_filter['cross_distance_tidally_corrected_swir'],
                                 'ci':new_df_all_filter['max_pos']-new_df_all_filter['min_pos'],
                                 'transect_id':[uid]*len(new_df_all_filter),
                                 'avg_suitability':new_df_all_filter['avg_suitability'],
                                 'satname':new_df_all_filter['satname_rgb'],
                                 'avg_slope':new_df_all_filter['avg_slope_cleaned_rgb'],
                                 'tide':new_df_all_filter['tide_rgb']
                                 }
                                )
            data2 = pd.DataFrame({'dates':new_df_all['dates'],
                                 'cross_distance':new_df_all['median_pos'],
                                 'cross_distance_min':new_df_all['min_pos'],
                                 'cross_distance_max':new_df_all['max_pos'],
                                 'cross_distance_rgb':new_df_all['cross_distance_tidally_corrected_rgb'],
                                 'cross_distance_nir':new_df_all['cross_distance_tidally_corrected_nir'],
                                 'cross_distance_swir':new_df_all['cross_distance_tidally_corrected_swir'],
                                 'ci':new_df_all['max_pos']-new_df_all['min_pos'],
                                 'transect_id':[uid]*len(new_df_all),
                                 'avg_suitability':new_df_all['avg_suitability'],
                                 'satname':new_df_all['satname_rgb'],
                                 'avg_slope':new_df_all['avg_slope_cleaned_rgb'],
                                 'tide':new_df_all['tide_rgb']
                                 }
                                )
            data_resample = resample_timeseries(data,resample_freq)
            data_resample = fill_nans(data_resample)
            data_resample = pd.DataFrame({'dates':data_resample['dates'].dt.round('1s'),
                                          'cross_distance':data_resample['cross_distance'],
                                          'ci':data_resample['ci'],
                                          'transect_id':[uid]*len(data_resample)}).reset_index(drop=True)

            dfs[i] = data1
            dfs2[i] = data2
            dfs_resample_filter[i] = data_resample
            if len(new_df_all_filter)>0:
                try:
                    t1 = new_df_all['dates']
                    y1 = new_df_all['median_pos']

                    t2 = new_df_all_filter['dates']
                    y2 = new_df_all_filter['median_pos']
                    
                    ##snr
                    snr_unfiltered = snr_median(t1, y1, eps=1e-12)
                    snr_filtered = snr_median(t2, y2, eps=1e-12)
                    print('SNR Unfiltered = '+str(np.round(snr_unfiltered,decimals=3)))
                    print('SNR Filtered = '+str(np.round(snr_filtered,decimals=3)))
                    snr_sign = sign(snr_filtered-snr_unfiltered)
                    print(snr_sign)
                    
                    ##entropy
                    entropy_unfiltered = shannon_entropy(t1, y1, bins=30)
                    entropy_filtered = shannon_entropy(t2, y2, bins=30)
                    print('Shannon Entropy Unfiltered = '+str(np.round(entropy_unfiltered,decimals=3)))
                    print('Shannon Entropy Filtered = '+str(np.round(entropy_filtered,decimals=3)))
                    entropy_sign = sign(entropy_filtered-entropy_unfiltered)
                    print(entropy_sign)
                    if snr_sign=='positive' and entropy_sign=='negative':
                        pass
                    else:
                        print('investigate')
                    print('\n')
                    arrow_map = {
                        'positive': '↑',
                        'negative': '↓',
                        'zero': '→'
                    }

                    snr_arrow = arrow_map.get(snr_sign, '')
                    entropy_arrow = arrow_map.get(entropy_sign, '')

                    with plt.rc_context({"figure.figsize": (18, 5)}):
                        plt.rcParams['lines.markersize'] = 5
                        plt.rcParams['lines.linewidth'] = 2

                        sc = plt.scatter(
                            new_df_all['dates'],
                            new_df_all['median_pos'],
                            c=new_df_all['avg_suitability'],
                            cmap='viridis',
                            s=20,
                        )
                        cbar = plt.colorbar(sc)
                        cbar.set_label('Average Suitability', fontsize=20)
                        cbar.ax.tick_params(labelsize=12)

                        plt.plot(
                            new_df_all_filter['dates'],
                            new_df_all_filter['median_pos'],
                            '--o',
                            color='k'
                        )

                        plt.fill_between(
                            new_df_all_filter['dates'],
                            new_df_all_filter['min_pos'],
                            new_df_all_filter['max_pos'],
                            color='k',
                            alpha=0.25
                        )

                        plt.ylabel('Cross-Shore Position (m)', fontsize=20)
                        plt.xlabel('Time (UTC)', fontsize=20)

                        years = np.arange(1984, 2026, 2)
                        ticks = [np.datetime64(f"{y}-01-01") for y in years]
                        plt.xticks(ticks, years, rotation=60, fontsize=20)
                        plt.yticks(fontsize=20)

                        plt.xlim(min(ticks), max(ticks))
                        plt.ylim(min(new_df_all['min_pos']), max(new_df_all['max_pos']))
                        plt.minorticks_on()
                        plt.text(
                            0.01, 0.98,
                            f"SNR (unfiltered): {snr_unfiltered:.3f}\n"
                            f"SNR (filtered):   {snr_filtered:.3f}\n"
                            f"Entropy (unfiltered): {entropy_unfiltered:.3f}\n"
                            f"Entropy (filtered):   {entropy_filtered:.3f}\n"
                            f"SNR change: {snr_arrow}    Entropy change: {entropy_arrow}",
                            transform=plt.gca().transAxes,
                            fontsize=16,
                            va='top',
                            ha='left',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                        )
                        plt.tight_layout()
                        plt.savefig(fig_save, dpi=500)
                        plt.close('all')
                    i=i+1
                except Exception as e:
                    print('error making figure')
                    traceback.print_exc()
                    tb_list = traceback.extract_tb(e.__traceback__)
                    # The last element in tb_list corresponds to the line where the exception occurred
                    filename, line_number, function_name, text = tb_list[-1]
                    plt.close('all')
                    i=i+1
                    continue
            else:
                i=i+1
                continue
        except Exception as e:
            traceback.print_exc()
            tb_list = traceback.extract_tb(e.__traceback__)
            # The last element in tb_list corresponds to the line where the exception occurred
            filename, line_number, function_name, text = tb_list[-1]
            i=i+1
            continue

    unfiltered_df = pd.concat(dfs2)
    unfiltered_df['cross_distance'] = unfiltered_df['cross_distance'].astype(float)
    unfiltered_df['cross_distance_min'] = unfiltered_df['cross_distance_min'].astype(float)
    unfiltered_df['cross_distance_rgb'] = unfiltered_df['cross_distance_rgb'].astype(float)
    unfiltered_df['cross_distance_nir'] = unfiltered_df['cross_distance_nir'].astype(float)
    unfiltered_df['cross_distance_swir'] = unfiltered_df['cross_distance_swir'].astype(float)
    unfiltered_df['ci'] = unfiltered_df['ci'].astype(float)
    unfiltered_df['avg_suitability'] = unfiltered_df['avg_suitability'].astype(float)
    unfiltered_df['avg_slope'] = unfiltered_df['avg_slope'].astype(float)
    unfiltered_df['tide'] = unfiltered_df['tide'].astype(float)
    unfiltered_df_path = os.path.join(os.path.dirname(input_file_rgb), section_string+'_unfiltered_tidally_corrected_transect_time_series_merged.csv')
    unfiltered_df.to_csv(unfiltered_df_path)

    filtered_df = pd.concat(dfs)
    filtered_df['cross_distance'] = filtered_df['cross_distance'].astype(float)
    filtered_df['cross_distance_min'] = filtered_df['cross_distance_min'].astype(float)
    filtered_df['cross_distance_rgb'] = filtered_df['cross_distance_rgb'].astype(float)
    filtered_df['cross_distance_nir'] = filtered_df['cross_distance_nir'].astype(float)
    filtered_df['cross_distance_swir'] = filtered_df['cross_distance_swir'].astype(float)
    filtered_df['ci'] = filtered_df['ci'].astype(float)
    filtered_df['avg_suitability'] = filtered_df['avg_suitability'].astype(float)
    filtered_df['avg_slope'] = filtered_df['avg_slope'].astype(float)
    filtered_df['tide'] = filtered_df['tide'].astype(float)
    filtered_df_path = os.path.join(os.path.dirname(input_file_rgb), section_string+'_filtered_tidally_corrected_transect_time_series_merged.csv')
    filtered_df.to_csv(filtered_df_path)
    
    resampled_df = pd.concat(dfs_resample_filter)
    resampled_df_path = os.path.join(os.path.dirname(input_file_rgb), section_string+'_resampled_tidally_corrected_transect_time_series_merged.csv')
    resampled_df.to_csv(resampled_df_path)

    ##pivot to make the matrix
    resampled_mat = resampled_df.pivot(index='dates', columns='transect_id', values='cross_distance')
    resampled_mat.columns.name = None
    resampled_mat_path = os.path.join(os.path.dirname(input_file_rgb), section_string+'_resampled_tidally_corrected_transect_time_series_matrix.csv')
    resampled_mat.to_csv(resampled_mat_path)

    unfiltered_lines = os.path.join(os.path.dirname(input_file_rgb), section_string+'_unfiltered_tidally_corrected_lines.geojson')
    unfiltered_points = os.path.join(os.path.dirname(input_file_rgb), section_string+'_unfiltered_tidally_corrected_points.geojson')

    filtered_lines = os.path.join(os.path.dirname(input_file_rgb), section_string+'_filtered_tidally_corrected_lines.geojson')
    filtered_points = os.path.join(os.path.dirname(input_file_rgb), section_string+'_filtered_tidally_corrected_points.geojson')

    savename_lines = os.path.join(os.path.dirname(input_file_rgb), section_string+'_reprojected_lines.geojson')
    savename_points = os.path.join(os.path.dirname(input_file_rgb), section_string+'_reprojected_points.geojson')
   
    transect_timeseries_to_wgs84(unfiltered_df_path,
                                 input_file_transects,
                                 unfiltered_lines,
                                 unfiltered_points)

    transect_timeseries_to_wgs84(filtered_df_path,
                                 input_file_transects,
                                 filtered_lines,
                                 filtered_points)

    transect_timeseries_to_wgs84_resampled(resampled_df_path,
                                 input_file_transects,
                                 savename_lines,
                                 savename_points)



