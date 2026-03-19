"""
Author: Mark Lundine
Takes coastseg outputs
(specifically the transect_timeseries.csv or transect_timeseries_tidally_corrected_matrix.csv)
and makes a geojson that can be used to construct a web map
depicting linear shoreline change rates with linked csvs and figures of particular
transect data. Each transect's length is proportional to 100 years of shoreline growth/retreat
at the computed linear rate. The direction is shoreward if the computed rate is negative
and seaward if the computed rate is positive.

Some places to edit would lie within the get_trend() and plot_timeseries() functions.
More thought should be put into how each timeseries is sampled/processed.
In some cases, their are big gaps in time as well as obvious outliers due to faulty
shoreline delineation, due to noise in the source images (clouds, shadows, data gaps, etc).

A cool addtion would be a function that takes the geojson from get_trends() and
constructs the map in an open-source format.
"""
# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import datetime
import shapely
from math import degrees, atan2, radians
from scipy import stats

def add_north_arrow(ax, north_arrow_params):
    """
    adds a north arrow to a geopandas plot

    inputs:
    ax (geopandas plot axis): the plot you want to add the north arrow to
    north_arrow_params (tuple): (x, y, arrow_length)

    """
    x,y,arrow_length = north_arrow_params
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='white', width=2, headwidth=4),
                ha='center', va='center', fontsize=8, color='white',
                xycoords=ax.transAxes)
    
def gb(x1, y1, x2, y2):
    """
    gets bearing from point 1 to point 2, utm coords

    inputs:
    x1 (float): first x coordinate
    y1 (float): first y coordinate
    x2 (float): second x coordinate
    y2 (float): second y coordinate

    outputs:
    angle (float): the bearing/angle from point 1 to point 2
    """
    angle = degrees(atan2(y2 - y1, x2 - x1))
    return angle

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

def find_inflection_points(x, y):
    """
    Finds inflection points in a given curve using numerical differentiation.
    
    Args:
        x: Array of x values.
        y: Array of corresponding y values.
        
    Returns:
        A list of tuples, where each tuple contains the x and y values of an inflection point.
    """
    
    # Calculate the second derivative
    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)
    
    # Find points where the second derivative changes sign
    inflection_points = []
    for i in range(1, len(second_derivative)):
        if np.sign(second_derivative[i]) != np.sign(second_derivative[i-1]):
            inflection_points.append((x[i], y[i]))
    
    return inflection_points

def prep_trend_data(filter_df):
    filtered_datetimes = np.array(filter_df['date'])
    shore_pos = np.array(filter_df['pos'])
    datetimes_seconds = [None]*len(filtered_datetimes)
    initial_time = filtered_datetimes[0]
    for i in range(len(filter_df)):
        t = filter_df['date'].iloc[i]
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes_seconds[i] = dt_sec
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)
    x = datetimes_years
    y = shore_pos
    return x,y

def get_trend(filter_df, transect_id, output_dir):

    """
    LLS on single transect timeseries
    inputs:
    filter_df (pandas DataFrame): two columns, dates and cross-shore positions
    trend_plot_path (str): path to save plot to
    outputs:
    lls_result: all the lls results (slope, intercept, stderr, intercept_stderr, rvalue)
    """
    try:
        x,y = prep_trend_data(filter_df)
    except:
        return None, None, None, None, None, None, None, None, None
    lls_result = stats.linregress(x,y)
    filter_df_1980s = filter_df[filter_df['date'].dt.year<1990]
    try:
        x,y = prep_trend_data(filter_df_1980s)
        lls_result_1980s = stats.linregress(x,y)
    except:
        lls_result_1980s = None
    filter_df_1990s = filter_df[(filter_df['date'].dt.year<2000) & (filter_df['date'].dt.year>=1990)]
    try:
        x,y = prep_trend_data(filter_df_1990s)
        lls_result_1990s = stats.linregress(x,y)
    except:
        lls_result_1990s = None
    filter_df_2000s = filter_df[(filter_df['date'].dt.year<2010) & (filter_df['date'].dt.year>=2000)]
    try:
        x,y = prep_trend_data(filter_df_2000s)
        lls_result_2000s = stats.linregress(x,y)
    except:
        lls_result_2000s = None
    filter_df_2010s = filter_df[(filter_df['date'].dt.year<2020) & (filter_df['date'].dt.year>=2010)]
    try:
        x,y = prep_trend_data(filter_df_2010s)
        lls_result_2010s = stats.linregress(x,y)
    except:
        lls_result_2010s = None
    filter_df_2020s = filter_df[(filter_df['date'].dt.year<2030) & (filter_df['date'].dt.year>=2020)]
    try:
        x,y = prep_trend_data(filter_df_2020s)
        lls_result_2020s = stats.linregress(x,y)
    except:
        lls_result_2020s = None

    timeseries_plot_dir = os.path.join(output_dir, 'final_timeseries')
    try:
        os.mkdir(timeseries_plot_dir)
    except:
        pass
    timeseries_plot_path = os.path.join(timeseries_plot_dir, transect_id+'.png')
    timeseries_csv_path = os.path.join(timeseries_plot_dir, transect_id+'.csv')

    signal_to_noise = plot_timeseries(filter_df,
                                      lls_result,
                                      lls_result_1980s,
                                      lls_result_1990s,
                                      lls_result_2000s,
                                      lls_result_2010s,
                                      lls_result_2020s,
                                      timeseries_plot_path)
    filter_df = filter_df.rename(columns={'pos':'cross_distance'})
    filter_df.to_csv(timeseries_csv_path)
    return lls_result, signal_to_noise,lls_result_1980s,lls_result_1990s,lls_result_2000s,lls_result_2010s,lls_result_2020s,timeseries_plot_path,timeseries_csv_path

def plot_timeseries(filter_df,
                    lls_result,
                    lls_result_1980s,
                    lls_result_1990s,
                    lls_result_2000s,
                    lls_result_2010s,
                    lls_result_2020s,
                    timeseries_plot_path):
    """
    Makes and saves a plot of the timeseries
    inputs:
    filter_df (pandas DataFrame): two columns, dates and cross-shore positions
    timeseries_plot_path (str): path to save figure to
    outputs:
    nothing
    """
    plt.rcParams["figure.figsize"] = (16,6)
    x = filter_df['date']
    y = filter_df['pos']
    s = filter_df['ci']
    signal_to_noise = np.mean(y)/np.std(y)
    try:
        trend = lls_result.slope
        unc = lls_result.stderr*1.96
    except:
        trend=np.nan
        unc=np.nan
    try:
        trend_1980s = lls_result_1980s.slope
        unc_1980s = lls_result_1980s.stderr*1.96
    except:
        trend_1980s=np.nan
        unc_1980s=np.nan
    try:
        trend_1990s = lls_result_1990s.slope
        unc_1990s = lls_result_1990s.stderr*1.96
    except:
        trend_1990s=np.nan
        unc_1990s=np.nan
    try:
        trend_2000s = lls_result_2000s.slope
        unc_2000s = lls_result_2000s.stderr*1.96
    except:
        trend_2000s=np.nan
        unc_2000s=np.nan
    try:
        trend_2010s = lls_result_2010s.slope
        unc_2010s = lls_result_2010s.stderr*1.96
    except:
        trend_2010s=np.nan
        unc_2010s=np.nan
    try:
        trend_2020s = lls_result_2020s.slope
        unc_2020s = lls_result_2020s.stderr*1.96
    except:
        trend_2020s=np.nan
        unc_2020s=np.nan
    lab=('Ensemble Annual Mean'+
         '\nSNR = '+str(np.round(signal_to_noise,decimals=3))+
         '\nOverall Trend = '+str(np.round(trend,decimals=2))+' +/- ' + str(np.round(unc, decimals=1))+
         '\n1980s Trend = '+str(np.round(trend_1980s,decimals=2))+' +/- ' + str(np.round(unc_1980s, decimals=1))+
         '\n1990s Trend = '+str(np.round(trend_1990s,decimals=2))+' +/- ' + str(np.round(unc_1990s, decimals=1))+
         '\n2000s Trend = '+str(np.round(trend_2000s,decimals=2))+' +/- ' + str(np.round(unc_2000s, decimals=1))+
         '\n2010s Trend = '+str(np.round(trend_2010s,decimals=2))+' +/- ' + str(np.round(unc_2010s, decimals=1))+
         '\n2020s Trend = '+str(np.round(trend_2020s,decimals=2))+' +/- ' + str(np.round(unc_2020s, decimals=1))
    )

    plt.plot(x, y,  '--o', color='k', label=lab)
    plt.fill_between(x, y-s, y+s, color='k', alpha=0.5, label='95% CI')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(x), max(x))
    plt.ylim(np.nanmin(y-s), np.nanmax(y+s))
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(timeseries_plot_path,dpi=300)
    plt.close()
    return signal_to_noise
    
def get_trends(transect_timeseries_path,
               config_gdf_path,
               save_path):
    """
    Computes linear trends with LLS on each transect's timeseries data
    Saves geojson linking transect id's to trend values
    inputs:
    transect_timeseries (str): path to the transect_timeseries csv (or transect_timeseries_tidally_corrected_matrix.csv)
    config_gdf_path (str): path to the config_gdf (.geojson), it's assumed these are in WGS84
    outputs:
    save_path (str): path to geojson with adjusted transects (in WGS84), trends, csv path, timeseries plot path, trend plot path
    """

    ##Load in data
    timeseries_data = pd.read_csv(transect_timeseries_path)
    timeseries_data['date'] = pd.to_datetime(timeseries_data['dates'], utc=True)
    transects = gpd.read_file(config_gdf_path)

    ##Make new directories
    home = os.path.dirname(transect_timeseries_path)
    
    ##For each transect, compute LLS, make plots, make csvs
    slopes = [None]*len(transects) 
    intercepts = [None]*len(transects) 
    r_squares = [None]*len(transects) 
    slope_uncertainties = [None]*len(transects) 
    intercept_uncertainties = [None]*len(transects) 
    slopes_1980s = [None]*len(transects)
    slopes_1990s = [None]*len(transects)
    slopes_2000s = [None]*len(transects)
    slopes_2010s = [None]*len(transects)
    slopes_2020s = [None]*len(transects)
    unc_1980s = [None]*len(transects)
    unc_1990s = [None]*len(transects)
    unc_2000s = [None]*len(transects)
    unc_2010s = [None]*len(transects)
    unc_2020s = [None]*len(transects)
    fig_paths = [None]*len(transects)
    csv_paths = [None]*len(transects)
    breakpoints = [None]*len(transects) 
    signal_to_noises = [None]*len(transects)
    for i in range(len(slopes)):
        transect_id = int(transects['transect_id'].iloc[i])
        transect_data = timeseries_data[timeseries_data['transect_id']==transect_id].reset_index(drop=True)

        if len(transect_data)==0:
            i=i+1
            continue
        x = transect_data['date']
        y = transect_data['cross_distance']
        s = transect_data['ci']
        df = pd.DataFrame({'date':x,
                           'pos':y,
                           'ci':s
                           })
        filter_df = df
        filter_df = filter_df.dropna(how='any')
        lls_result,signal_to_noise,lls_result_1980s,lls_result_1990s,lls_result_2000s,lls_result_2010s,lls_result_2020s,fig_path,csv_path = get_trend(filter_df, str(transect_id), home)
        
        try:
            slopes[i] = lls_result.slope
        except:
            slopes[i] = None
        try:
            intercepts[i] = lls_result.intercept
        except:
            intercepts[i] = None
        try:
            r_squares[i] = lls_result.rvalue**2
        except:
            r_squares[i] = None
        try:
            slope_uncertainties[i] = lls_result.stderr*1.96
        except:
            slope_uncertainties[i] = None
        try:
            intercept_uncertainties[i] = lls_result.intercept_stderr*1.96
        except:
            intercept_uncertainties[i] = None
            #breakpoints[i] = break_point
        try:
            signal_to_noises[i] = signal_to_noise
        except:
            signal_to_noises[i] = None
        try:
            slopes_1980s[i] = lls_result_1980s.slope
        except:
            slopes_1980s[i] = None
        try:
            slopes_1990s[i] = lls_result_1990s.slope
        except:
            slopes_1990s[i] = None
        try:
            slopes_2000s[i] = lls_result_2000s.slope
        except:
            slopes_2000s[i] = None
        try:
            slopes_2010s[i] = lls_result_2010s.slope
        except:
            slopes_2010s[i] = None
        try:
            slopes_2020s[i] = lls_result_2020s.slope
        except:
            slopes_2020s[i] = None
        try:
            unc_1980s[i] = lls_result_1980s.stderr*1.96
        except:
            unc_1980s[i] = None
        try:
            unc_1990s[i] = lls_result_1990s.stderr*1.96
        except:
            unc_1990s[i] = None
        try:
            unc_2000s[i] = lls_result_2000s.stderr*1.96
        except:
            unc_2000s[i] = None
        try:
            unc_2010s[i] = lls_result_2010s.stderr*1.96
        except:
            unc_2010s[i] = None
        try:
            unc_2020s[i] = lls_result_2020s.stderr*1.96
        except:
            unc_2020s[i] = None
        try:
            fig_paths[i] = fig_path
        except:
            fig_paths[i] = None
        try:
            csv_paths[i] = csv_path
        except:
            csv_paths[i] = None

    ###Making the vector file with trends
    transect_ids = [None]*len(slopes)    
    max_slope = np.nanmax(np.abs(np.array(slopes,dtype=np.float64)))
    scaled_slopes = np.array(slopes,dtype=np.float64)*100
    new_lines = [None]*len(slopes)
    org_crs = transects.crs
    utm_crs = transects.estimate_utm_crs()
    transects_utm = transects.to_crs(utm_crs)
    for i in range(len(transects)):
        transect = transects_utm.iloc[i]
        if slopes[i]==None:
            continue
        else:
            transect_id = transect['transect_id']
            transect_ids[i] = transect_id
            first = transect.geometry.coords[0]
            last = transect.geometry.coords[1]
            midpoint = transect.geometry.centroid
            distance = scaled_slopes[i]
            if distance<0:
                angle = radians(gb(first[0], first[1], last[0], last[1])+180)
            else:
                angle = radians(gb(first[0], first[1], last[0], last[1]))
            northing = midpoint.y + abs(distance)*np.sin(angle)
            easting = midpoint.x + abs(distance)*np.cos(angle)
            line_arr = [(midpoint.x,midpoint.y),(easting,northing)]
            try:
                line = arr_to_LineString(line_arr)
                new_lines[i] = line
            except:
                line = None
            

    ##This file can be used to link figures and csvs in a web-based GIS map
    new_df = pd.DataFrame({'transect_id':transect_ids,
                           'linear_trend':slopes,
                           'linear_trend_95_confidence':slope_uncertainties,
                           'intercept':intercepts,
                           'intercept_95_confidence':intercept_uncertainties,
                           #'n_breakpoints':breakpoints,
                           'snr':signal_to_noises,
                           'linear_trend_1980s':slopes_1980s,
                           'linear_trend_1990s':slopes_1990s,
                           'linear_trend_2000s':slopes_2000s,
                           'linear_trend_2010s':slopes_2010s,
                           'linear_trend_2020s':slopes_2020s,
                           'linear_trend_1980s_95_confidence':unc_1980s,
                           'linear_trend_1990s_95_confidence':unc_1990s,
                           'linear_trend_2000s_95_confidence':unc_2000s,
                           'linear_trend_2010s_95_confidence':unc_2010s,
                           'linear_trend_2020s_95_confidence':unc_2020s,
                           'figure_path':fig_paths,
                           'csv_path':csv_paths})

    new_geo_df = gpd.GeoDataFrame(new_df, crs=utm_crs, geometry=new_lines)
    new_geo_df = new_geo_df[new_geo_df['geometry']!=None].reset_index(drop=True)
    new_geo_df_org_crs = new_geo_df.to_crs(org_crs)
    new_geo_df_org_crs.to_file(save_path)
    return save_path

def plot_trend_maps(transect_trends_geojson,
                    site,
                    north_arrow_parms=(0.15, 0.93, 0.2),
                    scale_bar_loc='upper left'):
    """
    Uses contextily and geopandas plotting to plot the trends on a map
    inputs:
    transect_trends_geojson (str): path to the transect with trends
                                   output from get_trends()
    site (str): site name
    north_arrow_params (tuple): (x, y, arrow_length), need to play with this for different locations
    scale_bar_loc (str): position for scale bar, need to play with this for different locations
    returns:
    None
    """
    transect_trends_gdf = gpd.read_file(transect_trends_geojson)
    transect_trends_gdf = transect_trends_gdf.to_crs('3857')
    ax = transect_trends_gdf.plot(column='linear_trend',
                                  legend=True,
                                  legend_kwds={'label':'Trend (m/year'},
                                  cmap='RdBu',
                                  )
    ax.set_title(site)
    cx.add_basemap(ax,
                   source=cx.providers.CartoDB.DarkMatter,
                   attribution=False
                   )
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1,
                           location=scale_bar_loc
                           )
                  )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(transect_trends_geojson),
                             site+'_trend_map.png'),
                dpi=500)
    plt.close('all')
    





