"""
Comparing CoastSeg generated SDS data with in-situ measurements.

Mark Lundine, USGS
"""


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx
import numpy as np
import os
import glob
import datetime
import warnings
import shapely
from math import degrees, atan2, radians
from scipy import stats

warnings.filterwarnings("ignore")

# USGS-style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 4,
    "lines.markersize": 10,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "0.85",
    "grid.linestyle": "--"
})

##set basemap source
BASEMAP_DARK = cx.providers.CartoDB.DarkMatter ##dark mode for trend maps
BASEMAP_IMAGERY = cx.providers.Esri.WorldImagery ##world imagery for error maps

def MAPE(in_situ, sds):
    """
    computes mean absolute percentage error

    inputs:
    in_situ (pandas series): in_situ cross-distance measurements
    sds (pandas series): sds cross-distance measurements

    mape (float): mean absolute percentage error
    """
    mape = np.mean(np.abs((in_situ-sds)/in_situ))*100
    return mape

def RMSPE(in_situ, sds):
    """
    computes root mean sqaured percentage error

    inputs:
    in_situ (pandas series): in_situ cross-distance measurements
    sds (pandas series): sds cross-distance measurements

    rmspe (float): root mean squared percentage error
    """

    rmspe = np.sqrt(np.mean(np.square(((in_situ - sds) / in_situ)), axis=0))*100
    return rmspe

def RMSE(in_situ, sds):
    """
    computes root mean squared error

    inputs:
    in_situ (pandas series): in_situ cross-distance measurements
    sds (pandas series): sds cross-distance measurements

    outputs:
    rmse (float): root mean squared error
    """

    rmse = np.sqrt(np.mean(np.square(in_situ - sds), axis=0))
    return rmse

def MAE(in_situ, sds):
    """
    computes mean absolute error

    inputs:
    in_situ (pandas series): in_situ cross-distance measurements
    sds (pandas series): sds cross-distance measurements

    outputs:
    mae (float): mean absolute error
    """

    mae = np.mean(abs(in_situ - sds))
    return mae

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
    
def remove_nans(arr):
    """
    removes nans from numpy array

    inputs:
    arr (numpy.array): array with nans

    outputs:
    arr (numpy.array): array with no nans
    """
    return arr[~np.isnan(arr)]

def remove_nones(my_list):
    """
    removes nans from numpy array

    inputs:
    my_list (list): list with nones

    outputs:
    new_list (list): list without nones
    """
    new_list = [x for x in my_list if x is not None]
    return new_list

def in_situ_comparison(r_home,
                       g,
                       c,
                       rr,
                       sss,
                       plot_timeseries=True,
                       which_estimate='ensemble',
                       window=10,
                       legend_loc=(0.4,0.6),
                       north_arrow_params=(0.05,0.2,0.1),
                       scale_bar_loc='lower left'):
    """
    compares in situ shoreline measurements with sds measurements from CoastSeg
    will output a number of figures, a csv with the matched up comparisons, and a new transects geojson with linear trends as a column
    outputs are saved to home/site/analysis_outputs
    inputs:
    """
    ##Constants
    ##fig parameter
    DPI = 500
    ##fig extension
    EXT = '.png'
    ##whether or not to make a plot for each timeseries
    PLOT_TIMESERIES = plot_timeseries
    ##window for analysis (comparing SDS vs in-situ within +/- WINDOW days)
    WINDOW = window
    ##HOME
    r_home = r_home

    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 2

    ##Setting up directories and data for analysis
    section_str = g+c+rr+sss
    data_dir = os.path.join(r_home, 'SSS'+sss)
    analysis_outputs = os.path.join(data_dir, 'in_situ_analysis_outputs')
    try:
        os.mkdir(analysis_outputs)
    except:
        pass
    analysis_outputs = os.path.join(data_dir, 'in_situ_analysis_outputs', which_estimate)
    try:
        os.mkdir(analysis_outputs)
    except:
        pass
    try:
        os.mkdir(os.path.join(analysis_outputs, 'distributions'))
    except:
        pass
    try:
        os.mkdir(os.path.join(analysis_outputs, 'error_maps'))
    except:
        pass


    ##Analysis starts here
    ##loading data
    ##transects_gdf
    transects_path = os.path.join(data_dir, section_str + '_transects.geojson')
    transects_gdf = gpd.read_file(transects_path)
    transects_gdf['transect_id'] = transects_gdf['transect_id'].astype(int)

    ##for the datetimes, i'm chopping of the hours, minutes, and seconds so that there not duplicate sds measurements from the same day
    ##in situ
    in_situ_csv = os.path.join(data_dir, section_str+'_in_situ_data_ar.csv')
    in_situ_df = pd.read_csv(in_situ_csv)
    in_situ_df['dates'] = pd.to_datetime(in_situ_df['dates'], utc=True).dt.floor('d')


    ##sds
    sds_csv = os.path.join(data_dir, section_str+'_filtered_tidally_corrected_transect_time_series_merged.csv')
    df_sds = pd.read_csv(sds_csv)
    df_sds['dates'] = pd.to_datetime(df_sds['dates'], utc=True).dt.floor('d')
    if which_estimate=='rgb':
        df_sds['cross_distance'] = df_sds['cross_distance_rgb']
    elif which_estimate=='nir':
        df_sds['cross_distance'] = df_sds['cross_distance_nir']
    elif which_estimate=='swir':
        df_sds['cross_distance'] = df_sds['cross_distance_swir']
    else:
        df_sds['cross_distance'] = df_sds['cross_distance']

    ##loop over unique transects in sds data
    transects = sorted(df_sds['transect_id'].unique())

    ##Skipping transects
    skip_transects = [None]

    ##just removing transects at Barter that pick up old ridges
    remove_idxes=[True]*len(transects)
    i=0
    for transect in transects:
        if transect in skip_transects:
            remove_idxes[i]=False
        i=i+1
        
    transects = np.array(transects)

    transects = transects[remove_idxes]
            
    """
    Initializing arrays,
    rmse_ arrays will contain the per-transect RMSE
    mae_ arrays will contain the per-trasnect MAE
    err_ arrays will contain the in-situ vs SDS error for each measurement pair
    square_err_ arrays will contain the squared error for each measurement pair
    """

    ##sds
    rmse_sdss = [None]*len(transects)
    mae_sdss = [None]*len(transects)
    abs_sdss = [None]*len(transects)
    err_sdss = [None]*len(transects)
    square_err_sdss = [None]*len(transects)

    ##Looping over each transect in study area
    comparisons = [None]*len(transects)
    for j in range(len(transects)):
        transect = transects[j]
        print(transect)
        ##sds
        filter_df_sds = df_sds[df_sds['transect_id']==transect]
        filter_df_sds = filter_df_sds.drop_duplicates(subset=['dates'], keep='first').reset_index(drop=True)
        filter_df_sds = filter_df_sds.sort_values(by='dates').reset_index(drop=True)

        #in situ
        filter_in_situ_df = in_situ_df[in_situ_df['transect_id']==transect]
        filter_in_situ_df = filter_in_situ_df.drop_duplicates(subset=['dates'], keep='first').reset_index(drop=True)
        filter_in_situ_df = filter_in_situ_df.sort_values(by='dates').reset_index(drop=True)


        ##matching up in-situ with sds,
        ##look at points where sds observation is within 11 days of in-situ observation
        filter_df_sds_merge = [None]*len(filter_in_situ_df)

        ##no corrections matching
        if (len(filter_in_situ_df) == 0) or len(filter_df_sds) == 0:
            print('no matches')
            abs_sds = None
            abs_sdss[j] = abs_sds
            square_err_sds = None
            square_err_sdss[j] = square_err_sds
            rmse_sds = np.nan
            mae_sds = np.nan
            rmse_sdss[j] = rmse_sds
            mae_sdss[j] = mae_sds         
        else:
            merged_in_time_sds = pd.merge_asof(filter_in_situ_df.rename(columns={'dates':'dates_in_situ'}),
                                               filter_df_sds.rename(columns={'dates':'dates_sds'}),
                                               left_on='dates_in_situ',
                                               right_on='dates_sds',
                                               direction='nearest',
                                               suffixes=['_in_situ', '_sds'],
                                               tolerance=pd.Timedelta(days=WINDOW)).dropna()
            merged_in_time_sds['timedelta'] = merged_in_time_sds['dates_in_situ']-merged_in_time_sds['dates_sds']
            merged_in_time_sds = merged_in_time_sds.sort_values(by='timedelta').reset_index(drop=True)
            merged_in_time_sds = merged_in_time_sds.drop_duplicates(subset=['dates_sds'], keep='first').reset_index(drop=True)
            merged_in_time_sds = merged_in_time_sds.sort_values(by='dates_in_situ').reset_index(drop=True)
            if len(merged_in_time_sds)==0:
                print('no matches')
                abs_sds = None
                abs_sdss[j] = abs_sds
                square_err_sds = None
                square_err_sdss[j] = square_err_sds
                rmse_sds = np.nan
                mae_sds = np.nan
                rmse_sdss[j] = rmse_sds
                mae_sdss[j] = mae_sds          
            else:
                ##computing rmse for a transect

                ##no corrections
                abs_sds = np.array(np.abs(merged_in_time_sds['cross_distance_in_situ'] - merged_in_time_sds['cross_distance_sds']))
                abs_sdss[j] = abs_sds
                square_err_sds = np.array(((merged_in_time_sds['cross_distance_in_situ'] - merged_in_time_sds['cross_distance_sds'])**2))
                square_err_sdss[j] = square_err_sds
                rmse_sds = np.sqrt(((merged_in_time_sds['cross_distance_in_situ'] - merged_in_time_sds['cross_distance_sds'])**2).mean())
                try:
                    mae_sds = sum(abs_sds)/len(abs_sds)
                except:
                    mae_sds = np.nan
                rmse_sdss[j] = rmse_sds
                mae_sdss[j] = mae_sds
                
                ##plotting each timeseries
                lab_sds = ('SDS\nRMSE = ' +str(np.round(rmse_sds,decimals=3)) + ' m' +
                          '\nMAE = ' +str(np.round(mae_sds,decimals=3))+ ' m')

                ##merging filtered sds data
                merged_in_time_sds = merged_in_time_sds.dropna().reset_index(drop=True)

                comparisons[j] = merged_in_time_sds
                ##merging unfiltered sds data
                merged_df = filter_df_sds

                ##plotting
                if PLOT_TIMESERIES==True:
                    transect_dir = os.path.join(analysis_outputs, 'timeseries')
                    try:
                        os.mkdir(transect_dir)
                    except:
                        pass
                    with plt.rc_context({"figure.figsize":(16,5)}):
                        plt.title('Transect ' + str(transect))


                        timedelta = datetime.timedelta(days=WINDOW)
                        ##plot raw
                        ##plot tide
                        ##plot in situ
                        plt.plot(merged_df['dates'],
                                 merged_df['cross_distance'], '--', color='k', label='SDS')
                        plt.scatter(merged_in_time_sds['dates_sds'],
                                    merged_in_time_sds['cross_distance_sds'],
                                    s=14,
                                    color='red',
                                    label='SDS Observations in Comparison')

                        ##in situ
                        plt.scatter(filter_in_situ_df['dates'],
                                    filter_in_situ_df['cross_distance'],
                                    s=14,
                                    color='lightsteelblue',
                                    label='In Situ')
                        plt.scatter(merged_in_time_sds['dates_in_situ'],
                                    merged_in_time_sds['cross_distance_in_situ'],
                                    s=14,
                                    color='blue',
                                    label='In Situ Observations in Comparison')
                        for i in range(len(merged_in_time_sds)):
                            plt.plot([merged_in_time_sds['dates_in_situ'].iloc[i],
                                      merged_in_time_sds['dates_sds'].iloc[i]],
                                     [merged_in_time_sds['cross_distance_in_situ'].iloc[i],
                                      merged_in_time_sds['cross_distance_sds'].iloc[i]],
                                     color='gray')
                                      

                        plt.ylabel('Cross-Shore Position (m)')
                        plt.xlabel('Time (UTC)')
                        plt.xlim(min(merged_in_time_sds['dates_in_situ'])-timedelta,
                                 max(merged_in_time_sds['dates_in_situ'])+timedelta)
                
                        plt.legend()
                        plt.minorticks_on()
                        plt.tight_layout()
                        plt.savefig(os.path.join(transect_dir, str(transect)+'_timeseries'+EXT), dpi=DPI)
                        plt.close('all')

    comparisons_df = pd.concat(comparisons)
    rem_cols = ['shore_x_raw',
                'shore_y_raw',
                'x_raw',
                'y_raw',
                'transect_id_sds_raw',
                'timedelta_raw',
                'dates_in_situ_tide',
                'transect_id_in_situ_tide',
                'cross_distance_in_situ_tide',
                'shore_x_tide',
                'shore_y_tide',
                'x_tide',
                'y_tide',
                'tide_tide'
                'transect_id_sds_tide'
                ]
    for col in rem_cols:
        try:
            comparisons_df = comparisons_df.drop(columns=[col])
        except:
            pass



    comparisons_df = comparisons_df.rename(columns={'dates_in_situ_sds':'dates_in_situ',
                                                    'dates_sds':'dates_sds',
                                                    'timedelta_tide':'timedelta',
                                                    'transect_id_in_situ_sds':'transect_id',
                                                    }
                                           )
    keep_cols = ['dates_in_situ',
                 'dates_sds',
                 'timedelta',
                 'transect_id',
                 'cross_distance_in_situ',
                 'cross_distance_sds_sds',
                ]
    comparisons_df.to_csv(os.path.join(analysis_outputs, section_str+'_compared_obs.csv'))

    rmse_df = pd.DataFrame({'transect_id':transects.astype(int),
                            'rmse_sds':rmse_sdss,
                            }
                           )
    mae_df = pd.DataFrame({'transect_id':transects.astype(int),
                           'mae_sds':mae_sdss,
                           }
                          )

    ##removing nans, these were in-situ points without an sds observation within 10 days
    rmse_sdss = remove_nans(np.array(rmse_sdss))
    mae_sdss = remove_nans(np.array(mae_sdss))
                           

    ##this list is used in the box plot later on
    data_rmse = [rmse_sdss]
    data_mae = [mae_sdss]

    ##errors
    err_sdss_concat = comparisons_df['cross_distance_in_situ']-comparisons_df['cross_distance_sds']

    ##MAPE
    MAPE_sds = MAPE(comparisons_df['cross_distance_in_situ'], comparisons_df['cross_distance_sds'])

    ##RMSPE
    RMSPE_sds = RMSPE(comparisons_df['cross_distance_in_situ'], comparisons_df['cross_distance_sds'])

    ##absolute errors
    abs_sdss_concat = np.abs(comparisons_df['cross_distance_in_situ']-comparisons_df['cross_distance_sds'])

    ##MAE
    MAE_sds = MAE(comparisons_df['cross_distance_in_situ'], comparisons_df['cross_distance_sds'])
    
    ##RMSE
    RMSE_sds = RMSE(comparisons_df['cross_distance_in_situ'], comparisons_df['cross_distance_sds'])

    ##iqr of sds values
    iqr_sds_sds = np.nanquantile(df_sds['cross_distance'], 0.75)-np.nanquantile(df_sds['cross_distance'], 0.25)

    ##labels for error distribution plots
    sds_lab = ('SDS\nMean Error = ' +
              str(np.round(np.mean(err_sdss_concat), decimals=3)) + ' m' +
              '\nsd = ' + str(np.round(np.std(err_sdss_concat), decimals=3)) + ' m' +
              '\n# of Observations = ' + str(len(err_sdss_concat))
              )


    comparisons_df['err_sds'] = err_sdss_concat
    comparisons_df['abs_err_sds'] = abs_sdss_concat

    keep_cols = ['dates_in_situ',
                 'dates_sds',
                 'timedelta',
                 'transect_id',
                 'tide',
                 'cross_distance_in_situ',
                 'cross_distance_sds_raw',
                 'cross_distance_sds_tide',
                 'err_raw',
                 'abs_err_raw',
                 'err_tide',
                 'abs_err_tide'
                 ]
    
    for col in comparisons_df.columns:
        if col not in keep_cols:
            try:
                comparisons_df = comparisons_df.drop(columns=[col])
            except:
                pass
    comparisons_df.to_csv(os.path.join(analysis_outputs, section_str+'_compared_obs.csv'))
    """
    Plotting error distributions
    """
    with plt.rc_context({"figure.figsize":(12,12)}):
        plt.subplot(1,1,1)
        plt.suptitle(section_str)
        plt.hist(err_sdss_concat,
                 label=sds_lab,
                 color='gray',
                 density=False,
                 bins=np.arange(min(err_sdss_concat),max(err_sdss_concat),1))
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.hist(err_sdss_concat,
                 label='Cumulative',
                 histtype='step',
                 color='blue',
                 cumulative=True,
                 density=True,
                 bins=np.arange(min(err_sdss_concat),max(err_sdss_concat),1))
        ax2.set_yticks([0,0.25,0.5,0.75,1])
        plt.hlines([0.5],min(err_sdss_concat), max(err_sdss_concat), colors = ['k'])
        plt.minorticks_on()
        ax.set_xlabel('Error (In-Situ vs. SDS, m)')
        ax.set_ylabel('Count')
        ax2.set_ylabel('Cumulative Density (1/m)')
        ax.set_xlim(min(err_sdss_concat), max(err_sdss_concat))
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_outputs,'distributions', section_str+'_err_dists'+EXT), dpi=DPI)
        plt.close('all')
        
    ##labels for absolute error distribution plots
    sds_lab = ('SDS\nMAE = ' +
              str(np.round(MAE_sds, decimals=3)) + ' m' +
              '\nRMSE = ' + str(np.round(RMSE_sds, decimals=3)) + ' m' +
              '\nMAPE = ' + str(np.round(MAPE_sds, decimals=3)) + '%' +
              '\nRMSPE = ' + str(np.round(RMSPE_sds, decimals=3)) + '%' +
              '\nSDS Position IQR = ' + str(np.round(iqr_sds_sds, decimals=3)) + ' m' +
              '\n# of Observations = ' + str(len(abs_sdss_concat))
              )

    """
    Plotting absolute error distributions
    """
    with plt.rc_context({"figure.figsize":(12,12)}):
        plt.subplot(1,1,1)
        plt.suptitle(section_str)
        plt.hist(abs_sdss_concat, label=sds_lab, color='gray', density=False, bins=np.arange(0,max(abs_sdss_concat),1))
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.hist(abs_sdss_concat,
                 label='Cumulative',
                 histtype='step',
                 color='blue',
                 cumulative=True,
                 density=True,
                 bins=np.arange(0,max(abs_sdss_concat),1))
        ax2.set_yticks([0,0.25,0.5,0.75,1])
        plt.hlines([0.5],0, max(abs_sdss_concat), colors = ['k'])
        plt.minorticks_on()
        ax.set_xlabel('Absolute Error (In-Situ vs. SDS, m)')
        ax.set_ylabel('Count')
        ax2.set_ylabel('Cumulative Density (1/m)')
        ax.set_xlim(0, max(abs_sdss_concat))
        ax.legend(loc=legend_loc)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_outputs, 'distributions', section_str+'_abs_err_dists'+EXT), dpi=DPI)
        plt.close('all')

    """
    Next set of code plots the maps of RMSE across the transects
    """

    transects_gdf = transects_gdf.merge(rmse_df, on='transect_id')
    transects_gdf = transects_gdf.merge(mae_df, on='transect_id')
    centroids = transects_gdf['geometry'].centroid
    all_data_rmse = np.array(transects_gdf['rmse_sds'])
    all_data_mae = np.array(transects_gdf['mae_sds'])
    transects_gdf["markersize_sds_rmse"] = np.linspace(min(all_data_rmse),
                                                 max(all_data_rmse),
                                                 len(transects_gdf['rmse_sds']))/max(all_data_rmse)

    transects_gdf["markersize_sds_mae"] = np.linspace(min(all_data_mae),
                                                 max(all_data_mae),
                                                 len(transects_gdf['mae_sds']))/max(all_data_mae)
    transects_gdf_centroid = transects_gdf.copy()
    transects_gdf_centroid['geometry'] = centroids
    transects_gdf_centroid = transects_gdf_centroid.to_crs(epsg=3857)

    ###Raw Experiment RMSE Map
    ax = transects_gdf_centroid.plot(column='rmse_sds',
                            legend=True,
                            vmin=min(all_data_rmse),
                            vmax=max(all_data_rmse),
                            legend_kwds={"label": "RMSE (m)", "orientation": "vertical", 'shrink': 0.3},
                            cmap='Reds',
                            markersize='rmse_sds',
                            )
    ax.set_title('SDS')
    cx.add_basemap(ax,
                   source=BASEMAP_IMAGERY,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_outputs,  'error_maps', section_str+'_sds_map_rmse'+EXT), dpi=DPI)
    plt.close('all')

    ###Raw Experiment MAE Map
    ax = transects_gdf_centroid.plot(column='mae_sds',
                            legend=True,
                            vmin=min(all_data_mae),
                            vmax=max(all_data_mae),
                            legend_kwds={"label": "MAE (m)", "orientation": "vertical", 'shrink': 0.3},
                            cmap='Reds',
                            markersize='mae_sds',
                            )
    ax.set_title('SDS')
    cx.add_basemap(ax,
                   source=BASEMAP_IMAGERY,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_outputs, 'error_maps', section_str+'_sds_map_mae'+EXT), dpi=DPI)
    plt.close('all')


