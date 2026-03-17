"""
Code to map satellite shorelines in Alaska.

Mark Lundine, Sharon Fitzpatrick, Daniel Buscombe
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # legacy tf.keras API for HF TF models
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF logs
# os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # optional: exact CPU numerics

# ---- Standard libs ----
import sys
import shutil
import glob
import datetime
import gc
import logging
import traceback
import yaml
import argparse   # CLI parsing

# ---- Numerics & plotting ----
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use("Agg")              # safe for headless servers (comment out if interactive)
import matplotlib.pyplot as plt

# ---- GIS / raster IO ----
from osgeo import gdal
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.features import shapes

# ---- Geometry / transforms ----
from shapely.geometry import box
from skimage.transform import resize
from skimage import measure
from skimage.filters import threshold_multiotsu
from PIL import Image
from pyproj import CRS, Transformer

# ---- Project-specific utilities ----
from common.common import *

# ---- Shoreline filter models ----
from ShorelineFilter import image_filter
from ShorelineFilter import image_segmentation_filter
from ShorelineFilter import shoreline_change_envelope

# ---- CoastSeg (TF SegFormer path lives here) ----
from coastseg import coastseg_logs
from coastseg.common import initialize_gee
from coastseg import coastseg_map
from coastseg import zoo_model
from coastseg import tide_correction
from segmentation.simple_segmentation import do_seg_array, binary_lab_to_color_lab

# ---- Transect, pansharpening, coregistration ----
from transects import generate_transects
from pansharpen import pansharpen
from coregistration import coregister_single

# ---- Post processing ----
from post_processing import make_transect_csvs
from post_processing.trend_maps import get_trends

# ---- Beach slope & validation ----
from BeachSlope import dem_to_beach_slope
from validation import in_situ_validation

# ---- Profiler (optional) ----
from line_profiler import profile

# ---- Early runtime checks ----
def _preflight():
    import tensorflow as tf
    import transformers

    gpus = tf.config.list_physical_devices("GPU")
    print("TF:", tf.__version__, "| GPUs:", gpus)

    from transformers import TFSegformerForSemanticSegmentation
    print("Transformers:", transformers.__version__, "| TF SegFormer import OK")

    print("GDAL (osgeo):", gdal.__version__)
    print("Rasterio:", rasterio.__version__)

# Uncomment to run once at startup
_preflight()

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", type=str, help="config file, e.g., config.yaml")
args = parser.parse_args()

# parser.add_argument("-g", "--global_region", type=str, help='global region, ex: 1 for USA')
# parser.add_argument("-c", "--coastal_area", type=str, help='coastal area, ex: 4 for Alaska')
# parser.add_argument("-rr", "--subregion", type=str, help='subregion')
# parser.add_argument("-sss", "--shoreline_section", type=str, required=False, default = '', help='shoreline section')
# parser.add_argument("-gpu", "--gpu_id", type=str, required=False, default='-1', help='0,1,etc. for use gpu, -1 for cpu')
# parser.add_argument("-u", "--update", action='store_true', help='updating with new imagery')
# parser.add_argument("-r", "--reset", action='store_true', help='resetting')
# parser.add_argument("-p", "--planet_bool", action='store_true', help='doing PlanetScope imagery')
# parser.add_argument("-f", "--function", type=str, help='specify function to run, rename_transects will rename the transects to USGS scheme\n'+
#                                                         'transects will make transects given a reference shoreline and reference polygon'+
#                                                         'get_slope will get beach slopes with DEM specified (flag -d)\n'+
#                                                         'prep_slope will either set a constant slope for each transect or smooth the slopes gathered from the dem\n'+
#                                                         'rois will make rois\n'+
#                                                         'download will download imagery through GEE\n'+
#                                                         'find_rasters will find the rasters that intersect with the rois\n'+
#                                                         'check_planet will check if we have planet data for a particular section\n'+
#                                                         'image_filter will run the image suitability filter\n'+
#                                                         'reorg will reorganize the tiffs from CoastSeg format to the headless format\n'+
#                                                         'pansharpen_coreg will pansharpen and coregister the imagery\n'+
#                                                         'seg will segment the imagery\n'+
#                                                         'seg_filter will run the segmentation filter\n'+
#                                                         'extract will find the waterline contours\n'+
#                                                         'post_process will find intersections, tidally correct, ensemble, filter, resample, and compute trends\n'+
#                                                         'record_stats will compute the mean, median, q1, q3, min, max shoreline from the satellite annual record\n'+
#                                                         'validate will run an in-situ comparison')
# parser.add_argument("-d", "--dem", type=str,required=False, default = '', help='for computing slopes, if DEM is not specified, then slopes are computed from ArticDEM')
# parser.add_argument("-s", "--slope", type=str,required=False, default = '', help='specify a constant beach slope, ex: 0.05')
# parser.add_argument("-ss", "--smooth_slopes", action='store_true', help='if using slopes on transects, this will smooth the slopes')
# parser.add_argument("-re", "--reference_elevation", type=str,required=False, default = '', help='reference elevation, ex: 0')
# parser.add_argument("-rf", "--resample_frequency", type=str,required=False, default = '365D', help='resampling frequency, default is yearly or 365D')
# parser.add_argument("-m", "--model", type=str,required=False, default = 'global', help='which model to use, ak should be used on Alaska sites, global for non-Alaska')
# parser.add_argument("-e", "--estimate", type=str,required=False, default = 'ensemble', help='for validation tests, make rgb to see rgb stats, nir to see nir stats, swir to see swir stats, or leave as ensemble to see ensemble stats')
# parser.add_argument("-cs", "--custom_sections", type=str, required=False, default = '', help='custom sections to run on')
# parser.add_argument("-wf", "--waterline_filter", action="store_true", help="turns waterline filter on")
# parser.add_argument("-ymin", "--year_min", type=str, required=False, default="1984", help="year minimum for downloading imagery")
# parser.add_argument("-ymax", "--year_max", type=str, required=False, default="2026", help="year maximum for downloading imagery")
# parser.add_argument("-csrf", "--coastseg_roi_folder", type=str, required=True, help="path to CoastSeg/data")
# parser.add_argument("-h", "--home", type=str, required=True, help="path for gcrrsss organization of data")


def load_simple_yaml(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
    return data

def get_script_path():
    """
    Returns the path to this script
    outputs:
    path to this script (str)
    """
    return os.path.dirname(os.path.abspath(__file__))

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

def check_planet_section(g, c, rr, sss, r_home):
    section = 'SSS'+sss
    section_dir = os.path.join(r_home, section)
    section_string = g+c+rr+section[3:]
    planet_list = os.path.join(section_dir, section_string+'_ms_lists', 'planet_ms_paths.csv')
    planet_list_df = pd.read_csv(planet_list)
    if len(planet_list_df)>0:
        print(g+c+rr+sss)
    
def download_imagery_section(g, c, rr, sss, r_home, landsat_sentinel_dir, min_date, max_date, ee_project):
    """
    Downloads imagery for a shoreline section using CoastSeg

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path to subregion directory
    landsat_sentinel_dir (str): path to save the imagery to
    min_date (str): 'YYYY-MM-DD' min acquisition date
    max_date (str): 'YYYY-MM-DD' max acquisition date
    """
    section = 'SSS'+sss
    section_dir = os.path.join(r_home, section)
    section_string = g+c+rr+section[3:]
    rois_path = os.path.join(r_home, section_dir, section_string+'_rois.geojson')
    data_dir = landsat_sentinel_dir    

    # if you get an error here, enter your project id
    initialize_gee(auth_mode = "localhost", project=ee_project)

    coastsegmap=coastseg_map.CoastSeg_Map(create_map=False)

    # sample ROI (Region of Interest) file
    roi = coastsegmap.load_feature_from_file('roi',rois_path)
    # get the select all the ROI IDs from the file and store them in a list
    roi_ids =  list(roi.gdf.id)
    print(f"Downloading imagery for ROI with ID {roi_ids}")
    # customize the settings for the imagery download
    settings = {
        'sat_list':['L5', 'L7', 'L8', 'L9','S2'],                    # list of satellites to download imagery from. Options: 'L5', 'L7', 'L8', 'L9','S2' or 'S1'---SAR
        'dates':[min_date, max_date], # Start and end date to download imagery
        'landsat_collection':'C02',           # GEE collection to use. CoastSeg uses the Landsat Collection 2 (C02) by default
        "image_size_filter": True,            # filter images into bad folder if the images are less than 60% of the expected area. If False, no images will be filtered
        "apply_cloud_mask": False,             # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
        "months_list":[1,2,3,4,5,6,7,8,9,10,11,12]
        }

    # download the imagery for that ROI to the /data folder
    coastsegmap.download_imagery(rois=roi.gdf,selected_ids=roi_ids,settings=settings,file_path=data_dir)

def rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home, sorted_planet_home = None, planet=False):
    """
    Gets all of the landsat and sentinel satellite imagery for Alaska

    inputs:
    roi_dir (str): path to the folder containing rois

    """
    section = 'SSS'+sss
    section_dir = os.path.join(r_home, section)
    section_string = g+c+rr+section[3:]
    new_csv_path_dir = os.path.join(section_dir, section_string+'_ms_lists')
    try:
        os.mkdir(new_csv_path_dir)
    except:
        pass
    if planet==True:
        new_csv_path = os.path.join(new_csv_path_dir, 'planet_ms_paths_scored_.csv')
        sat_image_list_df_path = os.path.join(section_dir, 
                                            section_string + '_ms_lists', 
                                            'planet_ms_paths_scored.csv')
    else:
        new_csv_path = os.path.join(new_csv_path_dir, 'landsat_sentinel_ms_paths_scored_.csv')
        sat_image_list_df_path = os.path.join(section_dir, 
                                            section_string + '_ms_lists', 
                                            'landsat_sentinel_ms_paths_scored.csv')
    sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    for col in sat_image_list_df.columns:
        if 'Unnamed' in col:
            sat_image_list_df = sat_image_list_df.drop(columns=[col])
            
    sat_image_list_df.to_csv(new_csv_path)       
    try:
        sat_image_list_df = sat_image_list_df[sat_image_list_df['model_scores']>=0.335].reset_index(drop=True)
        new_tiff_dir = os.path.join(section_dir, 'ms_tiff_paths')
        try:
            os.mkdir(new_tiff_dir)
        except:
            pass
        for i in range(len(sat_image_list_df)):
            image = sat_image_list_df['ms_tiff_path'].iloc[i]
            satname = sat_image_list_df['satnames'].iloc[i]
            if planet==False:
                sorted_alaska_home = sorted_alaska_home
            else:
                sorted_alaska_home = sorted_planet_home
            coastseg_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image))))
            coastseg_dir = os.path.join(sorted_alaska_home, coastseg_dir)
            image = os.path.join(coastseg_dir, satname, 'ms', os.path.basename(image))
            destination_folder = os.path.dirname(coastseg_dir)
            archive_file = coastseg_dir+'.tar.gz'
            if os.path.isfile(archive_file) and not os.path.isdir(coastseg_dir):
                print('need to unarchive')
                unpack_tar_gz(archive_file, destination_folder)
            elif os.path.isfile(archive_file) and not os.listdir(coastseg_dir):
                print('need to unarchive')
                unpack_tar_gz(archive_file, destination_folder)
            try:
                os.mkdir(os.path.join(new_tiff_dir, satname))
            except:
                pass
            new_image = os.path.join(new_tiff_dir, satname, os.path.basename(image))
            shutil.copyfile(image, new_image)
    except Exception as e:
        traceback.print_exc()
        print(section_string)

def update_metadata_section(g, c, rr, sss, r_home, planet=False):
    section = 'SSS'+sss
    sss = section[3:]
    section_dir = os.path.join(r_home, section)
    tiff_dir = os.path.join(section_dir, 'ms_tiff_paths')
    section_string = g+c+rr+section[3:]
    if planet==True:
        scored_csv_path = os.path.join(section_dir, section_string+'_ms_lists', 'planet_ms_paths_scored.csv')
    else:
        scored_csv_path = os.path.join(section_dir, section_string+'_ms_lists', 'landsat_sentinel_ms_paths_scored.csv')
    scored_csv_path = pd.read_csv(scored_csv_path)
    scored_csv_path = scored_csv_path[scored_csv_path['model_scores']>=0.335].reset_index(drop=True)
    scored_csv_path['old_ms_tiff_path'] = [None]*len(scored_csv_path)
    scored_csv_path['roi_folder'] = [None]*len(scored_csv_path)
    for i in range(len(scored_csv_path)):
        satname = scored_csv_path['satnames'].iloc[i]
        ms_tiff_path = scored_csv_path['ms_tiff_path'].iloc[i]
        im_path = scored_csv_path['im_paths'].iloc[i]
        analysis_image = scored_csv_path['im_paths'].iloc[i]
        roi_folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ms_tiff_path))))
        try:
            new_ms_tiff_path = os.path.join(tiff_dir, satname, os.path.basename(ms_tiff_path))
        except:
            print(ms_tiff_path)
            print(tiff_dir)
            print(satname)
        new_im_path = os.path.join(tiff_dir, satname, os.path.basename(im_path))
        new_roi_folder = os.path.basename(roi_folder)
        scored_csv_path.loc[i, 'old_ms_tiff_path'] = ms_tiff_path
        scored_csv_path.loc[i, 'ms_tiff_path'] = new_ms_tiff_path
        scored_csv_path.loc[i, 'im_paths'] = new_im_path
        scored_csv_path.loc[i, 'roi_folder'] = new_roi_folder
    if planet==True:
        scored_csv_path.to_csv(os.path.join(section_dir, section_string+'_ms_lists', 'planet_ms_paths_scored_update.csv'))
    else:
        scored_csv_path.to_csv(os.path.join(section_dir, section_string+'_ms_lists', 'landsat_sentinel_ms_paths_scored_update.csv'))

def pansharpen_and_co_register_section(g, c, rr, sss, r_home, landsat_sentinel_folder, planet=False):
    """
    Pansharpens and co-registers all imagery for a section.
    Must have run image suitability before using this function.
    This will pansharpen all bands for all imagery in a section, then will try to coregister all of the pansharpened imagery.
    Algorithm works by picking the highest scored LANDSAT image for an ROI as the reference image, 
    then will try to co-register all imagery in that ROI. 


    inputs:
    g (str): global region (ex: '1' for USA)
    c (str): coastal area (ex: '4' for Alaska)
    rr (str): subregion (ex: '00')
    sss (str): shoreline section (ex: '000' for section 000)
    r_home (str): path to the subregion folder (ex: r'/mnt/hdd_6tb/MerbokCode/G1/C4/RR00')
    """
    section = 'SSS'+sss
    section_dir = os.path.join(r_home, section)
    section_string = g+c+rr+section[3:]
    
    if planet==True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_update.csv')
        if os.path.isfile(sat_image_list_df_path)==True:
            sat_image_list_df = pd.read_csv(sat_image_list_df_path)
            sat_image_list_df['use_coregistered'] = [False]*len(sat_image_list_df)
            sat_image_list_df['use_pansharpened'] = [False]*len(sat_image_list_df)
            sat_image_list_df['use_original'] = [True]*len(sat_image_list_df)
            sat_image_list_df['analysis_image'] = sat_image_list_df['ms_tiff_path']
            sat_image_list_df.to_csv(sat_image_list_df_path)
        else:
            print('no planet data')
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_update.csv')
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
        sat_image_list_df['use_coregistered'] = [False]*len(sat_image_list_df)
        sat_image_list_df['use_pansharpened'] = [False]*len(sat_image_list_df)
        sat_image_list_df['use_original'] = [False]*len(sat_image_list_df)
        sat_image_list_df['analysis_image'] = [None]*len(sat_image_list_df)
        sat_image_list_df = get_new_df_with_reference_image(g, c, rr, r_home, section, landsat_sentinel_folder)
        sat_image_list_df.to_csv(sat_image_list_df_path)
        for idx in range(len(sat_image_list_df['ms_tiff_path'])):
            try:
                roi_folder = sat_image_list_df['roi_folder'].iloc[idx]
                image = sat_image_list_df['ms_tiff_path'].iloc[idx]
                satname = sat_image_list_df['satnames'].iloc[idx]
                potential_old_pansharpen = os.path.join('/', 'mnt', 'f', 'Merbok', 'sorted_alaska', roi_folder, satname, 'ms', 'pansharpen', os.path.basename(image))
                potential_old_co_reg = os.path.join('/', 'mnt', 'f', 'Merbok', 'sorted_alaska', roi_folder, satname, 'ms', 'pansharpen', 'coregistered', os.path.basename(image))
                pansharpen_output = os.path.join(os.path.dirname(image), 'pansharpen', os.path.basename(image))
                co_reg_image = os.path.join(os.path.dirname(image), 'pansharpen', 'coregistered', os.path.basename(image))
                satname = sat_image_list_df['satnames'].iloc[idx]
                date = sat_image_list_df['datetimes_utc'].iloc[idx]
                if os.path.isfile(pansharpen_output)==True and os.path.isfile(co_reg_image)==True:
                    print('already pansharpened and coregistered')
                    sat_image_list_df.at[idx,'analysis_image'] = co_reg_image
                    sat_image_list_df.at[idx,'use_coregistered'] = True
                    continue
                elif os.path.isfile(potential_old_pansharpen)==True and os.path.isfile(potential_old_co_reg)==True:
                    print('already pansharpened and coregistered, need to move')
                    try:
                        os.mkdir(os.path.join(os.path.dirname(image), 'pansharpen'))
                    except:
                        pass
                    try:
                        os.mkdir(os.path.join(os.path.dirname(image), 'pansharpen', 'coregistered'))
                    except:
                        pass
                    sat_image_list_df.at[idx,'analysis_image'] = co_reg_image
                    sat_image_list_df.at[idx,'use_coregistered'] = True
                    print('df updated')
                    try:
                        shutil.copyfile(potential_old_pansharpen, pansharpen_output)
                        shutil.copyfile(potential_old_co_reg, co_reg_image)
                    except Exception as e:
                        traceback.print_exc()
                        print(section_string)
                    print('files copied')
                    continue
                
                if sat_image_list_df['model_scores'].iloc[idx]<0.335:
                    continue

                ##loading the tif
                with rasterio.open(image) as src:
                    blue = src.read(1)
                    green = src.read(2)
                    red = src.read(3)
                    nir = src.read(4)
                    if satname != 'PS':
                        swir = src.read(5)
                    bounds = src.bounds
                    resolution = src.res
                    width = src.width
                    height = src.height
                    mask_value = src.meta['nodata']
                    transform = src.transform
                    count = src.count
                    crs = src.crs

                    ##no data mask
                    if mask_value is not None:
                        mask = nir != mask_value
                    else:
                        mask = None
                    data_polygon = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for i, (s, v) 
                        in enumerate(
                            shapes(nir, mask=mask, transform=src.transform)))
                    data_polygon = gpd.GeoDataFrame.from_features(list(data_polygon), crs=src.crs)

                    if mask_value is not None:
                        mask = nir == mask_value
                    else:
                        mask = None
                        
                    no_data_polygon = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for i, (s, v) 
                        in enumerate(
                            shapes(nir, mask=mask, transform=src.transform)))   
                    try:
                        no_data_polygon = gpd.GeoDataFrame.from_features(list(no_data_polygon), crs=src.crs)
                    except:
                        no_data_polygon = None

                ##make no data pixels zero
                if satname!='PS':
                    swir[swir==src.meta['nodata']]=0
                nir[nir==src.meta['nodata']]=0
                blue[blue==src.meta['nodata']]=0
                red[red==src.meta['nodata']]=0
                green[green==src.meta['nodata']]=0 

                ##getting important image constraints      
                xmin = bounds.left
                ymax = bounds.top
                x_res = resolution[0]
                y_res = resolution[1]
                sat_image_list_df.at[idx, 'xres'] = x_res
                sat_image_list_df.at[idx, 'yres'] = y_res

                ##making no data polygon
                try:
                    no_data_polygon = no_data_polygon.buffer(x_res*2).union_all()
                    data_polygon = data_polygon.union_all().difference(no_data_polygon)
                except:
                    pass

                # rescale rgb image
                rgb = rescale(np.dstack([red,green,blue]))

                ##pansharpen
                pansharpen_dir =  os.path.join(os.path.dirname(image), 'pansharpen')
                pansharpen_output = os.path.join(os.path.dirname(image), 'pansharpen', os.path.basename(image))
                try:
                    os.mkdir(pansharpen_dir)
                except:
                    pass
                if satname!='PS' and satname!='L7':
                    if os.path.isfile(pansharpen_output)==False:
                        pansharpened_image = pan_sharpen(image, roi_folder, satname, landsat_sentinel_folder)
                        with rasterio.open(
                            pansharpen_output,
                            'w',
                            driver='GTiff',
                            height=height,
                            width=width,
                            dtype=np.float32,
                            crs=crs,
                            transform=transform,
                            nodata=mask_value,
                            count=count
                        ) as dst:
                            for band in range(1,count+1):
                                data = pansharpened_image[:,:,band-1]
                                dst.write(data, band)
                    reference_image = sat_image_list_df.at[idx, 'reference_image']
                    co_reg_image = os.path.join(os.path.dirname(image), 'pansharpen', 'coregistered', os.path.basename(image))
                    if os.path.isfile(co_reg_image)==False:
                        try:
                            coregister_single.co_register_single(reference_image, pansharpen_output)
                        except:
                            pass
                    if os.path.isfile(co_reg_image)==True:
                        sat_image_list_df.at[idx, 'use_coregistered'] = True
                        sat_image_list_df.at[idx, 'analysis_image'] = co_reg_image
                    else:
                        if os.path.isfile(pansharpen_output)==True:
                            sat_image_list_df.at[idx, 'use_pansharpened'] = True
                            sat_image_list_df.at[idx, 'analysis_image'] = pansharpen_output
                        else:
                            sat_image_list_df.at[idx, 'use_original'] = True
                            sat_image_list_df.at[idx, 'analysis_image'] = image
                elif satname=='PS':
                    reference_image = sat_image_list_df.at[idx, 'reference_image']
                    co_reg_image = os.path.join(os.path.dirname(image), 'coregistered', os.path.basename(image))
                    if os.path.isfile(co_reg_image)==False:
                        try:
                            coregister_single.co_register_single(reference_image, image)
                        except:
                            pass
                    if os.path.isfile(co_reg_image)==True:
                        sat_image_list_df.at[idx, 'use_coregistered'] = True
                        sat_image_list_df.at[idx, 'analysis_image'] = co_reg_image
                    else:
                        sat_image_list_df.at[idx, 'use_original'] = True
                        sat_image_list_df.at[idx, 'analysis_image'] = image
                else:
                    if os.path.isfile(pansharpen_output)==False:
                        pansharpened_image = pan_sharpen(image, roi_folder, satname, landsat_sentinel_folder)
                        with rasterio.open(
                            pansharpen_output,
                            'w',
                            driver='GTiff',
                            height=height,
                            width=width,
                            dtype=np.float32,
                            crs=crs,
                            transform=transform,
                            nodata=mask_value,
                            count=count
                        ) as dst:
                            for band in range(1,count+1):
                                data = pansharpened_image[:,:,band-1]
                                dst.write(data, band)
                    if os.path.isfile(pansharpen_output)==True:
                        sat_image_list_df.at[idx, 'use_pansharpened'] = True
                        sat_image_list_df.at[idx, 'analysis_image'] = pansharpen_output
                    else:
                        sat_image_list_df.at[idx, 'use_original'] = True
                        sat_image_list_df.at[idx, 'analysis_image'] = image
            except Exception as e:
                exception_string = traceback.format_exc()
                print(exception_string)
                pass
            sat_image_list_df.to_csv(sat_image_list_df_path)


def get_raster_bounds(raster_path, which):
    with rasterio.open(raster_path) as src:
        original_bounds = src.bounds
        
        # Create a shapely box from the original bounds
        bbox_original = box(original_bounds.left, original_bounds.bottom, 
                            original_bounds.right, original_bounds.top)

        # Define the original CRS and the target CRS (WGS84 - EPSG:4326)
        original_crs = CRS(src.crs)
        wgs84_crs = CRS("EPSG:4326")

        # Create a transformer for the coordinate transformation
        transformer = Transformer.from_crs(original_crs, wgs84_crs, always_xy=True)

        # Transform the coordinates of the bounding box
        transformed_coords = transformer.transform(bbox_original.exterior.coords.xy[0], 
                                                bbox_original.exterior.coords.xy[1])
        
        # Create a new shapely box from the transformed coordinates
        bbox_wgs84 = box(min(transformed_coords[0]), min(transformed_coords[1]),
                        max(transformed_coords[0]), max(transformed_coords[1]))

        # Extract the minx, miny, maxx, maxy from the WGS84 bounding box
        minx, miny, maxx, maxy = bbox_wgs84.bounds

    if which=='minx':
        return minx
    elif which=='miny':
        return miny
    elif which=='maxx':
        return maxx
    elif which=='maxy':
        return maxy
    else:
        return None

def get_all_satellite_imagery_tiffs(landsat_sentinel_dir, planet_dir, update=True):
    """
    Gets all of the CoastSeg downloaded landsat and sentinel and Planet satellite imagery

    inputs:
    landsat_sentinel_dir (str): path to the folder containing Landsat and Sentinel ROIs
    planet_dir (str): path to the Planet ROIs
    update (bool): True to update the dataframe, False to not update
    """
    df_save_path = os.path.join(landsat_sentinel_dir, 'coastseg_satellite_imagery.csv')
    if os.path.isfile(df_save_path) and update==False:
        df = pd.read_csv(df_save_path)
    else:
        rois = get_immediate_subdirectories(landsat_sentinel_dir)
        ms_tiffs_list = [None]*len(rois)
        satnames_list = [None]*len(rois)
        rois_list = [None]*len(rois)
        rois_min_x_list = [None]*len(rois)
        rois_min_y_list = [None]*len(rois)
        rois_max_x_list = [None]*len(rois)
        rois_max_y_list = [None]*len(rois)
        datetimes_list = [None]*len(rois)
        i=0
        for roi in rois:
            end_idx = roi.find('_datetime')
            roi_id = roi[3:end_idx]
            config_gdf = gpd.read_file(os.path.join(landsat_sentinel_dir, roi, 'config_gdf.geojson'))
            roi_gdf = config_gdf[config_gdf['type']=='roi']
            roi_gdf = roi_gdf[roi_gdf['id']==roi_id]
            bounds = roi_gdf.bounds
            L5 = glob.glob(os.path.join(landsat_sentinel_dir, roi, 'L5', 'ms') + '/*ms.tif')
            satname_L5 = ['L5']*len(L5)
            datetimes_L5 = [os.path.basename(name)[0:19] for name in L5]
            L7 = glob.glob(os.path.join(landsat_sentinel_dir, roi, 'L7', 'ms') + '/*ms.tif')
            satname_L7 = ['L7']*len(L7)
            datetimes_L7 = [os.path.basename(name)[0:19] for name in L7]
            L8 = glob.glob(os.path.join(landsat_sentinel_dir, roi, 'L8', 'ms') + '/*ms.tif')
            satname_L8 = ['L8']*len(L8)
            datetimes_L8 = [os.path.basename(name)[0:19] for name in L8]
            L9 = glob.glob(os.path.join(landsat_sentinel_dir, roi, 'L9', 'ms') + '/*ms.tif')
            satname_L9 = ['L9']*len(L9)
            datetimes_L9 = [os.path.basename(name)[0:19] for name in L9]
            S2 = glob.glob(os.path.join(landsat_sentinel_dir, roi, 'S2', 'ms') + '/*ms.tif')
            satname_S2 = ['S2']*len(S2)
            datetimes_S2 = [os.path.basename(name)[0:19] for name in S2]
            tifs = L5+L7+L8+L9+S2
            satnames = satname_L5 + satname_L7 + satname_L8 + satname_L9 + satname_S2
            datetimes = datetimes_L5 + datetimes_L7 + datetimes_L8 + datetimes_L9 + datetimes_S2
            ms_tiffs_list[i] = tifs
            satnames_list[i] = satnames
            rois_list[i] = [roi]*len(tifs)
            rois_min_x_list[i] = [bounds.minx.iloc[0]]*len(tifs)
            rois_min_y_list[i] = [bounds.miny.iloc[0]]*len(tifs)
            rois_max_x_list[i] = [bounds.maxx.iloc[0]]*len(tifs)
            rois_max_y_list[i] = [bounds.maxy.iloc[0]]*len(tifs)
            datetimes_list[i] = datetimes

            i=i+1
        
        ps_dirs = get_immediate_subdirectories(planet_dir)
        planet_paths = [None]*len(ps_dirs)
        planet_satnames = [None]*len(ps_dirs)
        planet_datetimes = [None]*len(ps_dirs)
        planet_rois_min_x = [None]*len(ps_dirs)
        planet_rois_min_y = [None]*len(ps_dirs)
        planet_rois_max_x = [None]*len(ps_dirs)
        planet_rois_max_y = [None]*len(ps_dirs)
        j=0
        for d in ps_dirs:
            PS = glob.glob(os.path.join(planet_dir, d, 'PS', 'ms') + '/*.tif')
            datetimes_PS = [os.path.basename(name)[0:4] + '-' + os.path.basename(name)[4:6] + '-' + os.path.basename(name)[6:8] + '-' + os.path.basename(name)[9:11] + '-' + os.path.basename(name)[11:13] + '-' + os.path.basename(name)[13:15] for name in PS]
            satname_PS = ['PS']*len(PS)
            planet_rois_min_x[j] = [get_raster_bounds(image, 'minx') for image in PS]
            planet_rois_min_y[j] = [get_raster_bounds(image, 'miny') for image in PS]
            planet_rois_max_x[j] = [get_raster_bounds(image, 'maxx') for image in PS]
            planet_rois_max_y[j] = [get_raster_bounds(image, 'maxy') for image in PS]
            planet_paths[j] = PS
            planet_satnames[j] = satname_PS
            planet_datetimes[j] = datetimes_PS
            j=j+1

        ms_tiffs = [x for xs in ms_tiffs_list for x in xs]
        satnames = [x for xs in satnames_list for x in xs]
        rois = [x for xs in rois_list for x in xs]
        rois_min_x = [x for xs in rois_min_x_list for x in xs]
        rois_min_y = [x for xs in rois_min_y_list for x in xs]
        rois_max_x = [x for xs in rois_max_x_list for x in xs]
        rois_max_y = [x for xs in rois_max_y_list for x in xs]
        datetimes = [x for xs in datetimes_list for x in xs]
        planet_rois_min_x_concat = [x for xs in planet_rois_min_x for x in xs]
        planet_rois_min_y_concat = [x for xs in planet_rois_min_y for x in xs]
        planet_rois_max_x_concat = [x for xs in planet_rois_max_x for x in xs]
        planet_rois_max_y_concat = [x for xs in planet_rois_max_y for x in xs]
        planet_paths_concat = [x for xs in planet_paths for x in xs]
        planet_satnames_concat = [x for xs in planet_satnames for x in xs]
        planet_datetimes_concat = [x for xs in planet_datetimes for x in xs]

        ms_tiffs = ms_tiffs + planet_paths_concat
        satnames = satnames + planet_satnames_concat
        rois_min_x = rois_min_x + planet_rois_min_x_concat
        rois_min_y = rois_min_y + planet_rois_min_y_concat
        rois_max_x = rois_max_x + planet_rois_max_x_concat
        rois_max_y = rois_max_y + planet_rois_max_y_concat
        datetimes = datetimes + planet_datetimes_concat
        df = pd.DataFrame({
                        'ms_tiff_path':ms_tiffs,
                        'satnames':satnames,
                        'min_x':rois_min_x,
                        'min_y':rois_min_y,
                        'max_x':rois_max_x,
                        'max_y':rois_max_y,
                        'datetimes':datetimes})
    planet_df = df[df['satnames']=='PS'].reset_index(drop=True)
    planet_df.to_csv(os.path.join(landsat_sentinel_dir, 'planet_satellite_imagery.csv'))
    df.to_csv(os.path.join(landsat_sentinel_dir, 'coastseg_satellite_imagery.csv'))
    return df

def find_intersecting_rasters(df, roi_path, section_string, copy_folder):
    """
    gets rasters that intersect with roi
    inputs:
    raster_path (str): path to the source raster
    roi_path (str): path to the geojson shape
    copy_folder (str): path to save output to
    """
    roi = gpd.read_file(roi_path)
    dissolved = roi.dissolve()
    roi_bounds = dissolved.bounds
    bounds = box(*dissolved.total_bounds)
    roi_xmin = roi_bounds.minx.iloc[0]
    roi_ymin = roi_bounds.miny.iloc[0]
    roi_xmax = roi_bounds.maxx.iloc[0]
    roi_ymax = roi_bounds.maxy.iloc[0]
    intersect_df = df[df['min_x']<=roi_xmax]
    intersect_df = intersect_df[intersect_df['max_x']>=roi_xmin]
    intersect_df = intersect_df[intersect_df['min_y']<=roi_ymax]
    intersect_df = intersect_df[intersect_df['max_y']>=roi_ymin]
    intersect_df['datetimes_utc'] = pd.to_datetime(intersect_df['datetimes'], format = '%Y-%m-%d-%H-%M-%S', utc=True)
    intersect_df['year'] = intersect_df['datetimes_utc'].dt.year
    intersect_df['month'] = intersect_df['datetimes_utc'].dt.month
    planet_df = intersect_df[intersect_df['satnames']=='PS'].reset_index(drop=True)
    keep_columns = ['ms_tiff_path', 'satnames', 'min_x', 'min_y', 'max_x', 'max_y', 'datetimes', 'datetimes_utc', 'year', 'month']
    for col in intersect_df.columns:
        if col not in keep_columns:
            intersect_df = intersect_df.drop(columns=[col])
    for col in planet_df.columns:
        if col not in keep_columns:
            planet_df = planet_df.drop(columns=[col])        
    intersect_df.to_csv(os.path.join(copy_folder, 'landsat_sentinel_ms_paths.csv'))
    planet_df.to_csv(os.path.join(copy_folder, 'planet_ms_paths.csv'))

def batch_find_intersecting_section(g, c, rr, sss, r_home, landsat_sentinel_folder, planet_folder, satellite_images, update=True):
    """
    Gets satellite imagery that intersect with shoreline sections
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    r_home (str): path to subregion
    alaska_folder (str): path to where the alaska landsat and sentinel imagery lives
    """
    section = 'SSS'+sss
    print(section)
    section_dir = os.path.join(r_home, section)
    section_string = g+c+rr+sss
    polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    sat_image_list_dir = os.path.join(section_dir, section_string + '_ms_lists')
    try:
        os.mkdir(sat_image_list_dir)
    except:
        pass
    find_intersecting_rasters(satellite_images, polygon, section_string, sat_image_list_dir)

def batch_find_intersecting(g, c, rr, r_home, landsat_sentinel_folder, planet_folder, update=True):
    """
    Gets satellite imagery that intersect with shoreline sections
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    r_home (str): path to subregion
    landsat_sentinel_folder (str): path to folder containing all CoastSeg downloaded Landsat and Sentinel ROIs
    planet_folder (str): path to folder containing all Planet ROIs
    """
    sections = sorted(get_immediate_subdirectories(r_home))
    satellite_images = get_all_satellite_imagery_tiffs(landsat_sentinel_folder, planet_folder, update=update)
    print('got the satellite images')
    for section in sections:
        print(section)
        section_dir = os.path.join(r_home, section)
        section_string = g+c+rr+section[3:]
        polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
        sat_image_list_dir = os.path.join(section_dir, section_string + '_ms_lists')
        try:
            os.mkdir(sat_image_list_dir)
        except:
            pass
        find_intersecting_rasters(satellite_images, polygon, section_string, sat_image_list_dir)

def image_suitability_section(g, c, rr, sss, r_home, gpu=0, planet=False):
    """
    Image suitability on shoreline section

    g (str): global region
    c (str): coastal area
    rr (str): subregion
    r_home (str): path to subregion
    gpu (int): which GPU to use (0 or 1 or -1 for CPU)
    """  
    section = 'SSS'+sss
    print(section)
    section_dir = os.path.join(r_home, section)
    section_string = g+c+rr+sss
    if planet==True:
        sat_image_list_df = pd.read_csv(os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths.csv'))
    else:
        sat_image_list_df = pd.read_csv(os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths.csv'))
    sat_image_list_df = sat_image_list_df.dropna()
    path_to_model_ckpt = os.path.join(get_script_path(), 'ShorelineFilter', 'models', 'image_rgb', 'best.h5')
    output_folder = None
    if planet==True:
        result_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_image_suitability.csv')
    else:
        result_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_image_suitability.csv')
    image_filter.run_inference_rgb(path_to_model_ckpt,
                                    sat_image_list_df,
                                    None,
                                    result_path,
                                    sort=False,
                                    input_df=True, 
                                    gpu=gpu)
    good_bad_result = pd.read_csv(result_path)
    sat_image_list_df_merged = pd.merge(sat_image_list_df, good_bad_result, left_on='ms_tiff_path', right_on='im_paths')

    if planet==True:
        sat_image_list_df_merged.to_csv(os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored.csv'))
    else:
        sat_image_list_df_merged.to_csv(os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored.csv'))


def get_reference_image(unique_folder, sat_image_list_df):
    """
    gets reference image for a shoreline section

    inputs:
    unique_folder (str): name of CoastSeg downloaded ROI
    sat_image_list_df (str): df containing satellite image names for shoreline section

    outputs:
    reference_image (str): path to the reference image for that ROI
    satname (str): name of satellite ('L9', 'L8', or 'L5')
    """
    filter_df = sat_image_list_df[sat_image_list_df['roi_folder']==unique_folder].reset_index(drop=True)
    filter_df = filter_df.sort_values(by='model_scores', ascending=False)
    filter_df_landsat = filter_df[filter_df['satnames']=='L9'].reset_index(drop=True)

    if len(filter_df_landsat)==0:
        filter_df_landsat = filter_df[filter_df['satnames']=='L8'].reset_index(drop=True)
        if len(filter_df_landsat)==0:
            filter_df_landsat = filter_df[filter_df['satnames']=='L5'].reset_index(drop=True)
            if len(filter_df_landsat)==0:
                print('no landsat available')
                return None, None
    reference_image = filter_df_landsat['ms_tiff_path'].iloc[0]
    satname = filter_df_landsat['satnames'].iloc[0]
    return reference_image, satname

def get_new_df_with_reference_image(g, c, rr, r_home, section, landsat_sentinel_folder):
    """
    Makes new df of satellite imagery with the reference image assigned for a whole subregion

    g (str): global region
    c (str): coastal area
    rr (str): subregion
    r_home (str): path to subregion
    """
    section_dir = os.path.join(r_home, section)
    section_string = g+c+rr+section[3:]
    sat_image_list_df = pd.read_csv(os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_update.csv'))


    #sat_image_list_df['roi_folder']  = [None]*len(sat_image_list_df) 
    sat_image_list_df['reference_image'] = [None]*len(sat_image_list_df)
    # for i in range(len(sat_image_list_df)):
    #     file = sat_image_list_df['ms_tiff_path'].iloc[i]
    #     unique_folder = os.path.dirname(os.path.dirname(os.path.dirname(file)))
    #     sat_image_list_df.at[i,'roi_folder'] = unique_folder
    for i in range(len(sat_image_list_df)):
        try:
            file = sat_image_list_df['ms_tiff_path'].iloc[i]
            unique_folder = sat_image_list_df['roi_folder'].iloc[i]
            reference_image, satname = get_reference_image(unique_folder, sat_image_list_df)
            pansharpen_dir =  os.path.join(os.path.dirname(reference_image), 'pansharpen')
            pansharpen_output = os.path.join(os.path.dirname(reference_image), 'pansharpen', os.path.basename(reference_image))
            ##loading the tif
            if os.path.isfile(pansharpen_output)==False:
                print('pansharpening reference image')
                print(reference_image)
                with rasterio.open(reference_image) as src:
                    blue = src.read(1)
                    green = src.read(2)
                    red = src.read(3)
                    nir = src.read(4)
                    swir = src.read(5)
                    bounds = src.bounds
                    resolution = src.res
                    width = src.width
                    height = src.height
                    mask_value = src.meta['nodata']
                    transform = src.transform
                    count = src.count
                    crs = src.crs

                swir[swir==src.meta['nodata']]=0
                nir[nir==src.meta['nodata']]=0
                blue[blue==src.meta['nodata']]=0
                red[red==src.meta['nodata']]=0
                green[green==src.meta['nodata']]=0 

                # rescale rgb image
                rgb = rescale(np.dstack([red,green,blue]))

                try:
                    os.mkdir(pansharpen_dir)
                except:
                    pass
                pansharpened_image = pan_sharpen(reference_image, unique_folder, satname, landsat_sentinel_folder)
                with rasterio.open(
                    pansharpen_output,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    dtype=np.float32,
                    crs=crs,
                    transform=transform,
                    nodata=mask_value,
                    count=count
                ) as dst:
                    for band in range(1,count+1):
                        data = pansharpened_image[:,:,band-1]
                        dst.write(data, band)
            sat_image_list_df.at[i, 'reference_image'] = pansharpen_output
            print(pansharpen_output)
            print('reference image pansharpened')
        except Exception as e:
            traceback.print_exc()
            print(section_string)
            pass


    return sat_image_list_df            

def pan_sharpen(fn, roi_folder, satname, landsat_sentinel_folder):
    """
    pansharpens imagery
    fn (str): path to the tif
    satname (str): satellite

    outputs:
    im_ms_arr (np.ndarray): pansharpened multispectral imagery
    """
    fn = os.path.join(landsat_sentinel_folder, roi_folder, satname, 'ms', os.path.basename(fn))
    im_ms_arr = pansharpen.preprocess_single(fn, 
                                             satname,
                                             cloud_mask_issue=False, 
                                             pan_off=False, 
                                             collection='C02', 
                                             do_cloud_mask=False, 
                                             s2cloudless_prob=60
                                             )

    return im_ms_arr

def make_rois_section(g, c, rr, sss, r_home):
    """
    makes rois for a subregion

    g (str): global region
    c (str): coastal area
    rr (str): subregion
    r_home (str): path to subregion
    """
    section = 'SSS'+sss
    ##get files in section
    section_dir = os.path.join(r_home, section)
    section_string = g+c+rr+sss
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    rois = os.path.join(section_dir, section_string + '_rois.geojson')
    try:
        reference_shoreline_to_rois(reference_shoreline, reference_polygon, rois, distance=4000)
    except:
        try:
            reference_shoreline_to_rois(reference_shoreline, reference_polygon, rois, distance=3000)
        except:
            try:
                reference_shoreline_to_rois(reference_shoreline, reference_polygon, rois, distance=2000)
            except:
                try:
                    reference_shoreline_to_rois(reference_shoreline, reference_polygon, rois, distance=1000)
                except:
                    reference_shoreline_to_rois(reference_shoreline, reference_polygon, rois, distance=500)

def segment_imagery_section(g, c, rr, sss, r_home, model = 'global', gpu=0, save_seg_to_raster=True, planet=False):
    """
    Segments imagery for single shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    ##establishing section and necessary index csvs
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss
    if planet==True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_update.csv')
        if os.path.isfile(sat_image_list_df_path)==True:
            sat_image_list_df_to_seg_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        else:
            print('no planet data')
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_update.csv')
        sat_image_list_df_to_seg_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')

    ref_shore_buffer = 400

    # Extract Shoreline Settings, these are not used but necessary to access CoastSeg
    settings = {
        'min_length_sl': 50,       # minimum length (m) of shoreline perimeter to be valid
        'max_dist_ref':400,         # maximum distance (m) from reference shoreline to search for valid shorelines. This detrmines the width of the buffer around the reference shoreline  
        'cloud_thresh': 0.8,        # threshold on maximum cloud cover (0-1). If the cloud cover is above this threshold, no shorelines will be extracted from that image
        'dist_clouds': 200,         # distance(m) around clouds where shoreline will not be mapped
        'min_beach_area': 100,      # minimum area (m^2) for an object to be labelled as a beach
        'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        "apply_cloud_mask": False,   # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
    }

    if model == 'ak':
        model_str = "ak_segformer_RGB_4class_14037041"
        print('using AK model')

    else:
        model_str = "global_segformer_RGB_4class_14036903"
        print('using RGB model')

    # Getting model set up using CoastSeg zoo_model
    model_setting = {
                "sample_direc": None,
                "use_GPU": "1",  # 0 or 1 0 means no GPU
                "implementation": "BEST",  # BEST or ENSEMBLE 
                "model_type": model_str,
                "otsu": False, # Otsu Thresholding
                "tta": False,  # Test Time Augmentation
                "use_local_model": False,
                "local_model_path":None
            }

    zoo_model_instance = zoo_model.Zoo_Model()

    model_setting["img_type"] = 'RGB'

    # save settings to the zoo model instance
    settings.update(model_setting)

    # save the settings to the model instance
    zoo_model_instance.set_settings(**settings)
    
    ## prepare the model
    zoo_model_instance.prepare_model('BEST', model_str)
    model_list = zoo_model_instance.model_list


    sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    sat_image_list_df['seg_paths'] = [None]*len(sat_image_list_df)

    ##working with the tifs
    num_images = len(sat_image_list_df)

    ##dropping poorly scored images so we don't have to segment them
    sat_image_list_df_to_seg = sat_image_list_df[sat_image_list_df['model_scores']>=0.335]
    try:
        sat_image_list_df_to_seg['done'] = sat_image_list_df_to_seg['done']
    except:
        sat_image_list_df_to_seg['done'] = [None]*len(sat_image_list_df_to_seg)

    ##loop over each image
    shoreline_i = 0
    if len(sat_image_list_df)>0:
        for i in range(len(sat_image_list_df_to_seg['analysis_image'])):
            try:
                print(i/num_images*100)
                image = sat_image_list_df_to_seg['analysis_image'].iloc[i]
                if image==None or pd.isnull(image)==True:
                    sat_image_list_df_to_seg.at[i,'done'] = True
                    sat_image_list_df_to_seg.to_csv(sat_image_list_df_path)
                    continue
                basename = os.path.splitext(os.path.basename(image))[0]
                seg_dir = os.path.join(section_dir, 'segmentation')
                color_seg_lab_path = os.path.join(seg_dir, basename+'_seg.png')
                if sat_image_list_df_to_seg['done'].iloc[i] == True:
                    print('done')
                    sat_image_list_df_to_seg.at[i,'seg_paths'] = color_seg_lab_path
                    sat_image_list_df_to_seg.to_csv(sat_image_list_df_path)
                    continue
                image_suitability_score = sat_image_list_df_to_seg['model_scores'].iloc[i]
                image_score = image_suitability_score
                if image_suitability_score<0.335:
                    sat_image_list_df_to_seg.at[i,'done'] = True
                    sat_image_list_df_to_seg.to_csv(sat_image_list_df_path)
                    continue
            
                satname = sat_image_list_df_to_seg['satnames'].iloc[i]
                date = sat_image_list_df_to_seg['datetimes_utc'].iloc[i]


                try:
                    os.mkdir(seg_dir)
                except:
                    pass
                
                ##loading the tif
                with rasterio.open(image) as src:
                    blue = src.read(1)
                    green = src.read(2)
                    red = src.read(3)
                    nir = src.read(4)
                    if satname != 'PS':
                        swir = src.read(5)
                    bounds = src.bounds
                    resolution = src.res
                    width = src.width
                    height = src.height
                    mask_value = src.meta['nodata']
                    transform = src.transform
                    count = src.count
                    crs = src.crs

                    ##no data mask
                    if mask_value is not None:
                        mask = nir != mask_value
                    else:
                        mask = None
                        
                ##make no data pixels zero
                if satname!='PS':
                    swir[swir==src.meta['nodata']]=0
                nir[nir==src.meta['nodata']]=0
                blue[blue==src.meta['nodata']]=0
                red[red==src.meta['nodata']]=0
                green[green==src.meta['nodata']]=0 

                ##getting important image constraints      
                xmin = bounds.left
                ymax = bounds.top
                x_res = resolution[0]
                y_res = resolution[1]
                sat_image_list_df_to_seg.at[i, 'xres'] = x_res
                sat_image_list_df_to_seg.at[i, 'yres'] = y_res

                rgb = rescale(np.dstack([red,green,blue]))
                
                ##compute thresholds
                nir = np.nan_to_num(nir, nan=0.0, posinf=0.0, neginf=0.0)
                thresholds_nir = threshold_multiotsu(nir)

                if satname != 'PS':
                    swir = np.nan_to_num(swir, nan=0.0, posinf=0.0, neginf=0.0)
                    thresholds_swir = threshold_multiotsu(swir)

                # Apply the threshold to create a binary image
                binary_image_nir = (nir > min(thresholds_nir)).astype(int)
                binary_image_nir = scipy.ndimage.median_filter(binary_image_nir, size=5)

                if satname!= 'PS':
                    binary_image_swir = (swir > min(thresholds_swir)).astype(int)
                    binary_image_swir = scipy.ndimage.median_filter(binary_image_swir, size=5)

                height_p, width_p, channels_p = np.shape(rgb)
                array_to_seg = rgb
                data_mask = array_to_seg != (0,0,0)
                data_mask = data_mask[:,:,0]
                array_to_seg = np.nan_to_num(array_to_seg, nan=0.0, posinf=0.0, neginf=0.0)
                array_to_seg = (rescale(array_to_seg)*255).astype('uint8')

                print('segmenting')
                ##segment image
                seg_lab, color_seg_lab = do_seg_array(array_to_seg,
                                    model_list,
                                    gpu=gpu
                                    )
                
                ##resizing
                seg_lab = resize_image_numpy(seg_lab, (width_p, height_p))

                ##simplify segmented image
                seg_lab[seg_lab==1] = 0
                seg_lab[seg_lab>0] = 1
                color_seg_lab = Image.fromarray(color_seg_lab)
                color_seg_lab.save(color_seg_lab_path)

                ##put seg results in array
                sat_image_list_df_to_seg.at[i, 'seg_paths'] = color_seg_lab_path
                
                if save_seg_to_raster==True:
                    if satname != 'PS':
                        data_to_save = np.dstack([blue, 
                                                    green, 
                                                    red, 
                                                    nir, 
                                                    swir, 
                                                    np.float32(seg_lab), 
                                                    np.float32(binary_image_nir), 
                                                    np.float32(binary_image_swir)
                                                    ]
                                                    )
                        count = 8
                    else:
                        data_to_save = np.dstack([blue, 
                                                    green, 
                                                    red, 
                                                    nir, 
                                                    np.float32(seg_lab), 
                                                    np.float32(binary_image_nir)
                                                    ]
                                                    )
                        count = 6

                    with rasterio.open(
                        image,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        dtype=np.float32,
                        crs=crs,
                        transform=transform,
                        nodata=mask_value,
                        count=count
                    ) as dst:
                        for band in range(1,count+1):
                            data = data_to_save[:,:,band-1]
                            dst.write(data, band)
            except:
                pass
        sat_image_list_df_to_seg.to_csv(sat_image_list_df_to_seg_path)

def segmentation_suitability_section(g, c, rr, sss, r_home, gpu=0, planet=False):
    """
    Runs segmentation suitability for single shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss
    if planet==True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path)==False:
            print('no planet data')
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
    sat_image_list_df = pd.read_csv(sat_image_list_df_path)

    if len(sat_image_list_df)>0:
        for i in range(len(sat_image_list_df['seg_paths'])):
            color_seg_lab = sat_image_list_df['seg_paths'].iloc[i]
            
            if pd.isnull(color_seg_lab) != True:
                try:
                ##get seg score
                    color_seg_lab = np.array(Image.open(color_seg_lab))
                    seg_score = image_segmentation_filter.get_segmentation_score(color_seg_lab, gpu=gpu)
                    sat_image_list_df.at[i, 'seg_scores'] = seg_score
                except:
                    pass
    sat_image_list_df.to_csv(sat_image_list_df_path)
    gc.collect()

#@profile
def extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in geotiffs

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    crs = reference_shoreline_gdf.crs

    if planet==True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path)==True:
            sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
        else:
            print('no planet data')
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    shorelines_dir = os.path.join(section_dir, 'shorelines') 
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
 
    ##working with the tifs
    num_images = len(sat_image_list_df)
    if len(sat_image_list_df)>0:
        ##reset tasks, set everything to not done, delete old shorelines
        if reset==True:
            sat_image_list_df['shoreline_done'] = [None]*num_images
            try:
                shutil.rmtree(zoo_shoreline_dir)
                shutil.rmtree(nir_shoreline_dir)
                shutil.rmtree(swir_shoreline_dir)
            except:
                pass
        else:
            try:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['shoreline_done'] 
            except:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['done']
    
    dirs = [shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass 
    zoo_shorelines_list=glob.glob(zoo_shoreline_dir+'/*.geojson')
    ##loop over each image
    shoreline_i = 0
    if len(sat_image_list_df)>0:
        for i in range(len(sat_image_list_df['analysis_image'])):
            try:
                print(i/num_images*100)
                image = sat_image_list_df['analysis_image'].iloc[i]
                roi_folder = sat_image_list_df['roi_folder'].iloc[i]
                if sat_image_list_df['shoreline_done'].iloc[i] == True:
                    print('done')
                    sat_image_list_df.to_csv(sat_image_list_df_path)
                    try:
                        shoreline_i = len(gpd.read_file(extracted_shorelines_path))
                    except:
                        shoreline_i = 0
                    continue
                image_suitability_score = sat_image_list_df['model_scores'].iloc[i]
                image_score = image_suitability_score
                if image_suitability_score<0.335:
                    sat_image_list_df.at[i,'shoreline_done'] = True
                    sat_image_list_df.to_csv(sat_image_list_df_path)
                    try:
                        shoreline_i = len(gpd.read_file(extracted_shorelines_path))
                    except:
                        shoreline_i = 0
                    continue
                if image==None:
                    sat_image_list_df.at[i,'shoreline_done'] = True
                    sat_image_list_df.to_csv(sat_image_list_df_path)
                    try:
                        shoreline_i = len(gpd.read_file(extracted_shorelines_path))
                    except:
                        shoreline_i = 0
                    continue          
                satname = sat_image_list_df['satnames'].iloc[i]
                date = sat_image_list_df['datetimes_utc'].iloc[i]
                check_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00")
                check_date = check_date.strftime("%Y-%m-%d-%H-%M-%S")
                shoreline_path = os.path.join(zoo_shoreline_dir, check_date+'_'+satname+'_'+roi_folder+'.geojson')
                print(shoreline_path)
                if shoreline_path in zoo_shorelines_list:
                    print('extracted already')
                    sat_image_list_df.at[i,'shoreline_done'] = True
                    sat_image_list_df.to_csv(sat_image_list_df_path)
                    try:
                        shoreline_i = len(gpd.read_file(extracted_shorelines_path))
                    except:
                        shoreline_i = 0
                    continue     
                ##loading the tif
                with rasterio.open(image) as src:
                    if satname != 'PS':
                        nir = src.read(4)
                        seg_lab = src.read(6)
                        binary_image_nir = src.read(7)
                        binary_image_swir = src.read(8)
                    else:
                        nir = src.read(4)
                        seg_lab = src.read(5)
                        binary_image_nir = src.read(6)

                    bounds = src.bounds
                    resolution = src.res 
                    width = src.width
                    height = src.height
                    mask_value = src.meta['nodata']
                    transform = src.transform
                    count = src.count
                    crs = src.crs
                    xmin = bounds.left
                    ymax = bounds.top
                    x_res = resolution[0]
                    y_res = resolution[1]
                    
                    ##no data mask
                    if mask_value is not None:
                        mask = nir != mask_value
                    else:
                        mask = None

                    data_polygon = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for i, (s, v) 
                        in enumerate(
                            shapes(nir, mask=mask, transform=src.transform)))
                    data_polygon = gpd.GeoDataFrame.from_features(list(data_polygon), crs=src.crs)

                    if mask_value is not None:
                        mask = nir == mask_value
                    else:
                        mask = None
                        
                    no_data_polygon = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for i, (s, v) 
                        in enumerate(
                            shapes(nir, mask=mask, transform=src.transform)))   
                    try:
                        no_data_polygon = gpd.GeoDataFrame.from_features(list(no_data_polygon), crs=src.crs)
                    except:
                        no_data_polygon = None

                xmin = bounds.left
                ymax = bounds.top
                x_res = resolution[0]
                y_res = resolution[1]

                ##making no data polygon
                try:
                    no_data_polygon = no_data_polygon.buffer(x_res*2).unary_union()
                    data_polygon = data_polygon.unary_union().difference(no_data_polygon)
                except:
                    pass
                
                data_mask = nir!=mask_value
                no_data_mask = nir==mask_value
                seg_lab[no_data_mask]=np.nan
                
                ##get contours
                linestrings = get_contours(seg_lab,
                                           satname,
                                           xmin,
                                           ymax,
                                           x_res,
                                           y_res,
                                           data_polygon,
                                           reference_shoreline_gdf,
                                           ref_shore_buffer,
                                           reference_polygon,
                                           data_mask,
                                           crs)

                linestrings_nir = get_contours(binary_image_nir,
                                            satname,
                                            xmin,
                                            ymax,
                                            x_res,
                                            y_res,
                                            data_polygon,
                                            reference_shoreline_gdf,
                                            ref_shore_buffer,
                                            reference_polygon,
                                            data_mask,
                                            crs)

                if satname!='PS':
                    linestrings_swir = get_contours(binary_image_swir,
                                                satname,
                                                xmin,
                                                ymax,
                                                x_res,
                                                y_res,
                                                data_polygon,
                                                reference_shoreline_gdf,
                                                ref_shore_buffer,
                                                reference_polygon,
                                                data_mask,
                                                crs)

                image_score = sat_image_list_df['model_scores'].iloc[i]
                seg_score = sat_image_list_df['seg_scores'].iloc[i]
                waterlines = [None]*len(linestrings)
                shoreline_seg_scores =  [None]*len(linestrings)
                shoreline_image_scores =  [None]*len(linestrings)
                waterlines_nir = [None]*len(linestrings_nir)
                shoreline_seg_scores_nir = [None]*len(linestrings_nir)
                shoreline_image_scores_nir = [None]*len(linestrings_nir)
                if satname!='PS':
                    waterlines_swir = [None]*len(linestrings_swir)
                    shoreline_seg_scores_swir = [None]*len(linestrings_swir)
                    shoreline_image_scores_swir = [None]*len(linestrings_swir)
                date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00")
                date = date.strftime("%Y-%m-%d-%H-%M-%S")

                k=0
                for linestring in linestrings:
                    coords = LineString_to_arr(linestring)
                    if len(coords)>5:
                        waterlines[k] = linestring
                        shoreline_seg_scores[k] = seg_score
                        shoreline_image_scores[k] = image_score
                    k=k+1
                remove_nones(waterlines)
                remove_nones(shoreline_seg_scores)
                remove_nones(shoreline_image_scores)

                k=0
                for linestring in linestrings_nir:
                    coords = LineString_to_arr(linestring)
                    if len(coords)>5:
                        waterlines_nir[k] = linestring
                        shoreline_seg_scores_nir[k] = seg_score
                        shoreline_image_scores_nir[k] = image_score
                    k=k+1
                remove_nones(waterlines_nir)
                remove_nones(shoreline_seg_scores_nir)
                remove_nones(shoreline_image_scores_nir)

                k=0
                if satname!='PS':
                    for linestring in linestrings_swir:
                        coords = LineString_to_arr(linestring)
                        if len(coords)>5:
                            waterlines_swir[k] = linestring
                            shoreline_seg_scores_swir[k] = seg_score
                            shoreline_image_scores_swir[k] = image_score
                        k=k+1
                    remove_nones(waterlines_swir)
                    remove_nones(shoreline_seg_scores_swir)
                    remove_nones(shoreline_image_scores_swir)
                ##saving extracted shorelines
                if len(waterlines) == 0 or len(waterlines_nir) ==0:
                    sat_image_list_df.at[i,'shoreline_done'] = True
                    sat_image_list_df.to_csv(sat_image_list_df_path)
                    try:
                        shoreline_i = len(gpd.read_file(extracted_shorelines_path))
                    except:
                        shoreline_i = 0
                    print('no shorelines extracted')
                    continue

                shorelines_gdf = gpd.GeoDataFrame({'dates':[date]*len(waterlines),
                                                'image_suitability_score':shoreline_image_scores,
                                                'segmentation_suitability_score':shoreline_seg_scores,
                                                'satname':[satname]*len(waterlines)}, geometry=waterlines, crs=crs)
                shoreline_path = os.path.join(zoo_shoreline_dir, date+'_'+satname+'_'+roi_folder+'.geojson')
                shorelines_gdf = utm_to_wgs84_df(shorelines_gdf).reset_index(drop=True)
                shorelines_gdf = split_line(shorelines_gdf,
                                            'LineString',
                                            smooth=True)
                shorelines_gdf.to_file(shoreline_path)

                ##saving extracted shorelines
                nir_shoreline_path = os.path.join(nir_shoreline_dir, date+'_'+satname+'_'+roi_folder+'.geojson')
                shorelines_nir_gdf = gpd.GeoDataFrame({'dates':[date]*len(waterlines_nir),
                                                'image_suitability_score':shoreline_image_scores_nir,
                                                'segmentation_suitability_score':shoreline_seg_scores_nir,
                                                'satname':[satname]*len(waterlines_nir)}, geometry=waterlines_nir, crs=crs).reset_index(drop=True)
                shorelines_nir_gdf = utm_to_wgs84_df(shorelines_nir_gdf)
                shorelines_nir_gdf = split_line(shorelines_nir_gdf,
                                                'LineString',
                                                smooth=True)
                shorelines_nir_gdf.to_file(nir_shoreline_path)
                if satname!='PS':
                    ##saving extracted shorelines
                    swir_shoreline_path = os.path.join(swir_shoreline_dir, date+'_'+satname+'_'+roi_folder+'.geojson')
                    shorelines_swir_gdf = gpd.GeoDataFrame({'dates':[date]*len(waterlines_swir),
                                                    'image_suitability_score':shoreline_image_scores_swir,
                                                    'segmentation_suitability_score':shoreline_seg_scores_swir,
                                                    'satname':[satname]*len(waterlines_swir)}, geometry=waterlines_swir, crs=crs).reset_index(drop=True)
                    shorelines_swir_gdf = utm_to_wgs84_df(shorelines_swir_gdf)
                    shorelines_swir_gdf = split_line(shorelines_swir_gdf,
                                                     'LineString',
                                                     smooth=True)
                    shorelines_swir_gdf.to_file(swir_shoreline_path)       
                sat_image_list_df.at[i,'shoreline_done'] = True
                sat_image_list_df.to_csv(sat_image_list_df_path_shore)        
                shoreline_i = i+1
            except Exception as e:
                tb_list = traceback.extract_tb(e.__traceback__)
                # The last element in tb_list corresponds to the line where the exception occurred
                filename, line_number, function_name, text = tb_list[-1]
                print(f"Exception occurred in {filename} at line {line_number} in function {function_name}: {text}")
                print(traceback.format_exc())
                sat_image_list_df.at[i,'shoreline_done'] = True
                sat_image_list_df.to_csv(sat_image_list_df_path_shore)        
                shoreline_i = i+1
                pass
            gc.collect()

def concat_gdfs_in_folder(folder):
    """
    Concatenates geojsons in the same folder into one geojson

    inputs:
    folder (str): path/to/folder/containing/geojsons

    outputs:
    final_gdf (gpd.GeoDataFrame): the concatenated geodataframe
    """
    geojsons = glob.glob(folder + '/*.geojson')
    gdfs = [None]*len(geojsons)
    for i in range(len(geojsons)):
        try:
            gdfs[i] = gpd.read_file(geojsons[i])
        except:
            gdfs[i] = None
    gdfs = remove_nones(gdfs)
    final_gdf = pd.concat(gdfs)
    return final_gdf

def merge_shorelines_section(g, c, rr, sss, r_home):
    """
    Merges individual shorelines into one geojson for an entire shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss

    ##establish file paths
    extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
    extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
    extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson') 
    reference_polygon_path = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline_path = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    shorelines_dir = os.path.join(section_dir, 'shorelines') 
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')

    ##concatenate individual shorelines into one file for each output (zoo rgb, nir threshold, swir threshold)
    zoo_gdf = concat_gdfs_in_folder(zoo_shoreline_dir)
    nir_gdf = concat_gdfs_in_folder(nir_shoreline_dir)
    swir_gdf = concat_gdfs_in_folder(swir_shoreline_dir)

    ##load in reference files
    reference_polygon_gdf = gpd.read_file(reference_polygon_path)
    reference_shoreline_gdf = gpd.read_file(reference_shoreline_path)

    ##clip to ref shoreline buffer and then reference polygon, need to explode so there aren't multilinestrings
    ##rgb zoo output
    zoo_gdf = gpd.clip(zoo_gdf, reference_shoreline_gdf['geometry'].iloc[0].buffer(ref_shore_buffer))
    zoo_gdf = gpd.clip(zoo_gdf, reference_polygon_gdf)
    zoo_gdf = zoo_gdf.explode(ignore_index=True)

    ##nir threshold output
    nir_gdf = gpd.clip(nir_gdf, reference_shoreline_gdf['geometry'].iloc[0].buffer(ref_shore_buffer))
    nir_gdf = gpd.clip(nir_gdf, reference_polygon_gdf)
    nir_gdf = nir_gdf.explode(ignore_index=True)

    ##swir threshold output
    swir_gdf = gpd.clip(swir_gdf, reference_shoreline_gdf['geometry'].iloc[0].buffer(ref_shore_buffer))
    swir_gdf = gpd.clip(swir_gdf, reference_polygon_gdf)
    swir_gdf = swir_gdf.explode(ignore_index=True)

    ##save each merged and clipped shorelines file, rgb zoo, nir, swir
    zoo_gdf.to_file(extracted_shorelines_path)
    nir_gdf.to_file(extracted_shorelines_nir_path)
    swir_gdf.to_file(extracted_shorelines_swir_path)

def resample_shorelines_section(g, c, rr, sss, r_home):
    """
    Resamples shorelines to a point every 10 m for a shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss

    ##establish file paths
    extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
    extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
    extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')

    ##load in gdfs and convert to utm (so we can resample in units of meters)
    extracted_shorelines_gdf = wgs84_to_utm_df(gpd.read_file(extracted_shorelines_path))
    extracted_shorelines_nir_gdf = wgs84_to_utm_df(gpd.read_file(extracted_shorelines_nir_path))
    extracted_shorelines_swir_gdf = wgs84_to_utm_df(gpd.read_file(extracted_shorelines_swir_path))

    ##rgb zoo output
    for idx,row in extracted_shorelines_gdf.iterrows():
        geom = row['geometry']
        new_geom = resample_line_by_distance(geom, 10)
        extracted_shorelines_gdf.at[idx,'geometry'] = new_geom

    ##nir threshold output
    for idx,row in extracted_shorelines_nir_gdf.iterrows():
        geom = row['geometry']
        new_geom = resample_line_by_distance(geom, 10)
        extracted_shorelines_nir_gdf.at[idx,'geometry'] = new_geom

    ##swir threshold output
    for idx,row in extracted_shorelines_swir_gdf.iterrows():
        geom = row['geometry']
        new_geom = resample_line_by_distance(geom, 10)
        extracted_shorelines_swir_gdf.at[idx,'geometry'] = new_geom

    ##drop nans from the gdfs
    extracted_shorelines_gdf = extracted_shorelines_gdf.dropna(subset=['geometry'])
    extracted_shorelines_nir_gdf = extracted_shorelines_nir_gdf.dropna(subset=['geometry'])
    extracted_shorelines_swir_gdf = extracted_shorelines_swir_gdf.dropna(subset=['geometry'])

    ##convert to wgs84 coordinate system
    extracted_shorelines_gdf = utm_to_wgs84_df(extracted_shorelines_gdf)
    extracted_shorelines_nir_gdf = utm_to_wgs84_df(extracted_shorelines_nir_gdf)
    extracted_shorelines_swir_gdf = utm_to_wgs84_df(extracted_shorelines_swir_gdf)
    
    ##save each file
    extracted_shorelines_gdf.to_file(extracted_shorelines_path)
    extracted_shorelines_nir_gdf.to_file(extracted_shorelines_nir_path)
    extracted_shorelines_swir_gdf.to_file(extracted_shorelines_swir_path)

def record_stats_shoreline_section(g, c, rr, sss, r_home):
    section_str = g+c+rr+sss
    section_dir = os.path.join(r_home, 'SSS'+sss)

    reprojected_points = os.path.join(section_dir, section_str+'_reprojected_points.geojson')
    transects = os.path.join(section_dir, section_str+'_transects.geojson')

    transects_gdf = gpd.read_file(transects)
    org_crs = transects_gdf.crs
    utm_crs = transects_gdf.estimate_utm_crs()

    transects_utm = transects_gdf.to_crs(utm_crs)

    points_gdf = gpd.read_file(reprojected_points)
    points_utm = points_gdf.to_crs(utm_crs)

    transect_ids = sorted(np.unique(points_gdf['transect_id']))

    min_vals = [None]*len(transect_ids)
    q1_vals = [None]*len(transect_ids)
    median_vals = [None]*len(transect_ids)
    mean_vals = [None]*len(transect_ids)
    q3_vals = [None]*len(transect_ids)
    max_vals = [None]*len(transect_ids)

    iqr_vals = [None]*len(transect_ids)
    mad_vals = [None]*len(transect_ids)
    std_vals = [None]*len(transect_ids)
    cv_vals = [None]*len(transect_ids)
    skew_vals = [None]*len(transect_ids)
    kurt_vals = [None]*len(transect_ids)

    geometry_from_centroids = [None]*len(transect_ids)
    transect_ids_ = [None]*len(transect_ids)

    geom_min_utm = [None]*len(transect_ids)
    geom_q1_utm = [None]*len(transect_ids)
    geom_median_utm = [None]*len(transect_ids)
    geom_mean_utm = [None]*len(transect_ids)
    geom_q3_utm = [None]*len(transect_ids)
    geom_max_utm = [None]*len(transect_ids)

    for i in range(len(transect_ids)):
        try:
            transect_id = transect_ids[i]

            transect = transects_utm[transects_utm['transect_id']==transect_id].reset_index(drop=True).iloc[0]
            first = transect.geometry.coords[0]
            last = transect.geometry.coords[1]
            angle = np.arctan2(last[1] - first[1], last[0] - first[0])

            pts_utm = points_utm[points_utm['transect_id']==transect_id]
            pts_orig = points_gdf[points_gdf['transect_id']==transect_id]

            centroid = pts_orig.geometry.centroid.iloc[0]

            x = pts_utm['cross_distance'].astype(float).values

            d_min = np.min(x)
            q1 = np.percentile(x, 25)
            med = np.median(x)
            mean_val = np.mean(x)
            q3 = np.percentile(x, 75)
            d_max = np.max(x)

            iqr = q3 - q1
            mad = np.median(np.abs(x - med))
            std = np.std(x, ddof=1)
            cv = std / mean_val if mean_val != 0 else np.nan
            skew = scipy.stats.skew(x, bias=False)
            kurt = scipy.stats.kurtosis(x, bias=False)

            geom_min_utm[i] = shapely.Point(first[0] + d_min*np.cos(angle),
                                            first[1] + d_min*np.sin(angle))
            geom_q1_utm[i] = shapely.Point(first[0] + q1*np.cos(angle),
                                           first[1] + q1*np.sin(angle))
            geom_median_utm[i] = shapely.Point(first[0] + med*np.cos(angle),
                                               first[1] + med*np.sin(angle))
            geom_mean_utm[i] = shapely.Point(first[0] + mean_val*np.cos(angle),
                                             first[1] + mean_val*np.sin(angle))
            geom_q3_utm[i] = shapely.Point(first[0] + q3*np.cos(angle),
                                           first[1] + q3*np.sin(angle))
            geom_max_utm[i] = shapely.Point(first[0] + d_max*np.cos(angle),
                                            first[1] + d_max*np.sin(angle))

            geometry_from_centroids[i] = centroid

            min_vals[i] = d_min
            q1_vals[i] = q1
            median_vals[i] = med
            mean_vals[i] = mean_val
            q3_vals[i] = q3
            max_vals[i] = d_max

            iqr_vals[i] = iqr
            mad_vals[i] = mad
            std_vals[i] = std
            cv_vals[i] = cv
            skew_vals[i] = skew
            kurt_vals[i] = kurt

            transect_ids_[i] = transect_id

        except:
            continue

    def clean(x): return remove_nones(x)

    geometry_from_centroids = clean(geometry_from_centroids)
    transect_ids_ = clean(transect_ids_)

    min_vals = clean(min_vals)
    q1_vals = clean(q1_vals)
    median_vals = clean(median_vals)
    mean_vals = clean(mean_vals)
    q3_vals = clean(q3_vals)
    max_vals = clean(max_vals)

    iqr_vals = clean(iqr_vals)
    mad_vals = clean(mad_vals)
    std_vals = clean(std_vals)
    cv_vals = clean(cv_vals)
    skew_vals = clean(skew_vals)
    kurt_vals = clean(kurt_vals)

    geom_min = gpd.GeoSeries(clean(geom_min_utm), crs=utm_crs).to_crs(org_crs)
    geom_q1 = gpd.GeoSeries(clean(geom_q1_utm), crs=utm_crs).to_crs(org_crs)
    geom_median = gpd.GeoSeries(clean(geom_median_utm), crs=utm_crs).to_crs(org_crs)
    geom_mean = gpd.GeoSeries(clean(geom_mean_utm), crs=utm_crs).to_crs(org_crs)
    geom_q3 = gpd.GeoSeries(clean(geom_q3_utm), crs=utm_crs).to_crs(org_crs)
    geom_max = gpd.GeoSeries(clean(geom_max_utm), crs=utm_crs).to_crs(org_crs)

    def save_points(name, geom, vals):
        gdf = gpd.GeoDataFrame(
            {
                'transect_id': transect_ids_,
                **vals,
                'geometry_from_centroids': geometry_from_centroids,
            },
            geometry=geom,
            crs=org_crs
        )
        gdf.to_file(os.path.join(section_dir, f"{section_str}_{name}_shoreline_points.geojson"))

        line = shapely.geometry.LineString(geom)
        gpd.GeoDataFrame(
            {'G':[g], 'C':[c], 'RR':[rr], 'SSS':[sss]},
            geometry=[line],
            crs=org_crs
        ).to_file(os.path.join(section_dir, f"{section_str}_{name}_shoreline.geojson"))

    save_points("min", geom_min, {'cross_distance_min': min_vals})
    save_points("q1", geom_q1, {'cross_distance_q1': q1_vals})
    save_points("median", geom_median, {
        'cross_distance_median': median_vals,
        'iqr': iqr_vals,
        'q1': q1_vals,
        'q3': q3_vals,
        'mad': mad_vals,
        'std': std_vals,
        'mean': mean_vals,
        'cv': cv_vals,
        'skewness': skew_vals,
        'kurtosis': kurt_vals,
    })
    save_points("mean", geom_mean, {'cross_distance_mean': mean_vals})
    save_points("q3", geom_q3, {'cross_distance_q3': q3_vals})
    save_points("max", geom_max, {'cross_distance_max': max_vals})
    print('record stats computed')

def spatial_kde_section(g, c, rr, sss, r_home):
    """
    Computes spatial kde by shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    crs = reference_shoreline_gdf.crs
    kde_path = os.path.join(section_dir, section_string + '_spatial_kde.tif')
    extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
    extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
    extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')
    extracted_shorelines_path_filter = os.path.join(section_dir, section_string + '_extracted_shorelines_filter.geojson')
    extracted_shorelines_nir_path_filter = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh_filter.geojson')
    extracted_shorelines_swir_path_filter = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh_filter.geojson')
    transects = os.path.join(section_dir, section_string + '_transects.geojson')
    otsu_path = os.path.join(section_dir, section_string + '_spatial_kde_otsu.tif')
    otsu_geojson = os.path.join(section_dir, section_string + '_spatial_kde_otsu.geojson')
    
    if os.path.isfile(kde_path)==False and os.path.isfile(extracted_shorelines_path)==True and os.path.isfile(extracted_shorelines_nir_path)==True and os.path.isfile(extracted_shorelines_swir_path)==True:
        shorelines_concat = pd.concat([gpd.read_file(extracted_shorelines_path),
                                        gpd.read_file(extracted_shorelines_nir_path),
                                        gpd.read_file(extracted_shorelines_swir_path)])
        shoreline_change_envelope.point_density_grid(shorelines_concat, 
                                                     kde_path,
                                                     15)
    if os.path.isfile(kde_path)==True:
        shoreline_change_envelope.compute_otsu_threshold(kde_path, otsu_path)
        shoreline_change_envelope.binary_raster_to_vector(otsu_path, otsu_geojson)
        otsu_geojson_gdf = gpd.read_file(otsu_geojson)
        utm_crs = otsu_geojson_gdf.crs
        zoo_shorelines_gdf = gpd.read_file(extracted_shorelines_path)
        wgs84_crs = zoo_shorelines_gdf.crs
        swir_shorelines_gdf = gpd.read_file(extracted_shorelines_swir_path)
        nir_shorelines_gdf = gpd.read_file(extracted_shorelines_nir_path)

        zoo_shorelines_gdf = zoo_shorelines_gdf.to_crs(utm_crs)
        swir_shorelines_gdf = swir_shorelines_gdf.to_crs(utm_crs)
        nir_shorelines_gdf = nir_shorelines_gdf.to_crs(utm_crs)

        zoo_shorelines_gdf = gpd.clip(zoo_shorelines_gdf, otsu_geojson_gdf).to_crs(wgs84_crs)
        swir_shorelines_gdf = gpd.clip(swir_shorelines_gdf, otsu_geojson_gdf).to_crs(wgs84_crs)
        nir_shorelines_gdf = gpd.clip(nir_shorelines_gdf, otsu_geojson_gdf).to_crs(wgs84_crs)

        zoo_shorelines_gdf = zoo_shorelines_gdf.explode()
        zoo_shorelines_gdf['type'] = zoo_shorelines_gdf['geometry'].type
        zoo_shorelines_gdf = zoo_shorelines_gdf[zoo_shorelines_gdf['type']=='LineString'].reset_index(drop=True)
        for idx,row in zoo_shorelines_gdf.iterrows():
            geom_arr = LineString_to_arr(row['geometry'])
            filtered_geom = geom_arr[5:-5]
            if len(filtered_geom)<5:
                filtered_geom=None
            else:
                filtered_geom = arr_to_LineString(filtered_geom)
            zoo_shorelines_gdf.at[idx,'geometry'] = filtered_geom
        zoo_shorelines_gdf = zoo_shorelines_gdf.dropna(subset='geometry').reset_index(drop=True)
        zoo_shorelines_gdf['year'] = pd.to_datetime(zoo_shorelines_gdf['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True).dt.year
        zoo_shorelines_gdf = vertex_filter(zoo_shorelines_gdf)

        swir_shorelines_gdf = swir_shorelines_gdf.explode()
        swir_shorelines_gdf['type'] = swir_shorelines_gdf['geometry'].type
        swir_shorelines_gdf = swir_shorelines_gdf[swir_shorelines_gdf['type']=='LineString'].reset_index(drop=True)
        for idx,row in swir_shorelines_gdf.iterrows():
            geom_arr = LineString_to_arr(row['geometry'])
            filtered_geom = geom_arr[5:-5]
            if len(filtered_geom)<5:
                filtered_geom=None
            else:
                filtered_geom = arr_to_LineString(filtered_geom)
            swir_shorelines_gdf.at[idx,'geometry'] = filtered_geom
        swir_shorelines_gdf = swir_shorelines_gdf.dropna(subset='geometry').reset_index(drop=True)
        swir_shorelines_gdf['year'] = pd.to_datetime(swir_shorelines_gdf['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True).dt.year
        swir_shorelines_gdf = vertex_filter(swir_shorelines_gdf)

        nir_shorelines_gdf = nir_shorelines_gdf.explode()
        nir_shorelines_gdf['type'] = nir_shorelines_gdf['geometry'].type
        nir_shorelines_gdf = nir_shorelines_gdf[nir_shorelines_gdf['type']=='LineString'].reset_index(drop=True)
        for idx,row in nir_shorelines_gdf.iterrows():
            geom_arr = LineString_to_arr(row['geometry'])
            filtered_geom = geom_arr[5:-5]
            if len(filtered_geom)<5:
                filtered_geom=None
            else:
                filtered_geom = arr_to_LineString(filtered_geom)
            nir_shorelines_gdf.at[idx,'geometry'] = filtered_geom
        nir_shorelines_gdf = nir_shorelines_gdf.dropna(subset='geometry').reset_index(drop=True)
        nir_shorelines_gdf['year'] = pd.to_datetime(nir_shorelines_gdf['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True).dt.year
        nir_shorelines_gdf = vertex_filter(nir_shorelines_gdf)

        zoo_shorelines_gdf.to_file(extracted_shorelines_path_filter)
        swir_shorelines_gdf.to_file(extracted_shorelines_swir_path_filter)
        nir_shorelines_gdf.to_file(extracted_shorelines_nir_path_filter)



    if os.path.isfile(kde_path)==True and os.path.isfile(extracted_shorelines_path)==True:
        try:
            sample_spatial_kde(kde_path,
                               extracted_shorelines_path,
                               crs)
        except Exception as e:
            print(e)
            pass
        try:
            sample_spatial_kde(kde_path,
                               extracted_shorelines_nir_path,
                               crs)
        except Exception as e:
            print(e)
            pass
        try:
            sample_spatial_kde(kde_path,
                               extracted_shorelines_swir_path,
                               crs)
        except Exception as e:
            print(e)
            pass

def get_trends_section(g, c, rr, sss, r_home):
    """
    Computes trends for shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    crs = reference_shoreline_gdf.crs
    extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
    extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
    extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')
    transects = os.path.join(section_dir, section_string + '_transects.geojson')
    if os.path.isfile(extracted_shorelines_path)==True and os.path.isfile(extracted_shorelines_nir_path)==True and os.path.isfile(extracted_shorelines_swir_path)==True: 
        transect_trends = os.path.join(section_dir, section_string+'_transects_trends.geojson')
        get_trends(os.path.join(section_dir, section_string+'_resampled_tidally_corrected_transect_time_series_merged.csv'),
                transects,
                transect_trends,
                )
        gdf = gpd.read_file(transect_trends)
        gdf['significant'] = np.abs(gdf['linear_trend'])>gdf['linear_trend_95_confidence']
        gdf['significant'] = gdf['significant'].astype(int)
        gdf.to_file(transect_trends)
 
def transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=False):
    """
    Computes transect and shoreline intersections by shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    crs = reference_shoreline_gdf.crs
    if waterline_filter == True:
        extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines_filter.geojson')
        extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh_filter.geojson')
        extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh_filter.geojson')
    else:
        extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
        extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
        extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')      

    transects = os.path.join(section_dir, section_string + '_transects.geojson')

    if os.path.isfile(extracted_shorelines_path)==True:
        transect_timeseries(extracted_shorelines_path,
                            transects,
                            reference_polygon,
                            crs,
                            os.path.join(section_dir, section_string+'_raw_transect_time_series_merged.csv'),
                            os.path.join(section_dir, section_string+'_raw_transect_time_series_matrix.csv'))
        transect_timeseries(extracted_shorelines_nir_path,
                            transects,
                            reference_polygon,
                            crs,
                            os.path.join(section_dir, section_string+'_raw_transect_time_series_merged_nir_thresh.csv'),
                            os.path.join(section_dir, section_string+'_raw_transect_time_series_matrix_nir_thresh.csv'))
        transect_timeseries(extracted_shorelines_swir_path,
                            transects,
                            reference_polygon,
                            crs,
                            os.path.join(section_dir, section_string+'_raw_transect_time_series_merged_swir_thresh.csv'),
                            os.path.join(section_dir, section_string+'_raw_transect_time_series_matrix_swir_thresh.csv'))

def get_distance_to_elevation_contours_section(g, c, rr, sss, r_home):
    """
    Computes point to contour distance for shorelines
    Rescales between 0 and 1 as distance score for each point
    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    crs = reference_shoreline_gdf.crs
    extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
    extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
    extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')
    extracted_shorelines = gpd.read_file(extracted_shorelines_path).to_crs(crs)
    extracted_shorelines_nir = gpd.read_file(extracted_shorelines_nir_path).to_crs(crs)
    extracted_shorelines_swir = gpd.read_file(extracted_shorelines_swir_path).to_crs(crs)

    crest_lines_path = os.path.join(section_dir, 'elevation_profile_lines', section_string + '_crest_points_smooth.geojson')
    inflection_lines_path = os.path.join(section_dir, 'elevation_profile_lines', section_string + '_inflection_points_smooth.geojson')
    toe_lines_path = os.path.join(section_dir, 'elevation_profile_lines', section_string + '_toe_points_smooth.geojson')
    crest_lines = gpd.read_file(crest_lines_path).to_crs(crs)
    inflection_lines = gpd.read_file(inflection_lines_path).to_crs(crs)
    toe_lines = gpd.read_file(toe_lines_path).to_crs(crs)
    lines_concat = reference_shoreline_gdf
    extracted_shorelines_points = convert_linestrings_to_multipoints(extracted_shorelines).explode(index_parts=True)
    extracted_shorelines_nir_points = convert_linestrings_to_multipoints(extracted_shorelines_nir).explode(index_parts=True)
    extracted_shorelines_swir_points = convert_linestrings_to_multipoints(extracted_shorelines_swir).explode(index_parts=True)

    extracted_shorelines_points_sjoin = extracted_shorelines_points.sjoin_nearest(lines_concat, distance_col='dist')
    extracted_shorelines_nir_points_sjoin = extracted_shorelines_nir_points.sjoin_nearest(lines_concat, distance_col='dist')
    extracted_shorelines_swir_points_sjoin = extracted_shorelines_swir_points.sjoin_nearest(lines_concat, distance_col='dist')

    extracted_shorelines_points_sjoin = utm_to_wgs84_df(extracted_shorelines_points_sjoin)
    extracted_shorelines_nir_points_sjoin = utm_to_wgs84_df(extracted_shorelines_nir_points_sjoin)
    extracted_shorelines_swir_points_sjoin = utm_to_wgs84_df(extracted_shorelines_swir_points_sjoin)

    extracted_shorelines_points_sjoin['dist_score'] = 1-rescale(extracted_shorelines_points_sjoin['dist'])
    extracted_shorelines_nir_points_sjoin['dist_score'] = 1-rescale(extracted_shorelines_nir_points_sjoin['dist'])
    extracted_shorelines_swir_points_sjoin['dist_score'] = 1-rescale(extracted_shorelines_swir_points_sjoin['dist'])

    extracted_shorelines_points_sjoin.to_file(os.path.join(section_dir, section_string+'_extracted_shorelines_points.geojson'))
    extracted_shorelines_nir_points_sjoin.to_file(os.path.join(section_dir, section_string+'_extracted_shorelines_nir_points.geojson'))
    extracted_shorelines_swir_points_sjoin.to_file(os.path.join(section_dir, section_string+'_extracted_shorelines_swir_points.geojson'))

def get_tide_data_section(g, c, rr, sss, r_home):
    """
    Gathers tide data for each section
    pulling from end point of transect and each unique timestamp in the extracted shoreline data

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/
    """
    section_dir = os.path.join(r_home, 'SSS'+sss)
    section_string = g+c+rr+sss
    transects = os.path.join(section_dir, section_string+'_transects.geojson')
    tide_model_dir = os.path.join(get_script_path(), 'CoastSeg', 'tide_model')
    timeseries_zoo = os.path.join(section_dir, section_string+'_raw_transect_time_series_merged.csv')
    timeseries_swir = os.path.join(section_dir, section_string+'_raw_transect_time_series_merged_swir_thresh.csv')
    timeseries_nir = os.path.join(section_dir, section_string+'_raw_transect_time_series_merged_nir_thresh.csv')
    timeseries_zoo_df = pd.read_csv(timeseries_zoo)
    timeseries_swir_df = pd.read_csv(timeseries_swir)
    timeseries_nir_df = pd.read_csv(timeseries_nir)
    tide_data_path = os.path.join(section_dir, section_string+'_tides.csv')
    zoo_dates = timeseries_zoo_df['dates']
    swir_dates = timeseries_swir_df['dates']
    nir_dates = timeseries_nir_df['dates']
    dates_concat = pd.concat([zoo_dates,swir_dates,nir_dates])
    unique_dates = np.unique(dates_concat)
    unique_dates = pd.to_datetime(unique_dates, format='%Y-%m-%d-%H-%M-%S', utc=True).values
    transects_gdf = gpd.read_file(transects)
    end_coords = transects_gdf['geometry'].apply(lambda line: Point(line.coords[-1]))
    transect_ids = transects_gdf['transect_id']
    x = end_coords.x
    y = end_coords.y
    transects_gdf['x'] = x
    transects_gdf['y'] = y
    tide_data = tide_correction.model_tides(x,
                                            y,
                                            unique_dates,
                                            model="FES2022",
                                            directory=tide_model_dir,
                                            epsg=4326,
                                            method="bilinear",
                                            extrapolate=True,
                                            cutoff=10.0)

    tides = pd.merge(tide_data, transects_gdf, how='left', on=['x','y'])
    tides.to_csv(tide_data_path)

def prep_slope_data(g,
                    c,
                    rr,
                    sss,
                    r_home,
                    constant=False,
                    smooth=True,
                    dem=''):

    section_dir = os.path.join(r_home, 'SSS'+sss)
    section_string = g+c+rr+sss
    transects_path = os.path.join(section_dir, g+c+rr+sss+'_transects.geojson')
    slope_data_path = os.path.join(section_dir, g+c+rr+sss+'_transects_slopes_'+dem+'.geojson')

    if not np.isnan(constant):
        
        transects_gdf = gpd.read_file(transects_path)
        transects_gdf['avg_slope'] = [slope_value]*len(transects_gdf)
        transects_gdf['avg_slope_cleaned'] = [slope_value]*len(transects_gdf)
        transects_gdf.to_file(slope_data_path)
    else:
        try:
            slope_data = gpd.read_file(slope_data_path)
            ###making slopes that are less than 1/100 equal to 1/100 and doing a moving average every 15 transects or 750 m
            if smooth==True:
                print('smoothing slopes and re-assigning slopes less than 1/100 to 1/100')

                slope_data['avg_slope_cleaned'] = moving_average_with_edge_padding(slope_data['avg_slope'], 15)
                slope_data['avg_slope_cleaned'] = slope_data['avg_slope_cleaned'].where(slope_data['avg_slope_cleaned'] > 1/100, 1/100)
                print(slope_data.head())
            else:
                print('re-assigning slopes less than 1/100 to 1/100')
                slope_data['avg_slope_cleaned'] = slope_data['avg_slope']
                slope_data['avg_slope_cleaned'] = slope_data['avg_slope_cleaned'].where(slope_data['avg_slope_cleaned'] > 1/100, 1/100)
                print(slope_data.head())
            slope_data.to_file(slope_data_path)
        except:
            print('slope data not available')
        
def tidally_correct_section(g, 
                            c, 
                            rr, 
                            sss, 
                            r_home, 
                            dem='',
                            reference_elevation=0):
    """
    Tidally corrects shoreline data within a shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/subregion
    reference elevation (float): reference elevation for correction
    """
    section_dir = os.path.join(r_home, 'SSS'+sss)
    section_string = g+c+rr+sss
    timeseries_zoo = pd.read_csv(os.path.join(section_dir, section_string+'_raw_transect_time_series_merged.csv'))
    timeseries_zoo['dates'] = pd.to_datetime(timeseries_zoo['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
    timeseries_swir = pd.read_csv(os.path.join(section_dir, section_string+'_raw_transect_time_series_merged_swir_thresh.csv'))
    timeseries_swir['dates'] = pd.to_datetime(timeseries_swir['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
    timeseries_nir = pd.read_csv(os.path.join(section_dir, section_string+'_raw_transect_time_series_merged_nir_thresh.csv'))
    timeseries_nir['dates'] = pd.to_datetime(timeseries_nir['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
    tide_data = pd.read_csv(os.path.join(section_dir, section_string+'_tides.csv'))
    tide_data['dates'] = pd.to_datetime(tide_data['dates'], utc=True)
    slope_data_path = os.path.join(section_dir, section_string+'_transects_slopes_'+dem+'.geojson')
    slope_data = gpd.read_file(slope_data_path)

    ##join tide data
    timeseries_zoo_merge = pd.merge(timeseries_zoo, tide_data, how='left', on = ['transect_id', 'dates'])
    timeseries_swir_merge = pd.merge(timeseries_swir, tide_data, how='left', on= ['transect_id', 'dates'])
    timeseries_nir_merge = pd.merge(timeseries_nir, tide_data, how='left', on=['transect_id', 'dates'])
    
    ##get rid of columns
    keep_cols = ['dates', 'image_suitability_score', 'segmentation_suitability_score',
       'satname', 'kde_value','transect_id', 'intersect_x', 'intersect_y',
       'cross_distance', 'x', 'y', 'tide',
       'G', 'C', 'RR', 'SSS']

    for col in timeseries_zoo_merge.columns:
        if col not in keep_cols:
            try:
                timeseries_zoo_merge = timeseries_zoo_merge.drop(columns=[col])
            except:
                pass
    for col in timeseries_swir_merge.columns:
        if col not in keep_cols:
            try:
                timeseries_swir_merge = timeseries_swir_merge.drop(columns=[col])
            except:
                pass
    for col in timeseries_nir_merge.columns:
        if col not in keep_cols:
            try:
                timeseries_nir_merge = timeseries_nir_merge.drop(columns=[col])
            except:
                pass
    
    ##some cleaning
    timeseries_zoo_merge['G'] = timeseries_zoo_merge['G'].astype('str')
    timeseries_zoo_merge['C'] = timeseries_zoo_merge['C'].astype('str')
    timeseries_zoo_merge['RR'] = timeseries_zoo_merge['RR'].astype('str')
    timeseries_zoo_merge['SSS'] = timeseries_zoo_merge['SSS'].astype('str')
    timeseries_zoo_merge['transect_id'] = timeseries_zoo_merge['transect_id'].astype('str')

    ##some cleaning
    timeseries_swir_merge['G'] = timeseries_swir_merge['G'].astype('str')
    timeseries_swir_merge['C'] = timeseries_swir_merge['C'].astype('str')
    timeseries_swir_merge['RR'] = timeseries_swir_merge['RR'].astype('str')
    timeseries_swir_merge['SSS'] = timeseries_swir_merge['SSS'].astype('str')
    timeseries_swir_merge['transect_id'] = timeseries_swir_merge['transect_id'].astype('str')

    ##some cleaning
    timeseries_nir_merge['G'] = timeseries_nir_merge['G'].astype('str')
    timeseries_nir_merge['C'] = timeseries_nir_merge['C'].astype('str')
    timeseries_nir_merge['RR'] = timeseries_nir_merge['RR'].astype('str')
    timeseries_nir_merge['SSS'] = timeseries_nir_merge['SSS'].astype('str')
    timeseries_nir_merge['transect_id'] = timeseries_nir_merge['transect_id'].astype('str')

    ##some cleaning
    slope_data['G'] = slope_data['G'].astype(str)
    slope_data['C'] = slope_data['C'].astype(str)
    slope_data['RR'] = slope_data['RR'].astype(str)
    slope_data['SSS'] = slope_data['SSS'].astype(str)
    slope_data['transect_id'] = slope_data['transect_id'].astype(str)

    ##join slope data
    timeseries_zoo_merge = pd.merge(timeseries_zoo_merge, slope_data, how='left', on=['transect_id'])
    timeseries_swir_merge = pd.merge(timeseries_swir_merge, slope_data, how='left', on=['transect_id'])
    timeseries_nir_merge = pd.merge(timeseries_nir_merge, slope_data, how='left', on=['transect_id'])
    keep_cols = ['dates','image_suitability_score', 'segmentation_suitability_score',
       'satname', 'transect_id', 'intersect_x', 'intersect_y',
       'cross_distance', 'x', 'y', 'tide', 'avg_slope_cleaned', 'kde_value']

    ##getting rid of columns
    for col in timeseries_zoo_merge.columns:
        if col not in keep_cols:
            try:
                timeseries_zoo_merge = timeseries_zoo_merge.drop(columns=[col])
            except:
                pass
    for col in timeseries_swir_merge.columns:
        if col not in keep_cols:
            try:
                timeseries_swir_merge = timeseries_swir_merge.drop(columns=[col])
            except:
                pass
    for col in timeseries_nir_merge.columns:
        if col not in keep_cols:
            try:
                timeseries_nir_merge = timeseries_nir_merge.drop(columns=[col])
            except:
                pass

    ##apply corrections
    timeseries_zoo_merge['cross_distance_tidally_corrected'] = timeseries_zoo_merge['cross_distance']-(timeseries_zoo_merge['tide']-reference_elevation)/timeseries_zoo_merge['avg_slope_cleaned']
    timeseries_swir_merge['cross_distance_tidally_corrected'] = timeseries_swir_merge['cross_distance']-(timeseries_swir_merge['tide']-reference_elevation)/timeseries_swir_merge['avg_slope_cleaned']
    timeseries_nir_merge['cross_distance_tidally_corrected'] = timeseries_nir_merge['cross_distance']-(timeseries_nir_merge['tide']-reference_elevation)/timeseries_nir_merge['avg_slope_cleaned']

    ##save merged files
    timeseries_zoo_merge.to_csv(os.path.join(section_dir, section_string+'_tidally_corrected_transect_time_series_merged.csv'))
    timeseries_swir_merge.to_csv(os.path.join(section_dir, section_string+'_tidally_corrected_transect_time_series_merged_swir_thresh.csv'))
    timeseries_nir_merge.to_csv(os.path.join(section_dir, section_string+'_tidally_corrected_transect_time_series_merged_nir_thresh.csv'))

def clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=False):
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home,'SSS'+ sss)
    section_string = g+c+rr+sss
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    if planet==True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path)==True:
            sat_image_list_df_path_clipped = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_clip.csv')
        else:
            print('no planet data')
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_clipped = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_clip.csv')
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path_clipped)
    except:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
        sat_image_list_df['clipped'] = [False]*len(sat_image_list_df)

    shorelines_dir = os.path.join(section_dir, 'shorelines') 
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
    
    ##working with the tifs
    num_images = len(sat_image_list_df)
    ##loop over each image
    if len(sat_image_list_df)>0:
        for i in range(len(sat_image_list_df['analysis_image'])):
            try:
                print(i/num_images*100)
                image = sat_image_list_df['analysis_image'].iloc[i]
                roi_folder = sat_image_list_df['roi_folder'].iloc[i]
                image_suitability_score = sat_image_list_df['model_scores'].iloc[i]
                image_score = image_suitability_score
                if image_suitability_score<0.335:
                    sat_image_list_df['clipped'].iloc[i] = True
                    sat_image_list_df.to_csv(sat_image_list_df_path_clipped)
                    continue
                if image==None:
                    sat_image_list_df['clipped'].iloc[i] = True
                    sat_image_list_df.to_csv(sat_image_list_df_path_clipped)
                    continue
                if sat_image_list_df['clipped'].iloc[i] == True:
                    sat_image_list_df.to_csv(sat_image_list_df_path_clipped)
                    continue        
                satname = sat_image_list_df['satnames'].iloc[i]
                date = sat_image_list_df['datetimes_utc'].iloc[i]
                
                ##loading the tif
                with rasterio.open(image) as src:
                    if satname != 'PS':
                        nir = src.read(4)
                        seg_lab = src.read(6)
                        binary_image_nir = src.read(7)
                        binary_image_swir = src.read(8)
                    else:
                        nir = src.read(4)
                        seg_lab = src.read(5)
                        binary_image_nir = src.read(6)

                    bounds = src.bounds
                    resolution = src.res 
                    width = src.width
                    height = src.height
                    mask_value = src.meta['nodata']
                    transform = src.transform
                    count = src.count
                    crs = src.crs
                    xmin = bounds.left
                    ymax = bounds.top
                    x_res = resolution[0]
                    y_res = resolution[1]
                    
                    ##no data mask
                    if mask_value is not None:
                        mask = nir != mask_value
                    else:
                        mask = None

                    data_polygon = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for i, (s, v) 
                        in enumerate(
                            shapes(nir, mask=mask, transform=src.transform)))
                    data_polygon = gpd.GeoDataFrame.from_features(list(data_polygon), crs=src.crs)

                    if mask_value is not None:
                        mask = nir == mask_value
                    else:
                        mask = None
                        
                    no_data_polygon = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for i, (s, v) 
                        in enumerate(
                            shapes(nir, mask=mask, transform=src.transform)))   
                    try:
                        no_data_polygon = gpd.GeoDataFrame.from_features(list(no_data_polygon), crs=src.crs)
                    except:
                        no_data_polygon = None

                ##making no data polygon
                try:
                    no_data_polygon = no_data_polygon.buffer(x_res*2).unary_union()
                    data_polygon = data_polygon.unary_union().difference(no_data_polygon)
                except:
                    data_polygon = None
                date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00")
                date = date.strftime("%Y-%m-%d-%H-%M-%S")
                zoo_shoreline_path = os.path.join(zoo_shoreline_dir, date+'_'+satname+'_'+roi_folder+'.geojson')
                if os.path.isfile(zoo_shoreline_path):
                    clip_extracted_shoreline(zoo_shoreline_path, data_polygon, reference_shoreline, reference_polygon, ref_shore_buffer)
                nir_shoreline_path = os.path.join(nir_shoreline_dir, date+'_'+satname+'_'+roi_folder+'.geojson')
                if os.path.isfile(nir_shoreline_path):
                    clip_extracted_shoreline(nir_shoreline_path, data_polygon, reference_shoreline, reference_polygon, ref_shore_buffer)
                swir_shoreline_path = os.path.join(swir_shoreline_dir, date+'_'+satname+'_'+roi_folder+'.geojson')
                if os.path.isfile(swir_shoreline_path):
                    clip_extracted_shoreline(swir_shoreline_path, data_polygon, reference_shoreline, reference_polygon, ref_shore_buffer)
                sat_image_list_df['clipped'].iloc[i] = True
                sat_image_list_df.to_csv(sat_image_list_df_path_clipped)
            except Exception as e:
                print(e)
                sat_image_list_df['clipped'].iloc[i] = True
                sat_image_list_df.to_csv(sat_image_list_df_path_clipped)
                pass
            gc.collect()

def rename_transects_section(g, c, rr, sss, r_home):
    version_name = '0'
    section_dir = os.path.join(r_home, 'SSS'+sss)
    transects_path = os.path.join(section_dir, g+c+rr+sss+'_transects.geojson')
    generate_transects.rename_transects(g, c, rr, sss, version_name, transects_path)

def make_transects_section(g, c, rr, sss, r_home):
    section = 'SSS'+sss
    section_dir = os.path.join(r_home, section)
    section_string = g + c + rr + sss
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    transects = os.path.join(section_dir, section_string + '_transects_new.geojson')
    transect_spacing = 50
    transect_length = 800
    version_name = 0

    generate_transects.make_transects_section_mod(reference_shoreline,
                          reference_polygon,
                          transects,
                          g,
                          c,
                          rr,
                          sss,
                          version_name,
                          transect_spacing,
                          transect_length)

def filter_shorelines_section(g, c, rr, sss, r_home):
    section = 'SSS'+sss
    section_dir = os.path.join(r_home, section)
    section_string = g + c + rr + sss

    ##establish shorelines files
    shorelines_zoo = os.path.join(section_dir, section_string+'_extracted_shorelines.geojson')
    shorelines_swir = os.path.join(section_dir, section_string+'_extracted_shorelines_swir_thresh.geojson')
    shorelines_nir = os.path.join(section_dir, section_string+'_extracted_shorelines_nir_thresh.geojson')

    ##establish filtered files
    shorelines_zoo_filter = os.path.join(section_dir, section_string+'_extracted_shorelines_filter.geojson')
    shorelines_swir_filter = os.path.join(section_dir, section_string+'_extracted_shorelines_swir_thresh_filter.geojson')
    shorelines_nir_filter = os.path.join(section_dir, section_string+'_extracted_shorelines_nir_thresh_filter.geojson')  

    shorelines_zoo_gdf = gpd.read_file(shorelines_zoo)
    shorelines_swir_gdf = gpd.read_file(shorelines_swir)
    shorelines_nir_gdf = gpd.read_file(shorelines_nir)

    shorelines_zoo_gdf = shorelines_zoo_gdf[shorelines_zoo_gdf['image_suitability_score']>=0.9].reset_index(drop=True)
    shorelines_zoo_gdf = shorelines_zoo_gdf[shorelines_zoo_gdf['kde_value']>0].reset_index(drop=True)

    shorelines_nir_gdf = shorelines_nir_gdf[shorelines_nir_gdf['image_suitability_score']>=0.9].reset_index(drop=True)
    shorelines_nir_gdf = shorelines_nir_gdf[shorelines_nir_gdf['kde_value']>0].reset_index(drop=True)

    shorelines_swir_gdf = shorelines_swir_gdf[shorelines_swir_gdf['image_suitability_score']>=0.9].reset_index(drop=True)
    shorelines_swir_gdf = shorelines_swir_gdf[shorelines_swir_gdf['kde_value']>0].reset_index(drop=True)

    shorelines_zoo_gdf.to_file(shorelines_zoo_filter)
    shorelines_nir_gdf.to_file(shorelines_nir_filter)
    shorelines_swir_gdf.to_file(shorelines_swir_filter)

def filter_timeseries_section(g, c, rr, sss, r_home, resample_freq='365D'):
    section = 'SSS'+sss
    section_dir = os.path.join(r_home, section)
    section_string = g + c + rr + sss
    transects = os.path.join(section_dir, section_string+'_transects.geojson')

    ##establish merged files
    timeseries_zoo_merge = os.path.join(section_dir, section_string+'_tidally_corrected_transect_time_series_merged.csv')
    timeseries_swir_merge = os.path.join(section_dir, section_string+'_tidally_corrected_transect_time_series_merged_swir_thresh.csv')
    timeseries_nir_merge = os.path.join(section_dir, section_string+'_tidally_corrected_transect_time_series_merged_nir_thresh.csv')

    ##filtering and ensembling
    make_transect_csvs.save_csv_per_id(timeseries_zoo_merge,
                                       timeseries_nir_merge,
                                       timeseries_swir_merge,
                                       'ensemble',
                                       transects,
                                       section_string,
                                       resample_freq=resample_freq
                                      )

def get_slopes_section(g, c, rr, sss, r_home, arctic_dem, custom_dem):
    section = 'SSS'+sss
    section_dir = os.path.join(r_home, section)
    section_string = g + c + rr + sss
    dem_to_beach_slope.profile_shoreline_section(g,
                            c,
                            rr,
                            '0',
                            r_home,
                            all_sections=False,
                            custom_sections = ['SSS'+sss],
                            arctic_dem=arctic_dem,
                            custom_dem=custom_dem,
                            vertical_datum='')

def merge_files_region(g, c, rr, r_home):
    sections = sorted(get_immediate_subdirectories(r_home))
    trends_list = [None]*len(sections)
    points_list = [None]*len(sections)
    ref_poly_list = [None]*len(sections)
    ref_shore_list = [None]*len(sections)
    transects_list = [None]*len(sections)
    kde_poly_list = [None]*len(sections)
    rois_list = [None]*len(sections)
    esi_list = [None]*len(sections)
    slopes_list = [None]*len(sections)
    crest1_list = [None]*len(sections)
    crest2_list = [None]*len(sections)
    crest3_list = [None]*len(sections)
    ip_list = [None]*len(sections)
    toe_list = [None]*len(sections)
    mean_shoreline_points_list = [None]*len(sections)
    mean_shoreline_list = [None]*len(sections)

    i=0
    for section in sections:
        print(section)
        sss = section[3:]
        section_dir = os.path.join(r_home, section)
        section_str = g+c+rr+sss
        trends = os.path.join(section_dir, section_str+'_transects_trends.geojson')
        points = os.path.join(section_dir, section_str+'_reprojected_points.geojson')
        ref_poly = os.path.join(section_dir, section_str+'_reference_polygon.geojson')
        ref_shore = os.path.join(section_dir, section_str+'_reference_shoreline.geojson')
        transects = os.path.join(section_dir, section_str+'_transects.geojson')
        kde_poly = os.path.join(section_dir, section_str+'_spatial_kde_otsu.geojson')
        rois = os.path.join(section_dir, section_str+'_rois.geojson')
        slopes = os.path.join(section_dir, section_str+'_transects_slopes_.geojson')
        esi = os.path.join(section_dir, section_str+'_esi_transects.geojson')
        crest1 = os.path.join(section_dir, 'elevation_profile_lines_',section_str+'_crest_points_smooth.geojson')
        crest2 = os.path.join(section_dir, 'elevation_profile_lines_',section_str+'_crest2_points_smooth.geojson')
        crest3 = os.path.join(section_dir, 'elevation_profile_lines_',section_str+'_crest3_points_smooth.geojson')
        ip = os.path.join(section_dir, 'elevation_profile_lines_',section_str+'_inflection_points_smooth.geojson')
        toe = os.path.join(section_dir, 'elevation_profile_lines_',section_str+'_toe_points_smooth.geojson')
        mean_shoreline_points = os.path.join(section_dir, section_str+'_mean_shoreline_points.geojson')
        mean_shoreline = os.path.join(section_dir, section_str+'_mean_shoreline.geojson')

        trends_list[i] = trends
        points_list[i] = points
        ref_poly_list[i] = ref_poly
        ref_shore_list[i] = ref_shore
        transects_list[i] = transects
        kde_poly_list[i] = kde_poly
        rois_list[i] = rois
        slopes_list[i] = slopes
        esi_list[i] = esi
        crest1_list[i] = crest1
        crest2_list[i] = crest2
        crest3_list[i] = crest3
        ip_list[i] = ip
        toe_list[i] = toe
        mean_shoreline_points_list[i] = mean_shoreline_points
        mean_shoreline_list[i] = mean_shoreline
        i=i+1

    ##concatenate sections
    region_trends = pd.concat([gpd.read_file(t).to_crs(4326) for t in trends_list])
    region_points = pd.concat([gpd.read_file(p).to_crs(4326)  for p in points_list])
    region_ref_poly = pd.concat([gpd.read_file(rp).to_crs(4326)  for rp in ref_poly_list])
    region_ref_shore = pd.concat([gpd.read_file(rs).to_crs(4326)  for rs in ref_shore_list])
    region_transects = pd.concat([gpd.read_file(tr).to_crs(4326)  for tr in transects_list])
    region_kde_poly = pd.concat([gpd.read_file(kde).to_crs(4326)  for kde in kde_poly_list])
    region_rois = pd.concat([gpd.read_file(r).to_crs(4326)  for r in rois_list])
    region_slopes = pd.concat([gpd.read_file(s).to_crs(4326) for s in slopes_list])
    region_esis = pd.concat([gpd.read_file(e).to_crs(4326) for e in esi_list])
    region_crest1 = pd.concat([gpd.read_file(c1).to_crs(4326)  for c1 in crest1_list])
    region_crest2 = pd.concat([gpd.read_file(c2).to_crs(4326)  for c2 in crest2_list]) 
    region_crest3 = pd.concat([gpd.read_file(c3).to_crs(4326)  for c3 in crest3_list]) 
    region_ip = pd.concat([gpd.read_file(ip).to_crs(4326)  for ip in ip_list])
    region_toe = pd.concat([gpd.read_file(t).to_crs(4326)  for t in toe_list])
    region_mean_shoreline_points = pd.concat([gpd.read_file(msp).to_crs(4326)  for msp in mean_shoreline_points_list])
    region_mean_shoreline = pd.concat([gpd.read_file(ms).to_crs(4326)  for ms in mean_shoreline_list])

    ##save to geoparquet
    region_points.to_parquet(os.path.join(r_home, g+c+rr+'_reprojected_points.parquet'))
    region_points.to_file(os.path.join(r_home, g+c+rr+'_reprojected_points.geojson'))

    region_trends.to_parquet(os.path.join(r_home, g+c+rr+'_trends.parquet'))
    region_trends.to_file(os.path.join(r_home, g+c+rr+'_trends.geojson'))
    
    region_ref_poly.to_parquet(os.path.join(r_home, g+c+rr+'_reference_polygons.parquet'))
    region_ref_poly.to_file(os.path.join(r_home, g+c+rr+'_reference_polygons.geojson'))
    
    region_ref_shore.to_parquet(os.path.join(r_home, g+c+rr+'_reference_shorelines.parquet'))
    region_ref_shore.to_file(os.path.join(r_home, g+c+rr+'_reference_shorelines.geojson'))

    region_transects.to_parquet(os.path.join(r_home, g+c+rr+'_transects.parquet'))
    region_transects.to_file(os.path.join(r_home, g+c+rr+'_transects.geojson'))

    region_kde_poly.to_parquet(os.path.join(r_home, g+c+rr+'_spatial_kde_otsu.parquet'))
    region_kde_poly.to_file(os.path.join(r_home, g+c+rr+'_spatial_kde_otsu.geojson'))

    region_rois.to_parquet(os.path.join(r_home, g+c+rr+'_rois.parquet'))
    region_rois.to_file(os.path.join(r_home, g+c+rr+'_rois.geojson'))

    region_slopes.to_parquet(os.path.join(r_home, g+c+rr+'_transects_slopes.parquet'))
    region_slopes.to_file(os.path.join(r_home, g+c+rr+'_transects_slopes.geojson'))

    region_slopes.to_parquet(os.path.join(r_home, g+c+rr+'_transects_slopes_arctic_dem.parquet'))
    region_slopes.to_file(os.path.join(r_home, g+c+rr+'_transects_slopes_arctic_dem.geojson'))

    region_esis.to_parquet(os.path.join(r_home, g+c+rr+'_esi_transects.parquet'))
    region_esis.to_file(os.path.join(r_home, g+c+rr+'_esi_transects.geojson'))

    region_crest1.to_parquet(os.path.join(r_home, g+c+rr+'_crest1.parquet'))
    region_crest1.to_file(os.path.join(r_home, g+c+rr+'_crest1.geojson'))

    region_crest2.to_parquet(os.path.join(r_home, g+c+rr+'_crest2.parquet'))
    region_crest2.to_file(os.path.join(r_home, g+c+rr+'_crest2.geojson'))

    region_crest3.to_parquet(os.path.join(r_home, g+c+rr+'_crest3.parquet'))
    region_crest3.to_file(os.path.join(r_home, g+c+rr+'_crest3.geojson'))

    region_ip.to_parquet(os.path.join(r_home, g+c+rr+'_inflection_point.parquet'))
    region_ip.to_file(os.path.join(r_home, g+c+rr+'_inflection_point.geojson'))

    region_toe.to_parquet(os.path.join(r_home, g+c+rr+'_toe.parquet'))
    region_toe.to_file(os.path.join(r_home, g+c+rr+'_toe.geojson'))

    region_mean_shoreline_points.to_parquet(os.path.join(r_home, g+c+rr+'_mean_shoreline_points.parquet'))
    region_mean_shoreline_points.to_file(os.path.join(r_home, g+c+rr+'_mean_shoreline_points.geojson'))

    region_mean_shoreline.to_parquet(os.path.join(r_home, g+c+rr+'_mean_shoreline.parquet'))
    region_mean_shoreline.to_file(os.path.join(r_home, g+c+rr+'_mean_shoreline.geojson'))

def merge_regions(g, c, rr_list, home):
    full_points_list = [None]*len(rr_list)
    full_trends_list = [None]*len(rr_list)
    full_ref_poly_list = [None]*len(rr_list)
    full_ref_shore_list = [None]*len(rr_list)
    full_transects_list = [None]*len(rr_list)
    full_kde_poly_list = [None]*len(rr_list)
    full_rois_list = [None]*len(rr_list)
    full_slopes_list = [None]*len(rr_list)
    full_esis_list = [None]*len(rr_list)
    full_crest1_list = [None]*len(rr_list)
    full_crest2_list = [None]*len(rr_list)
    full_crest3_list = [None]*len(rr_list)
    full_ip_list = [None]*len(rr_list)
    full_toe_list = [None]*len(rr_list)
    full_mean_shoreline_points_list = [None]*len(rr_list)
    full_mean_shoreline_list = [None]*len(rr_list)
    i=0
    for rr in rr_list:
        r_home = os.path.join(home, 'G'+g, 'C'+c, 'RR'+rr)
        region_points = os.path.join(r_home, g+c+rr+'_reprojected_points.parquet')
        region_trends = os.path.join(r_home, g+c+rr+'_trends.parquet')
        region_ref_poly = os.path.join(r_home, g+c+rr+'_reference_polygons.parquet')
        region_ref_shore = os.path.join(r_home, g+c+rr+'_reference_shorelines.parquet')
        region_transects = os.path.join(r_home, g+c+rr+'_transects.parquet')
        region_kde_poly = os.path.join(r_home, g+c+rr+'_spatial_kde_otsu.parquet')
        region_rois = os.path.join(r_home, g+c+rr+'_rois.parquet')
        region_slopes = os.path.join(r_home, g+c+rr+'_transects_slopes.parquet')
        region_esis = os.path.join(r_home, g+c+rr+'_esi_transects.parquet')
        region_crest1 = os.path.join(r_home, g+c+rr+'_crest1.parquet')
        region_crest2 = os.path.join(r_home, g+c+rr+'_crest2.parquet')
        region_crest3 = os.path.join(r_home, g+c+rr+'_crest3.parquet')
        region_ip = os.path.join(r_home, g+c+rr+'_inflection_point.parquet')
        region_toe = os.path.join(r_home, g+c+rr+'_toe.parquet')
        region_mean_shoreline_points = os.path.join(r_home, g+c+rr+'_mean_shoreline_points.parquet')
        region_mean_shoreline = os.path.join(r_home, g+c+rr+'_mean_shoreline.parquet')

        full_points_list[i] = region_points
        full_trends_list[i] = region_trends
        full_ref_poly_list[i] = region_ref_poly
        full_ref_shore_list[i] = region_ref_shore
        full_transects_list[i] = region_transects
        full_kde_poly_list[i] = region_kde_poly
        full_rois_list[i] = region_rois
        full_slopes_list[i] = region_slopes
        full_esis_list[i] = region_esis
        full_crest1_list[i] = region_crest1
        full_crest2_list[i] = region_crest2
        full_crest3_list[i] = region_crest3
        full_ip_list[i] = region_ip
        full_toe_list[i] = region_toe
        full_mean_shoreline_points_list[i] = region_mean_shoreline_points
        full_mean_shoreline_list[i] = region_mean_shoreline
        
        i=i+1

    full_points = pd.concat([gpd.read_parquet(p).to_crs(4326) for p in full_points_list])
    full_trends = pd.concat([gpd.read_parquet(t).to_crs(4326) for t in full_trends_list])
    full_ref_poly = pd.concat([gpd.read_parquet(rp).to_crs(4326) for rp in full_ref_poly_list])
    full_ref_shore = pd.concat([gpd.read_parquet(rs).to_crs(4326) for rs in full_ref_shore_list])
    full_transects = pd.concat([gpd.read_parquet(tr).to_crs(4326) for tr in full_transects_list])
    full_kde_poly = pd.concat([gpd.read_parquet(kp).to_crs(4326) for kp in full_kde_poly_list])
    full_rois = pd.concat([gpd.read_parquet(r).to_crs(4326) for r in full_rois_list])
    full_slopes = pd.concat([gpd.read_parquet(s).to_crs(4326) for s in full_slopes_list])
    full_esis = pd.concat([gpd.read_parquet(e).to_crs(4326) for e in full_esis_list])
    full_crest1 = pd.concat([gpd.read_parquet(c1).to_crs(4326) for c1 in full_crest1_list])
    full_crest2 = pd.concat([gpd.read_parquet(c2).to_crs(4326) for c2 in full_crest2_list])
    full_crest3 = pd.concat([gpd.read_parquet(c3).to_crs(4326) for c3 in full_crest3_list])
    full_ip = pd.concat([gpd.read_parquet(ip).to_crs(4326) for ip in full_ip_list])
    full_toe = pd.concat([gpd.read_parquet(toe).to_crs(4326) for toe in full_toe_list])
    full_mean_shoreline_points = pd.concat([gpd.read_parquet(msp).to_crs(4326) for msp in full_mean_shoreline_points_list])
    full_mean_shoreline = pd.concat([gpd.read_parquet(ms).to_crs(4326) for ms in full_mean_shoreline_list])

    full_points.to_parquet(os.path.join(home, g+c+'_reprojected_points.parquet'))
    full_points.to_file(os.path.join(home, g+c+'_reprojected_points.geojson'))

    full_trends.to_parquet(os.path.join(home, g+c+'_trends.parquet'))
    full_trends.to_file(os.path.join(home, g+c+'_trends.geojson'))

    full_ref_poly.to_parquet(os.path.join(home, g+c+'_reference_polygon.parquet'))
    full_ref_poly.to_file(os.path.join(home, g+c+'_reference_polygon.geojson'))

    full_ref_shore.to_parquet(os.path.join(home, g+c+'_reference_shoreline.parquet'))
    full_ref_shore.to_file(os.path.join(home, g+c+'_reference_shoreline.geojson'))

    full_transects.to_parquet(os.path.join(home, g+c+'_transects.parquet'))
    full_transects.to_file(os.path.join(home, g+c+'_transects.geojson'))

    full_kde_poly.to_parquet(os.path.join(home, g+c+'_spatial_kde_polygon.parquet'))
    full_kde_poly.to_file(os.path.join(home, g+c+'_spatial_kde_polygon.geojson'))

    full_rois.to_parquet(os.path.join(home, g+c+'_rois.parquet'))
    full_rois.to_file(os.path.join(home, g+c+'_rois.geojson'))

    full_slopes.to_parquet(os.path.join(home, g+c+'_slopes.parquet'))
    full_slopes.to_file(os.path.join(home, g+c+'_slopes.geojson'))

    full_esis.to_parquet(os.path.join(home, g+c+'_esi.parquet'))
    full_esis.to_file(os.path.join(home, g+c+'_esi.geojson'))

    full_crest1.to_parquet(os.path.join(home, g+c+'_crest1.parquet'))
    full_crest1.to_file(os.path.join(home, g+c+'_crest1.geojson'))

    full_crest2.to_parquet(os.path.join(home, g+c+'_crest2.parquet'))
    full_crest2.to_file(os.path.join(home, g+c+'_crest2.geojson'))

    full_crest3.to_parquet(os.path.join(home, g+c+'_crest3.parquet'))
    full_crest3.to_file(os.path.join(home, g+c+'_crest3.geojson'))

    full_ip.to_parquet(os.path.join(home, g+c+'_inflection_point.parquet'))
    full_ip.to_file(os.path.join(home, g+c+'_inflection_point.geojson'))

    full_toe.to_parquet(os.path.join(home, g+c+'_toe.parquet'))
    full_toe.to_file(os.path.join(home, g+c+'_toe.geojson'))

    full_mean_shoreline_points.to_parquet(os.path.join(home, g+c+'_mean_shoreline_points.parquet'))
    full_mean_shoreline_points.to_file(os.path.join(home, g+c+'_mean_shoreline_points.geojson'))

    full_mean_shoreline.to_parquet(os.path.join(home, g+c+'_mean_shoreline.parquet'))
    full_mean_shoreline.to_file(os.path.join(home, g+c+'_mean_shoreline.geojson'))


# ##Calling the mapping function
cfg = args.config
# g = args.global_region
# c = args.coastal_area
# rr = args.subregion
# gpu = args.gpu_id
# function = args.function
# custom_dem = args.dem
# slope_value = args.slope
# reference_elevation_value = args.reference_elevation
# smooth_slopes = args.smooth_slopes
# resample_freq = args.resample_frequency
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu
# model = args.model
# estimate = args.estimate
# planet_bool = args.planet_bool
# custom_sections = args.custom_sections
# waterline_filter = args.waterline_filter
# year_min = args.year_min
# year_max = args.year_max
# r_home = os.path.join(args.home, 'G'+g, 'C'+c, 'RR'+rr)
# coastseg_roi_folder = args.coastseg_roi_folder

# try:
#     custom_sections = custom_sections.split(',')
# except:
#     pass
# try:
#     update = args.update
# except:
#     update=False
# try:
#     reset=args.reset
# except:
#     reset=False
# if custom_dem != '':
#     arctic_dem = False
# else:
#     arctic_dem = True
# if reference_elevation_value == '':
#     reference_elevation_value = 0
# else:
#     reference_elevation_value = float(reference_elevation_value)
# if slope_value == '':
#     slope_value = np.nan
# else:
#     slope_value = float(slope_value)
# try:
#     r_home = os.path.join('/', 'mnt', 'hdd_6tb', 'Alaska_Analysis_Images', 'G'+g, 'C'+c, 'RR'+rr)
# except:
#     c_home = os.path.join('/', 'mnt', 'hdd_6tb', 'Alaska_Analysis_Images', 'G'+g, 'C'+c)

# validation_folder = os.path.join('/', 'home', 'aksup', 'doodleverse', 'CoastSeg', 'data', 'validation_sites')
# alaska_folder = os.path.join('/', 'mnt', 'f', 'Merbok', 'sorted_alaska')
# alaska_sar_folder = os.path.join('/', 'mnt', 'f', 'Merbok', 'SAR_rois')
# planet_folder = os.path.join('/', 'mnt', 'c', 'Merbok', 'MerbokPlanet')

# if rr == '99':
#     coastseg_roi_folder=validation_folder
# else:
#     coastseg_roi_folder=alaska_folder


with open(cfg, "r") as f:
    cfg = yaml.safe_load(f)

# Helper for getting values safely
def get(key, default=None):
    return cfg.get(key, default)

# Helper: expand env vars and ~ in paths
def expand_path(p):
    if p is None:
        return ""
    return os.path.expanduser(os.path.expandvars(str(p).strip()))

# Core identifiers
g = str(get("global_region"))
c = str(get("coastal_area"))
rr = str(get("subregion"))
sss = str(get("shoreline_section", ""))

# GPU
gpu = str(get("gpu_id", "-1"))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# Flags
update = bool(get("update", False))
reset = bool(get("reset", False))
planet_bool = bool(get("planet_bool", False))
smooth_slopes = bool(get("smooth_slopes", False))
waterline_filter = bool(get("waterline_filter", False))

# Function and model settings
function = str(get("function", ""))
model = str(get("model", "global"))
estimate = str(get("estimate", "ensemble"))
resample_freq = str(get("resample_frequency", "365D"))

# DEM logic
custom_dem = str(get("dem", ""))
arctic_dem = (custom_dem == "")

# Slope
slope_raw = str(get("slope", "")).strip()
slope_value = float(slope_raw) if slope_raw else np.nan

# Reference elevation
ref_raw = str(get("reference_elevation", "")).strip()
reference_elevation_value = float(ref_raw) if ref_raw else 0.0

# Years
year_min = str(get("year_min", "1984"))
year_max = str(get("year_max", "2026"))

# ee_project
ee_project = str(get("ee_project"))

# Custom sections
sec_raw = cfg.get("custom_sections", "")

if isinstance(sec_raw, list):
    # If a YAML list is provided:
    # - non-empty list -> normalize values
    # - empty list []   -> return [''] to match your branch checks
    custom_sections = [str(s).strip() for s in sec_raw] if sec_raw else ['']
else:
    # If a string is provided:
    # - blank string "" -> ['']
    # - comma string "001,002" -> ["001", "002"]
    sec_str = str(sec_raw)
    if sec_str.strip() == "":
        custom_sections = ['']
    else:
        custom_sections = [s.strip() for s in sec_str.split(",") if s.strip()]

# Paths
home = os.path.expanduser(get("home", ""))
coastseg_roi_folder = os.path.expanduser(get("coastseg_roi_folder", ""))
planet_folder = expand_path(get("planet_folder", "")) 

# Derived path
r_home = os.path.join(home, f"G{g}", f"C{c}", f"RR{rr}")
c_home = os.path.join(home, f"G{g}", f"C{c}")

if sss != '' and custom_sections == ['']:
    if function == 'download_and_process':
        print(sss)
        print('downloading imagery')
        download_imagery_section(g, c, rr, sss, r_home, coastseg_roi_folder, year_min+'-01-01', year_max+'-12-31', ee_project)
        print('finding intersecting rasters')
        satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
        batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
        print('running image suitability model')
        image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        print('reorganizing CoastSeg imagery')
        rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
        update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('pansharpening and co-registering imagery')
        pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
        print('segmenting imagery')
        segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
        print('running segmentation suitability model')
        segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        print('extracting shorelines')
        extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
        print('clipping shorelines')
        clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('merging shorelines')
        merge_shorelines_section(g, c, rr, sss, r_home)
        print('resampling shorelines')
        resample_shorelines_section(g, c, rr, sss, r_home)      
        print('computing spatial kde')
        spatial_kde_section(g, c, rr, sss, r_home)
        print('making transect timeseries')
        transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
        print('getting tide data from FES22 Model')
        get_tide_data_section(g, c, rr, sss, r_home)
        print('applying tide corrections')
        tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
        print('ensembling and filtering data')
        filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
        print('computing shoreline trends: overall, 1990s, 2000s, 2010s, 2020s')
        get_trends_section(g, c, rr, sss, r_home)
        print('computing record stats')
        record_stats_shoreline_section(g, c, rr, sss, r_home)
    elif function == 'process':
        print(sss)
        print('finding intersecting rasters')
        satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
        batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
        print('running image suitability model')
        image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        print('reorganizing CoastSeg imagery')
        rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
        update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('pansharpening and co-registering imagery')
        pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
        print('segmenting imagery')
        segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
        print('running segmentation suitability model')
        segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        print('extracting shorelines')
        extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
        print('clipping shorelines')
        clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('merging shorelines')
        merge_shorelines_section(g, c, rr, sss, r_home)
        print('resampling shorelines')
        resample_shorelines_section(g, c, rr, sss, r_home)      
        print('computing spatial kde')
        spatial_kde_section(g, c, rr, sss, r_home)
        print('making transect timeseries')
        transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
        print('getting tide data from FES22 Model')
        get_tide_data_section(g, c, rr, sss, r_home)
        print('applying tide corrections')
        tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
        print('ensembling and filtering data')
        filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
        print('computing shoreline trends: overall, 1990s, 2000s, 2010s, 2020s')
        get_trends_section(g, c, rr, sss, r_home)
        print('computing record stats')
        record_stats_shoreline_section(g, c, rr, sss, r_home)
    elif function == 'rename_transects':
        print('renaming transects')
        rename_transects_section(g, c, rr, sss, r_home)
    elif function == 'transects':
        print('making transects')
        make_transects_section(g, c, rr, sss, r_home)
    elif function == 'get_slope':
        print('getting beach slopes')
        get_slopes_section(g, c, rr, sss, r_home, arctic_dem, custom_dem)
    elif function == 'prep_slope':
        print('assigning constant slope to section')
        prep_slope_data(g,
                    c,
                    rr,
                    sss,
                    r_home,
                    constant=slope_value,
                    smooth=smooth_slopes,
                    dem=custom_dem)
    elif function == 'rois':
        print('making rois')
        make_rois_section(g, c, rr, sss, r_home)
    elif function == 'download':
        print('downloading imagery')
        download_imagery_section(g, c, rr, sss, r_home, coastseg_roi_folder, year_min+'-01-01', year_max+'-12-31', ee_project)
    elif function == 'find_rasters':
        print('finding intersecting rasters')
        satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
        batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
    elif function == 'check_planet':
        print('checking for planet data')
        check_planet_section(g, c, rr, sss, r_home)    
    elif function == 'image_filter':
        print('running image suitability model')
        image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
    elif function == 'reorg':
        print('reorganizing CoastSeg imagery')
        rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
        update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
    elif function == 'pansharpen_coreg':
        print('pansharpening and co-registering imagery')
        pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
    elif function == 'seg':
        print('segmenting imagery')
        segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
    elif function == 'seg_filter':
        print('running segmentation suitability model')
        segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
    elif function == 'extract':
        print('extracting shorelines')
        extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
        print('clipping shorelines')
        clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('merging shorelines')
        merge_shorelines_section(g, c, rr, sss, r_home)
        print('resampling shorelines')
        resample_shorelines_section(g, c, rr, sss, r_home)      
    elif function == 'post_process':
        print('computing spatial kde')
        spatial_kde_section(g, c, rr, sss, r_home)
        print('making transect timeseries')
        transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
        print('getting tide data')
        get_tide_data_section(g, c, rr, sss, r_home)
        print('applying tide corrections')
        tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
        print('ensembling and filtering data')
        filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
        print('computing shoreline trends')
        get_trends_section(g, c, rr, sss, r_home)
    elif function == 'record_stats':
        print('computing record stats')
        record_stats_shoreline_section(g, c, rr, sss, r_home)
    elif function == 'validate':
        in_situ_validation.in_situ_comparison(r_home,
                       g,
                       c,
                       rr,
                       sss,
                       plot_timeseries=True,
                       which_estimate=estimate,
                       window=10,
                       legend_loc=(0.4,0.6))
elif custom_sections != ['']:
    if function == 'download_and_process':
        sections = sorted(custom_sections)
        print('downloading imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            download_imagery_section(g, c, rr, sss, r_home, coastseg_roi_folder, year_min+'-01-01', year_max+'-12-31', ee_project)
        print('finding intersecting rasters')
        for section in sections:
            print(section)
            sss = section[3:]
            satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
            batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
        print('running image suitability model')
        for section in sections:
            print(section)
            sss = section[3:]
            image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        print('reorganizing CoastSeg imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
            update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('pansharpening and co-registering imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
        print('segmenting imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
        print('running segmentation suitability model')
        for section in sections:
            print(section)
            sss = section[3:]
            segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        for section in sections:
            print(section)
            sss = section[3:]
            print('extracting shorelines')
            extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
            print('clipping shorelines')
            clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
            print('merging shorelines')
            merge_shorelines_section(g, c, rr, sss, r_home)
            print('resampling shorelines')
            resample_shorelines_section(g, c, rr, sss, r_home)      
        for section in sections:
            print(section)
            sss = section[3:]
            print('computing spatial kde')
            spatial_kde_section(g, c, rr, sss, r_home)
            print('making transect timeseries')
            transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
            print('getting tide data from FES22 Model')
            get_tide_data_section(g, c, rr, sss, r_home)
            print('applying tide corrections')
            tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
            print('ensembling and filtering data')
            filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
            print('computing shoreline trends: overall, 1990s, 2000s, 2010s, 2020s')
            get_trends_section(g, c, rr, sss, r_home)
        print('computing record stats')
        for section in sections:
            print(section)
            sss = section[3:]
            record_stats_shoreline_section(g, c, rr, sss, r_home)
    if function == 'process':
        sections = sorted(custom_sections)
        print('finding intersecting rasters')
        for section in sections:
            print(section)
            sss = section[3:]
            satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
            batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
        print('running image suitability model')
        for section in sections:
            print(section)
            sss = section[3:]
            image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        print('reorganizing CoastSeg imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
            update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('pansharpening and co-registering imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
        print('segmenting imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
        print('running segmentation suitability model')
        for section in sections:
            print(section)
            sss = section[3:]
            segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        for section in sections:
            print(section)
            sss = section[3:]
            print('extracting shorelines')
            extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
            print('clipping shorelines')
            clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
            print('merging shorelines')
            merge_shorelines_section(g, c, rr, sss, r_home)
            print('resampling shorelines')
            resample_shorelines_section(g, c, rr, sss, r_home)      
        for section in sections:
            print(section)
            sss = section[3:]
            print('computing spatial kde')
            spatial_kde_section(g, c, rr, sss, r_home)
            print('making transect timeseries')
            transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
            print('getting tide data from FES22 Model')
            get_tide_data_section(g, c, rr, sss, r_home)
            print('applying tide corrections')
            tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
            print('ensembling and filtering data')
            filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
            print('computing shoreline trends: overall, 1990s, 2000s, 2010s, 2020s')
            get_trends_section(g, c, rr, sss, r_home)
        print('computing record stats')
        for section in sections:
            print(section)
            sss = section[3:]
            record_stats_shoreline_section(g, c, rr, sss, r_home)
    elif function == 'rename_transects':
        print('renaming transects')
        sections = sorted(custom_sections)
        print(custom_sections)
        for section in sections:
            print(section)
            sss = section
            rename_transects_section(g, c, rr, sss, r_home)
    elif function == 'transects':
        print('making transects')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            make_transects_section(g, c, rr, sss, r_home)
    elif function == 'get_slope':
        print('getting beach slopes')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            get_slopes_section(g, c, rr, sss, r_home, arctic_dem, custom_dem)
    elif function == 'prep_slope':
        print('assigning slope to section')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            prep_slope_data(g,
                        c,
                        rr,
                        sss,
                        r_home,
                        constant=slope_value,
                        smooth=smooth_slopes,
                        dem=custom_dem)
    elif function == 'rois':
        print('making rois')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            make_rois_section(g, c, rr, sss, r_home)
    elif function == 'download':
        print('downloading imagery')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            download_imagery_section(g, c, rr, sss, r_home, coastseg_roi_folder, year_min+'-01-01', year_max+'-12-31', ee_project)
    elif function == 'find_rasters':
        print('finding intersecting rasters')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
            batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
    elif function == 'check_planet':
        print('checking for planet data')
        sections = sorted(custom_sections)
        for section in sections:
            sss = section
            check_planet_section(g, c, rr, sss, r_home)
    elif function == 'image_filter':
        print('running image suitability model')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
    elif function == 'reorg':
        print('reorganizing CoastSeg imagery')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
            update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
    elif function == 'pansharpen_coreg':
        print('pansharpening and co-registering imagery')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
    elif function == 'seg':
        print('segmenting imagery')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
    elif function == 'seg_filter':
        print('running segmentation suitability model')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
    elif function == 'extract':
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            print('extracting shorelines')
            extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
            print('clipping shorelines')
            clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
            print('merging shorelines')
            merge_shorelines_section(g, c, rr, sss, r_home)
            print('resampling shorelines')
            resample_shorelines_section(g, c, rr, sss, r_home)      
    elif function == 'post_process':
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            print('computing spatial kde')
            spatial_kde_section(g, c, rr, sss, r_home)
            print('making transect timeseries')
            transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
            print('getting tide data from FES22 Model')
            get_tide_data_section(g, c, rr, sss, r_home)
            print('applying tide corrections')
            tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
            print('ensembling and filtering data')
            filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
            print('computing shoreline trends: overall, 1990s, 2000s, 2010s, 2020s')
            get_trends_section(g, c, rr, sss, r_home)
    elif function == 'record_stats':
        print('computing record stats')
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            record_stats_shoreline_section(g, c, rr, sss, r_home)
    elif function == 'validate':
        sections = sorted(custom_sections)
        for section in sections:
            print(section)
            sss = section
            in_situ_validation.in_situ_comparison(r_home,
                        g,
                        c,
                        rr,
                        sss,
                        plot_timeseries=True,
                        which_estimate=estimate,
                        window=10,
                        legend_loc=(0.4,0.6))
else:
    if function == 'download_and_process':
        sections = sorted(get_immediate_subdirectories(r_home))
        print('downloading imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            download_imagery_section(g, c, rr, sss, r_home, coastseg_roi_folder, year_min+'-01-01', year_max+'-12-31', ee_project)
        print('finding intersecting rasters')
        for section in sections:
            print(section)
            sss = section[3:]
            satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
            batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
        print('running image suitability model')
        for section in sections:
            print(section)
            sss = section[3:]
            image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        print('reorganizing CoastSeg imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
            update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('pansharpening and co-registering imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
        print('segmenting imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
        print('running segmentation suitability model')
        for section in sections:
            print(section)
            sss = section[3:]
            segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        for section in sections:
            print(section)
            sss = section[3:]
            print('extracting shorelines')
            extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
            print('clipping shorelines')
            clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
            print('merging shorelines')
            merge_shorelines_section(g, c, rr, sss, r_home)
            print('resampling shorelines')
            resample_shorelines_section(g, c, rr, sss, r_home)      
        for section in sections:
            print(section)
            sss = section[3:]
            print('computing spatial kde')
            spatial_kde_section(g, c, rr, sss, r_home)
            print('making transect timeseries')
            transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
            print('getting tide data from FES22 Model')
            get_tide_data_section(g, c, rr, sss, r_home)
            print('applying tide corrections')
            tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
            print('ensembling and filtering data')
            filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
            print('computing shoreline trends: overall, 1990s, 2000s, 2010s, 2020s')
            get_trends_section(g, c, rr, sss, r_home)
        print('computing record stats')
        for section in sections:
            print(section)
            sss = section[3:]
            record_stats_shoreline_section(g, c, rr, sss, r_home)
    if function == 'process':
        sections = sorted(get_immediate_subdirectories(r_home))
        print('finding intersecting rasters')
        for section in sections:
            print(section)
            sss = section[3:]
            satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
            batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
        print('running image suitability model')
        for section in sections:
            print(section)
            sss = section[3:]
            image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        print('reorganizing CoastSeg imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
            update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
        print('pansharpening and co-registering imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
        print('segmenting imagery')
        for section in sections:
            print(section)
            sss = section[3:]
            segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
        print('running segmentation suitability model')
        for section in sections:
            print(section)
            sss = section[3:]
            segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
        for section in sections:
            print(section)
            sss = section[3:]
            print('extracting shorelines')
            extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
            print('clipping shorelines')
            clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
            print('merging shorelines')
            merge_shorelines_section(g, c, rr, sss, r_home)
            print('resampling shorelines')
            resample_shorelines_section(g, c, rr, sss, r_home)      
        for section in sections:
            print(section)
            sss = section[3:]
            print('computing spatial kde')
            spatial_kde_section(g, c, rr, sss, r_home)
            print('making transect timeseries')
            transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
            print('getting tide data from FES22 Model')
            get_tide_data_section(g, c, rr, sss, r_home)
            print('applying tide corrections')
            tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
            print('ensembling and filtering data')
            filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
            print('computing shoreline trends: overall, 1990s, 2000s, 2010s, 2020s')
            get_trends_section(g, c, rr, sss, r_home)
        print('computing record stats')
        for section in sections:
            print(section)
            sss = section[3:]
            record_stats_shoreline_section(g, c, rr, sss, r_home)
    elif function == 'rename_transects':
        print('renaming transects')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            rename_transects_section(g, c, rr, sss, r_home)
    elif function == 'transects':
        print('making transects')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            make_transects_section(g, c, rr, sss, r_home)
    elif function == 'get_slope':
        print('getting beach slopes')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            get_slopes_section(g, c, rr, sss, r_home, arctic_dem, custom_dem)
    elif function == 'prep_slope':
        print('assigning slope to section')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            prep_slope_data(g,
                        c,
                        rr,
                        sss,
                        r_home,
                        constant=slope_value,
                        smooth=smooth_slopes,
                        dem=custom_dem)
    elif function == 'rois':
        print('making rois')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            make_rois_section(g, c, rr, sss, r_home)
    elif function == 'download':
        print('downloading imagery')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            download_imagery_section(g, c, rr, sss, r_home, coastseg_roi_folder, year_min+'-01-01', year_max+'-12-31', ee_project)
    elif function == 'find_rasters':
        print('finding intersecting rasters')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            satellite_images = get_all_satellite_imagery_tiffs(coastseg_roi_folder, planet_folder, update=update)
            batch_find_intersecting_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet_folder, satellite_images, update=update)
    elif function == 'check_planet':
        print('checking for planet data')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            sss = section[3:]
            check_planet_section(g, c, rr, sss, r_home)
    elif function == 'image_filter':
        print('running image suitability model')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            image_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
    elif function == 'reorg':
        print('reorganizing CoastSeg imagery')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home=coastseg_roi_folder, sorted_planet_home=planet_folder, planet=planet_bool)
            update_metadata_section(g, c, rr, sss, r_home, planet=planet_bool)
    elif function == 'pansharpen_coreg':
        print('pansharpening and co-registering imagery')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            pansharpen_and_co_register_section(g, c, rr, sss, r_home, coastseg_roi_folder, planet=planet_bool)
    elif function == 'seg':
        print('segmenting imagery')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            segment_imagery_section(g, c, rr, sss, r_home, gpu=gpu, model=model, planet=planet_bool)
    elif function == 'seg_filter':
        print('running segmentation suitability model')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            segmentation_suitability_section(g, c, rr, sss, r_home, gpu=gpu, planet=planet_bool)
    elif function == 'extract':
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            print('extracting shorelines')
            extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=reset, planet=planet_bool)
            print('clipping shorelines')
            clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=planet_bool)
            print('merging shorelines')
            merge_shorelines_section(g, c, rr, sss, r_home)
            print('resampling shorelines')
            resample_shorelines_section(g, c, rr, sss, r_home)      
    elif function == 'post_process':
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            print('computing spatial kde')
            spatial_kde_section(g, c, rr, sss, r_home)
            print('making transect timeseries')
            transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=waterline_filter)
            print('getting tide data from FES22 Model')
            get_tide_data_section(g, c, rr, sss, r_home)
            print('applying tide corrections')
            tidally_correct_section(g, c, rr, sss, r_home, dem=custom_dem, reference_elevation=reference_elevation_value)
            print('ensembling and filtering data')
            filter_timeseries_section(g, c, rr, sss, r_home, resample_freq=resample_freq)
            print('computing shoreline trends: overall, 1990s, 2000s, 2010s, 2020s')
            get_trends_section(g, c, rr, sss, r_home)
    elif function == 'record_stats':
        print('computing record stats')
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            record_stats_shoreline_section(g, c, rr, sss, r_home)
    elif function == 'validate':
        sections = sorted(get_immediate_subdirectories(r_home))
        for section in sections:
            print(section)
            sss = section[3:]
            in_situ_validation.in_situ_comparison(r_home,
                        g,
                        c,
                        rr,
                        sss,
                        plot_timeseries=True,
                        which_estimate=estimate,
                        window=10,
                        legend_loc=(0.4,0.6))
    elif function == 'merge_sections':
        merge_files_region(g, c, rr, r_home)
    elif function == 'merge_regions':
        home = os.path.dirname(os.path.dirname(c_home))
        region_list = sorted(get_immediate_subdirectories(os.path.join(home, 'G'+g, 'C'+c)))
        region_list = [rr[2:] for rr in region_list if rr!='RR99']
        merge_regions(g, c, region_list, home)
        


