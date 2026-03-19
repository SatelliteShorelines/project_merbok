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
from tqdm import tqdm
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
gdal.UseExceptions()        # prevents the warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="osgeo.gdal")
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.features import shapes
from rasterio.features import shapes as raster_shapes

from rasterio.features import shapes as rio_shapes  # consistent alias
from shapely.geometry import LineString
from shapely.geometry import shape as shapely_shape  # convert GeoJSON-like mappings to Shapely
from shapely.ops import unary_union                  # faster union on lists of shapely 

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

# ---- Profiler and logging and tracing ----
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

def rearrange_coastseg_data_section_old(g, c, rr, sss, r_home, sorted_alaska_home, sorted_planet_home = None, planet=False):
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


def _get_logger_reorganization(section_string: str, level: int = logging.INFO) -> logging.Logger:
    """
    File-only logger for data reorganization runs:
      <cwd>/logs/reorganization/<section_string>_<YYYYMMDD_%H%M%S>.log

    - Creates the log directory if missing.
    - Uses a timestamped file per invocation (no rotation).
    - Prevents duplicate handlers if requested multiple times.
    - No console handler (no stdout/stderr output).
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'reorganization')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"reorganization.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs via root

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler only
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def rearrange_coastseg_data_section(g, c, rr, sss, r_home, sorted_alaska_home, sorted_planet_home=None, planet=False):
    """
    Gets all of the Landsat and Sentinel (or Planet) satellite imagery for a section,
    filters/scopes by model score, and reorganizes/copies MS TIFFs into a section-level folder.

    Parameters
    ----------
    g : str
        Global region
    c : str
        Coastal area
    rr : str
        Subregion
    sss : str
        Shoreline section suffix
    r_home : str
        Path to subregion home directory
    sorted_alaska_home : str
        Root path where the CoastSeg-sorted Alaska imagery lives (Landsat/Sentinel)
    sorted_planet_home : str | None
        Root path where the CoastSeg-sorted Planet imagery lives (Planet only)
    planet : bool
        If True, operate on Planet imagery; else on Landsat/Sentinel imagery

    Returns
    -------
    dict
        Summary with keys:
        - 'new_csv_path': path to the rewritten CSV (underscored name)
        - 'filtered_count': number of rows meeting the score threshold
        - 'copied_count': number of images successfully copied
        - 'new_tiff_dir': destination directory for copied TIFFs
        - 'error': optional error message if the run failed
    """

    section = 'SSS' + sss
    section_dir = os.path.join(r_home, section)
    # Keep your original section_string formulation
    section_string = g + c + rr + section[3:]

    logger = _get_logger_reorganization(section_string, level=logging.INFO)

    logger.info("=== Reorganization Start ===")
    logger.info("Inputs | section=%s | section_dir=%s | section_string=%s | planet=%s", section, section_dir, section_string, planet)
    logger.info("Inputs | sorted_alaska_home=%s | sorted_planet_home=%s", sorted_alaska_home, sorted_planet_home)

    new_csv_path_dir = os.path.join(section_dir, section_string + '_ms_lists')
    try:
        os.mkdir(new_csv_path_dir)
        logger.info("Created directory: %s", new_csv_path_dir)
    except Exception as e:
        # If it exists or can't create, log and continue (original code did pass)
        logger.debug("Directory exists or could not be created (%s): %s", new_csv_path_dir, str(e))

    # Choose source scored CSV and destination underscored CSV name
    if planet:
        new_csv_path = os.path.join(new_csv_path_dir, 'planet_ms_paths_scored_.csv')
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored.csv')
        source_label = "planet"
    else:
        new_csv_path = os.path.join(new_csv_path_dir, 'landsat_sentinel_ms_paths_scored_.csv')
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored.csv')
        source_label = "landsat_sentinel"

    logger.info("Reading scored CSV (%s): %s", source_label, sat_image_list_df_path)
    sat_image_list_df = pd.read_csv(sat_image_list_df_path)

    # Drop any unnamed columns (likely index artifacts)
    unnamed_cols = [col for col in sat_image_list_df.columns if 'Unnamed' in col]
    if unnamed_cols:
        sat_image_list_df = sat_image_list_df.drop(columns=unnamed_cols)
        logger.info("Dropped unnamed columns: %s", unnamed_cols)

    # Write the underscored CSV (as in the original code)
    sat_image_list_df.to_csv(new_csv_path, index=False)
    logger.info("Wrote cleaned CSV (underscore version): %s | rows=%d", new_csv_path, len(sat_image_list_df))

    copied_count = 0
    try:
        # Filter by model score threshold (>= 0.335)
        if 'model_scores' not in sat_image_list_df.columns:
            raise KeyError("'model_scores' column not found in the input CSV")

        sat_image_list_df = sat_image_list_df[sat_image_list_df['model_scores'] >= 0.335].reset_index(drop=True)
        logger.info("Filtered rows with model_scores >= 0.335: %d", len(sat_image_list_df))

        new_tiff_dir = os.path.join(section_dir, 'ms_tiff_paths')
        try:
            os.mkdir(new_tiff_dir)
            logger.info("Created directory: %s", new_tiff_dir)
        except Exception as e:
            logger.debug("Directory exists or could not be created (%s): %s", new_tiff_dir, str(e))

        # Iterate and copy files
        for i in range(len(sat_image_list_df)):
            try:
                image = sat_image_list_df['ms_tiff_path'].iloc[i]
                satname = sat_image_list_df['satnames'].iloc[i]
            except KeyError as ke:
                logger.exception("Missing expected column while iterating: %s", str(ke))
                break

            # Resolve root (Planet vs L/S)
            root_sorted_home = sorted_planet_home if planet else sorted_alaska_home
            if not root_sorted_home:
                logger.error("Root sorted home is not provided for %s imagery.", source_label)
                break

            # CoastSeg dir resolution and potential archive handling
            coastseg_dir_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image))))
            coastseg_dir = os.path.join(root_sorted_home, coastseg_dir_name)
            destination_folder = os.path.dirname(coastseg_dir)
            archive_file = coastseg_dir + '.tar.gz'

            # Unarchive if needed
            try:
                if os.path.isfile(archive_file) and not os.path.isdir(coastseg_dir):
                    logger.info("Unarchiving (dir missing): %s -> %s", archive_file, destination_folder)
                    unpack_tar_gz(archive_file, destination_folder)
                elif os.path.isfile(archive_file) and not os.listdir(coastseg_dir):
                    logger.info("Unarchiving (dir empty): %s -> %s", archive_file, destination_folder)
                    unpack_tar_gz(archive_file, destination_folder)
            except Exception as ue:
                logger.exception("Failed to unarchive %s: %s", archive_file, str(ue))
                continue

            # Ensure per-satellite subdir in destination
            try:
                os.mkdir(os.path.join(new_tiff_dir, satname))
                logger.debug("Created satname dir: %s", os.path.join(new_tiff_dir, satname))
            except Exception:
                # Fine if it exists
                pass

            # Source file inside CoastSeg structure
            source_image = os.path.join(coastseg_dir, satname, 'ms', os.path.basename(image))
            new_image = os.path.join(new_tiff_dir, satname, os.path.basename(image))

            # Copy the file
            try:
                shutil.copyfile(source_image, new_image)
                copied_count += 1
                logger.info("Copied: %s -> %s", source_image, new_image)
            except FileNotFoundError:
                logger.warning("Missing source image, skipping: %s", source_image)
            except Exception as ce:
                logger.exception("Failed to copy %s -> %s: %s", source_image, new_image, str(ce))

        summary = {
            'new_csv_path': new_csv_path,
            'filtered_count': len(sat_image_list_df),
            'copied_count': copied_count,
            'new_tiff_dir': new_tiff_dir,
        }
        logger.info("=== Reorganization End === | Summary: %s", summary)
        return summary

    except Exception as e:
        # Quiet error path: no printing to console
        logger.exception("Reorganization failed for section_string=%s: %s", section_string, str(e))

        return {
            'new_csv_path': new_csv_path,
            'filtered_count': 0,
            'copied_count': copied_count,
            'new_tiff_dir': os.path.join(section_dir, 'ms_tiff_paths'),
            'error': str(e),
        }

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

def pansharpen_and_co_register_section_old(g, c, rr, sss, r_home, landsat_sentinel_folder, planet=False):
    """
    Pansharpens and co-registers all imagery for a section.
    Must have run image suitability before using this function.
    This will pansharpen all bands for all imagery in a section, then will try to coregister all of the pansharpened imagery (L5, L8, L9)
    Algorithm works by picking the highest scored LANDSAT image for an ROI as the reference image, 
    then will try to co-register all imagery in that ROI. 
    Planet and L7 are not co-registered


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


import contextlib

@contextlib.contextmanager
def _silence_console(silence_stderr: bool = True):
    """
    Temporarily redirect stdout (and optionally stderr) to os.devnull,
    silencing any print/logging-to-console from nested calls.
    """
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            if silence_stderr:
                with contextlib.redirect_stderr(devnull):
                    yield
            else:
                yield


def get_logger(section_string: str) -> logging.Logger:
    """
    Configure and return a logger that writes to:
      os.path.join(os.getcwd(), 'logs', 'pansharpen_coregister_logs', '<section_string>_<YYYYMMDD_HHMMSS>.log')

    Each call creates a new timestamped log file and a dedicated logger instance.
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'pansharpen_coregister_logs')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    # Ensure a fresh logger per run (unique name includes timestamp)
    logger_name = f"pansharpen_coregister.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # If console output is desired, uncomment:
    # ch = logging.StreamHandler()
    # ch.setFormatter(fmt)
    # logger.addHandler(ch)

    return logger



def pansharpen_and_co_register_section(g, c, rr, sss, r_home, landsat_sentinel_folder, planet=False):
    """
    Pansharpens and co-registers imagery for a section.

    Decision Flow (concise):
      1) If planet=True:
           - Load Planet CSV, set analysis_image=ms_tiff_path, use_original=True for all rows, save CSV, return.
         Else:
           - Load Landsat/Sentinel CSV, refresh via get_new_df_with_reference_image (per-ROI reference_image).

      2) For each image:
           - If local pansharpened+coregistered exist: use_coregistered=True.
           - Else if legacy outputs exist: copy, use_coregistered=True.
           - Else if model_scores < 0.335: skip.
           - Read minimal metadata (nodata, transform, width, height, count, crs); store xres/yres.
           - By satellite:
               * Non-PS & Non-L7: pansharpen → try coreg → choose coreg|pansharpen|original.
               * L7: pansharpen only → choose pansharpen|original.
           - Persist CSV updates progressively.

    Logs:
      Saved to: os.path.join(os.getcwd(), 'logs', 'pansharpen_coregister_logs', '<section_string>_<timestamp>.log')
    """
    section = 'SSS' + sss
    section_dir = os.path.join(r_home, section)
    section_string = g + c + rr + section[3:]
    logger = get_logger(section_string)

    if planet is True:
        df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_update.csv')
        if os.path.isfile(df_path):
            df = pd.read_csv(df_path)
            df['use_coregistered'] = [False] * len(df)
            df['use_pansharpened'] = [False] * len(df)
            df['use_original'] = [True] * len(df)
            df['analysis_image'] = df['ms_tiff_path']
            try:
                df.to_csv(df_path, index=False)
                logger.info("Planet: marked use_original=True; analysis_image=ms_tiff_path; CSV saved: %s", df_path)
            except Exception:
                logger.exception("Planet: CSV write failed: %s", df_path)
        else:
            logger.warning("Planet: CSV not found: %s", df_path)
        return

    # Landsat/Sentinel branch
    df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_update.csv')
    df = pd.read_csv(df_path)

    # Refresh DF with per-ROI reference_image
    df = get_new_df_with_reference_image(g, c, rr, r_home, section, landsat_sentinel_folder)

    # Ensure flag columns exist
    for col, default in [
        ("use_coregistered", False),
        ("use_pansharpened", False),
        ("use_original", False),
        ("analysis_image", None),
        ("xres", None),
        ("yres", None),
    ]:
        if col not in df.columns:
            df[col] = default

    try:
        df.to_csv(df_path, index=False)
        logger.info("Landsat/Sentinel: refreshed CSV: %s", df_path)
    except Exception:
        logger.exception("Initial CSV write failed: %s", df_path)

    # Progress bar over images
    for idx in tqdm(range(len(df['ms_tiff_path'])), desc=f"Processing {section_string}", unit="img"):
        try:
            roi_folder = df['roi_folder'].iloc[idx]
            image = df['ms_tiff_path'].iloc[idx]
            satname = df['satnames'].iloc[idx]

            # Legacy outputs (optional copy-forward)
            potential_old_pansharpen = os.path.join('/', 'mnt', 'f', 'Merbok', 'sorted_alaska', roi_folder, satname, 'ms', 'pansharpen', os.path.basename(image))
            potential_old_coreg = os.path.join('/', 'mnt', 'f', 'Merbok', 'sorted_alaska', roi_folder, satname, 'ms', 'pansharpen', 'coregistered', os.path.basename(image))

            # Local output paths
            pansharpen_output = os.path.join(os.path.dirname(image), 'pansharpen', os.path.basename(image))
            coreg_dir = os.path.join(os.path.dirname(image), 'pansharpen', 'coregistered')
            co_reg_image = os.path.join(coreg_dir, os.path.basename(image))

            # Already processed locally
            if os.path.isfile(pansharpen_output) and os.path.isfile(co_reg_image):
                df.at[idx, 'analysis_image'] = co_reg_image
                df.at[idx, 'use_coregistered'] = True
                logger.info("Already coregistered: %s", co_reg_image)
                continue

            # Copy legacy outputs if present
            if os.path.isfile(potential_old_pansharpen) and os.path.isfile(potential_old_coreg):
                try:
                    os.makedirs(os.path.dirname(pansharpen_output), exist_ok=True)
                    os.makedirs(coreg_dir, exist_ok=True)
                    shutil.copyfile(potential_old_pansharpen, pansharpen_output)
                    shutil.copyfile(potential_old_coreg, co_reg_image)
                    df.at[idx, 'analysis_image'] = co_reg_image
                    df.at[idx, 'use_coregistered'] = True
                    logger.info("Copied legacy outputs → %s", co_reg_image)
                except Exception:
                    logger.exception("Legacy copy failed for %s", image)
                continue

            # Quality threshold
            try:
                score_val = float(df['model_scores'].iloc[idx])
            except Exception:
                score_val = 0.0
            if score_val < 0.335:
                logger.info("Skipped (score=%.3f): %s", score_val, image)
                continue

            # Read minimal metadata
            with rasterio.open(image) as src:
                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                width = src.width
                height = src.height
                count = src.count
                crs = src.crs

                x_res = transform.a
                y_res = -transform.e  # positive pixel size
                df.at[idx, 'xres'] = x_res
                df.at[idx, 'yres'] = y_res

            # Ensure pansharpen directory exists
            try:
                os.makedirs(os.path.join(os.path.dirname(image), 'pansharpen'), exist_ok=True)
            except Exception:
                logger.exception("Failed to create pansharpen directory for %s", image)

            if satname != 'PS' and satname != 'L7':
                # Pansharpen if missing
                if not os.path.isfile(pansharpen_output):
                    try:
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
                            for band in range(1, count + 1):
                                data = pansharpened_image[:, :, band - 1]
                                dst.write(data, band)
                        logger.info("Wrote pansharpened: %s", pansharpen_output)
                    except Exception:
                        logger.exception("Pansharpen write failed: %s", image)

                # Coregister pansharpened to reference (SILENCED)
                reference_image = df.at[idx, 'reference_image']
                if not os.path.isfile(co_reg_image):
                    try:
                        os.makedirs(coreg_dir, exist_ok=True)
                        # Silence all console output during coregistration
                        with _silence_console():
                            coregister_single.co_register_single(reference_image, pansharpen_output)
                        logger.info("Coregister attempted: ref=%s moving=%s", reference_image, pansharpen_output)
                    except Exception:
                        logger.exception("Coregistration failed: %s", pansharpen_output)

                # Outcome selection
                if os.path.isfile(co_reg_image):
                    df.at[idx, 'use_coregistered'] = True
                    df.at[idx, 'analysis_image'] = co_reg_image
                elif os.path.isfile(pansharpen_output):
                    df.at[idx, 'use_pansharpened'] = True
                    df.at[idx, 'analysis_image'] = pansharpen_output
                else:
                    df.at[idx, 'use_original'] = True
                    df.at[idx, 'analysis_image'] = image

            elif satname == 'PS':
                # Coreg only (original image) (SILENCED)
                reference_image = df.at[idx, 'reference_image']
                co_reg_image_ps = os.path.join(os.path.dirname(image), 'coregistered', os.path.basename(image))
                if not os.path.isfile(co_reg_image_ps):
                    try:
                        os.makedirs(os.path.dirname(co_reg_image_ps), exist_ok=True)
                        # Silence all console output during coregistration
                        with _silence_console():
                            coregister_single.co_register_single(reference_image, image)
                        logger.info("Coregister attempted (PS): ref=%s moving=%s", reference_image, image)
                    except Exception:
                        logger.exception("Coregistration (PS) failed: %s", image)
                if os.path.isfile(co_reg_image_ps):
                    df.at[idx, 'use_coregistered'] = True
                    df.at[idx, 'analysis_image'] = co_reg_image_ps
                else:
                    df.at[idx, 'use_original'] = True
                    df.at[idx, 'analysis_image'] = image

            else:  # L7
                if not os.path.isfile(pansharpen_output):
                    try:
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
                            for band in range(1, count + 1):
                                data = pansharpened_image[:, :, band - 1]
                                dst.write(data, band)
                        logger.info("Wrote pansharpened (L7): %s", pansharpen_output)
                    except Exception:
                        logger.exception("Pansharpen write failed (L7): %s", image)
                if os.path.isfile(pansharpen_output):
                    df.at[idx, 'use_pansharpened'] = True
                    df.at[idx, 'analysis_image'] = pansharpen_output
                else:
                    df.at[idx, 'use_original'] = True
                    df.at[idx, 'analysis_image'] = image

        except Exception:
            logger.exception("Unhandled exception for row idx=%d", idx)

        # Persist progressively
        try:
            df.to_csv(df_path, index=False)
        except Exception:
            logger.exception("CSV write failed: %s", df_path)

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

def image_suitability_section_old(g, c, rr, sss, r_home, gpu=0, planet=False):
    """
    Image suitability on shoreline section

    g (str): global region
    c (str): coastal area
    rr (str): subregion
    r_home (str): path to subregion
    gpu (int): which GPU to use (0 or 1 or -1 for CPU)
    """  
    section = 'SSS'+sss
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



def _get_logger_image_suitability(section_string: str, level: int = logging.INFO) -> logging.Logger:
    """
    File-only logger for image suitability runs:
      <cwd>/logs/image_suitability/<section_string>_<YYYYMMDD_%H%M%S>.log

    - Creates the log directory if missing.
    - Uses a timestamped file per invocation (no rotation).
    - Prevents duplicate handlers if requested multiple times.
    - No console handler (no stdout/stderr output).
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'image_suitability')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"image_suitability.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False  # prevent bubbling to root/global handlers

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler only
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger




def image_suitability_section(g, c, rr, sss, r_home, gpu=0, planet=False):
    """
    Image suitability on shoreline section

    Parameters
    ----------
    g : str
        Global region
    c : str
        Coastal area
    rr : str
        Subregion
    sss : str
        Shoreline section suffix
    r_home : str
        Path to subregion home directory
    gpu : int
        Which GPU to use (0 or 1 or -1 for CPU)
    planet : bool
        If True, use Planet imagery; else Landsat/Sentinel

    Returns
    -------
    str | None
        Path to the scored CSV that was written, or None if the run failed.
    """

    # Compose section identifiers and logger name
    section = 'SSS' + sss
    section_dir = os.path.join(r_home, section)
    section_string = g + c + rr + sss

    # Use the timestamped image suitability logger (file-only)
    logger = _get_logger_image_suitability(section_string, level=logging.INFO)

    try:
        logger.info("=== Image Suitability Start ===")
        logger.info("section=%s | section_dir=%s | section_string=%s | planet=%s | gpu=%s",
                    section, section_dir, section_string, planet, gpu)

        # Select input list source and outputs
        if planet:
            list_path = os.path.join(section_dir, f"{section_string}_ms_lists", "planet_ms_paths.csv")
            result_path = os.path.join(section_dir, f"{section_string}_ms_lists", "planet_image_suitability.csv")
            scored_path = os.path.join(section_dir, f"{section_string}_ms_lists", "planet_ms_paths_scored.csv")
            source_label = "planet"
        else:
            list_path = os.path.join(section_dir, f"{section_string}_ms_lists", "landsat_sentinel_ms_paths.csv")
            result_path = os.path.join(section_dir, f"{section_string}_ms_lists", "landsat_sentinel_image_suitability.csv")
            scored_path = os.path.join(section_dir, f"{section_string}_ms_lists", "landsat_sentinel_ms_paths_scored.csv")
            source_label = "landsat_sentinel"

        logger.info("Reading input list (%s): %s", source_label, list_path)
        sat_image_list_df = pd.read_csv(list_path).dropna()
        logger.info("Loaded %d rows after dropna()", len(sat_image_list_df))

        # Model checkpoint
        path_to_model_ckpt = os.path.join(get_script_path(), 'ShorelineFilter', 'models', 'image_rgb', 'best.h5')
        logger.info("Model checkpoint: %s", path_to_model_ckpt)

        # Run inference
        logger.info("Running image_filter.run_inference_rgb -> result_path=%s", result_path)
        image_filter.run_inference_rgb(
            path_to_model_ckpt,
            sat_image_list_df,
            None,
            result_path,
            sort=False,
            input_df=True,
            gpu=gpu
        )
        logger.info("Inference completed")

        # Read inference results
        good_bad_result = pd.read_csv(result_path)
        logger.info("Inference results loaded: %d rows", len(good_bad_result))

        # Merge and save scored CSV
        sat_image_list_df_merged = pd.merge(
            sat_image_list_df,
            good_bad_result,
            left_on='ms_tiff_path',
            right_on='im_paths'
        )
        logger.info("Merged dataframe rows: %d", len(sat_image_list_df_merged))

        sat_image_list_df_merged.to_csv(scored_path, index=False)
        logger.info("Wrote scored CSV: %s", scored_path)

        # Summary (robust to missing is_good)
        n_good = int((sat_image_list_df_merged.get('is_good', pd.Series([0] * len(sat_image_list_df_merged))) == 1).sum())
        n_bad = int((sat_image_list_df_merged.get('is_good', pd.Series([1] * len(sat_image_list_df_merged))) == 0).sum())
        logger.info("Summary (%s): good=%d | bad=%d | total=%d",
                    source_label, n_good, n_bad, len(sat_image_list_df_merged))

        logger.info("=== Image Suitability End ===")
        return scored_path

    except Exception as e:
        # Log error to file only; return None to avoid console traceback
        logger.exception("image_suitability_section failed: %s", str(e))
        return None



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

def segment_imagery_section_old(g, c, rr, sss, r_home, model = 'global', gpu=0, save_seg_to_raster=True, planet=False):
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


def get_segmentation_logger(section_string: str) -> logging.Logger:
    """
    Configure and return a logger that writes to:
      os.path.join(os.getcwd(), 'logs', 'segmentation', '<section_string>_<YYYYMMDD_HHMMSS>.log')
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'segmentation')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    # Unique logger per run (includes timestamp)
    logger_name = f"segmentation.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # If console output is desired, uncomment:
    # ch = logging.StreamHandler()
    # ch.setFormatter(fmt)
    # logger.addHandler(ch)

    return logger


def segment_imagery_section(
    g, c, rr, sss, r_home,
    model='global', gpu=0, save_seg_to_raster=True, planet=False
):
    """
    Segments imagery for single shoreline section.

    Inputs:
      g (str): global region
      c (str): coastal area
      rr (str): subregion
      sss (str): shoreline section
      r_home (str): path/to/g#/c#/rr##/

    Input CSVs:
      Planet:   <section_dir>/<section_string>_ms_lists/planet_ms_paths_scored_update.csv
      L/S:      <section_dir>/<section_string>_ms_lists/landsat_sentinel_ms_paths_scored_update.csv
    Output CSV:
      Planet:   <section_dir>/<section_string>_ms_lists/planet_ms_paths_scored_segmented.csv
      L/S:      <section_dir>/<section_string>_ms_lists/landsat_sentinel_ms_paths_scored_segmented.csv

    Logging:
      <cwd>/logs/segmentation/<section_string>_<timestamp>.log
    """
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_segmentation_logger(section_string)

    # Select CSV paths
    if planet is True:
        src_csv = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_update.csv')
        out_csv = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if not os.path.isfile(src_csv):
            logger.warning("Planet: source CSV not found → %s", src_csv)
            return
    else:
        src_csv = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_update.csv')
        out_csv = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')

    # Model setup (CoastSeg zoo_model)
    if model == 'ak':
        model_str = "ak_segformer_RGB_4class_14037041"
        logger.info("Using AK model")
    else:
        model_str = "global_segformer_RGB_4class_14036903"
        logger.info("Using global RGB model")

    settings = {
        'min_length_sl': 50,
        'max_dist_ref': 400,
        'cloud_thresh': 0.8,
        'dist_clouds': 200,
        'min_beach_area': 100,
        'sand_color': 'default',
        "apply_cloud_mask": False,
        "sample_direc": None,
        "use_GPU": "1",                 # "1" to use GPU (aligned with CoastSeg settings style)
        "implementation": "BEST",
        "model_type": model_str,
        "otsu": False,
        "tta": False,
        "use_local_model": False,
        "local_model_path": None,
        "img_type": "RGB",
    }

    zoo_model_instance = zoo_model.Zoo_Model()
    zoo_model_instance.set_settings(**settings)
    zoo_model_instance.prepare_model('BEST', model_str)
    model_list = zoo_model_instance.model_list
    logger.info("Model prepared: %s", model_str)

    # Load source CSV
    df = pd.read_csv(src_csv)
    n_total = len(df)
    if 'seg_paths' not in df.columns:
        df['seg_paths'] = [None] * n_total
    logger.info("Loaded CSV: rows=%d → %s", n_total, src_csv)

    # Filter by score (>= 0.335)
    try:
        mask_sel = df['model_scores'].astype(float) >= 0.335
    except Exception:
        mask_sel = np.zeros(n_total, dtype=bool)
        logger.warning("Column 'model_scores' missing/invalid; no images selected.")

    df_to_seg = df.loc[mask_sel].copy()
    if 'done' not in df_to_seg.columns:
        df_to_seg['done'] = [None] * len(df_to_seg)

    seg_dir = os.path.join(section_dir, 'segmentation')
    os.makedirs(seg_dir, exist_ok=True)

    processed = 0

    # Progress bar over selected rows
    for i in tqdm(df_to_seg.index, desc=f"Segmenting {section_string}", unit="img", total=len(df_to_seg)):
        try:
            image = df_to_seg.at[i, 'analysis_image']
            if image is None or (isinstance(image, float) and np.isnan(image)):
                df_to_seg.at[i, 'done'] = True
                continue

            basename = os.path.splitext(os.path.basename(image))[0]
            color_seg_lab_path = os.path.join(seg_dir, basename + '_seg.png')

            if df_to_seg.at[i, 'done'] is True:
                df_to_seg.at[i, 'seg_paths'] = color_seg_lab_path
                continue

            satname = df_to_seg.at[i, 'satnames']

            # Read imagery
            with rasterio.open(image) as src:
                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                width = src.width
                height = src.height
                count = src.count
                crs = src.crs

                blue = src.read(1)
                green = src.read(2)
                red = src.read(3)
                nir = src.read(4)
                swir = src.read(5) if (satname != 'PS' and count >= 5) else None

            # Nodata→0
            if mask_value is not None:
                blue[blue == mask_value] = 0
                green[green == mask_value] = 0
                red[red == mask_value] = 0
                nir[nir == mask_value] = 0
                if swir is not None:
                    swir[swir == mask_value] = 0

            # Resolution from transform
            x_res = transform.a
            y_res = -transform.e
            df_to_seg.at[i, 'xres'] = x_res
            df_to_seg.at[i, 'yres'] = y_res

            # Original normalization path (expects rescale to be available in your environment)
            rgb = np.dstack([red, green, blue])
            array_to_seg = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
            array_to_seg = (rescale(array_to_seg) * 255).astype('uint8')  # keep original behavior

            # Segment
            seg_lab, color_seg_lab = do_seg_array(array_to_seg, model_list, gpu=gpu)

            # Resize segmentation to RGB shape (use nearest-neighbor for labels)
            height_p, width_p, _ = array_to_seg.shape
            seg_lab = resize(seg_lab, (height_p, width_p), order=0, preserve_range=True, anti_aliasing=False).astype(seg_lab.dtype)

            # Simplify segmented labels
            seg_lab[seg_lab == 1] = 0
            seg_lab[seg_lab > 0] = 1

            # Save color PNG
            Image.fromarray(color_seg_lab).save(color_seg_lab_path)
            df_to_seg.at[i, 'seg_paths'] = color_seg_lab_path

            # Thresholds and binary masks (NIR, SWIR) — original method
            nir = np.nan_to_num(nir, nan=0.0, posinf=0.0, neginf=0.0)
            thr_nir = threshold_multiotsu(nir)
            binary_nir = (nir > float(np.min(thr_nir))).astype(np.uint8)
            binary_nir = scipy.ndimage.median_filter(binary_nir, size=5)

            if satname != 'PS' and swir is not None:
                swir = np.nan_to_num(swir, nan=0.0, posinf=0.0, neginf=0.0)
                thr_swir = threshold_multiotsu(swir)
                binary_swir = (swir > float(np.min(thr_swir))).astype(np.uint8)
                binary_swir = scipy.ndimage.median_filter(binary_swir, size=5)
            else:
                binary_swir = None

            # Optional: save segmentation into raster (writes to same path)
            if save_seg_to_raster:
                if satname != 'PS' and binary_swir is not None:
                    data_to_save = np.dstack([
                        blue, green, red, nir, swir,
                        seg_lab.astype(np.float32),
                        binary_nir.astype(np.float32),
                        binary_swir.astype(np.float32),
                    ])
                    out_count = 8
                else:
                    data_to_save = np.dstack([
                        blue, green, red, nir,
                        seg_lab.astype(np.float32),
                        binary_nir.astype(np.float32),
                    ])
                    out_count = 6

                try:
                    with rasterio.open(
                        image, 'w',
                        driver='GTiff',
                        height=height, width=width,
                        dtype=np.float32,
                        crs=crs, transform=transform,
                        nodata=mask_value, count=out_count
                    ) as dst:
                        for band in range(1, out_count + 1):
                            dst.write(data_to_save[:, :, band - 1], band)
                    logger.info("Wrote segmentation bands → %s", image)
                except Exception:
                    logger.exception("Failed writing segmentation to raster → %s", image)

            df_to_seg.at[i, 'done'] = True
            processed += 1

            # Progressive CSV write (atomic replace)
            try:
                tmp_path = out_csv + ".tmp"
                df_to_seg.to_csv(tmp_path, index=False)
                os.replace(tmp_path, out_csv)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", out_csv)

        except Exception:
            logger.exception("Unhandled exception at index=%s", str(i))

    # Final write and summary
    try:
        df_to_seg.to_csv(out_csv, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", out_csv)

    logger.info("Summary: selected=%d, processed=%d, output CSV → %s", len(df_to_seg), processed, out_csv)


def segmentation_suitability_section_old(g, c, rr, sss, r_home, gpu=0, planet=False):
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

def get_segmentation_suitability_logger(section_string: str) -> logging.Logger:
    """
    Logger for segmentation suitability:
      <cwd>/logs/segmentation_suitability/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'segmentation_suitability')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"segmentation_suitability.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def segmentation_suitability_section(g, c, rr, sss, r_home, gpu=0, planet=False):
    """
    Runs segmentation suitability for a single shoreline section.

    Inputs:
      g (str): global region
      c (str): coastal area
      rr (str): subregion
      sss (str): shoreline section
      r_home (str): path/to/g#/c#/rr##/
      gpu (int): -1 for cpu, 0, 1, etc.
      planet (bool): whether or not to process planet data

    Logging:
      <cwd>/logs/segmentation_suitability/<section_string>_<timestamp>.log
    """
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_segmentation_suitability_logger(section_string)

    # Select CSV path
    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if not os.path.isfile(sat_image_list_df_path):
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')

    # Load CSV
    try:
        df = pd.read_csv(sat_image_list_df_path)
    except Exception:
        logger.exception("Failed to read CSV → %s", sat_image_list_df_path)
        return

    n_total = len(df)
    n_scored = 0
    n_missing = 0
    n_error = 0

    # If no rows, still log and exit
    if n_total == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    # Iterate rows
    for i in tqdm(range(n_total)):
        # Safely get path (column may be missing)
        color_seg_lab_path = df['seg_paths'].iloc[i] if 'seg_paths' in df.columns else None

        # Skip missing/null paths
        if color_seg_lab_path is None or (isinstance(color_seg_lab_path, float) and np.isnan(color_seg_lab_path)):
            n_missing += 1
            continue

        try:
            # Load color segmentation PNG and compute suitability score
            color_seg_lab = np.array(Image.open(color_seg_lab_path))
            seg_score = image_segmentation_filter.get_segmentation_score(color_seg_lab, gpu=gpu)
            df.at[i, 'seg_scores'] = seg_score
            n_scored += 1

            # Progressive CSV write (atomic replace)
            try:
                tmp_path = sat_image_list_df_path + ".tmp"
                df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, sat_image_list_df_path)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path)

        except Exception:
            n_error += 1
            logger.exception("Segmentation suitability error at index=%d, path=%s", i, str(color_seg_lab_path))

    # Final write and cleanup
    try:
        df.to_csv(sat_image_list_df_path, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path)

    try:
        gc.collect()
    except Exception:
        logger.exception("gc.collect() failed")

    # Completed summary log (finished properly)
    logger.info(
        "Summary: rows=%d, scored=%d, missing_paths=%d, errors=%d, CSV → %s",
        n_total, n_scored, n_missing, n_error, sat_image_list_df_path
    )

#@profile
def extract_shorelines_after_segmentation_section_old(g, c, rr, sss, r_home, reset=False, planet=False):
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



def get_shoreline_extraction_logger(section_string: str) -> logging.Logger:
    """
    Logger for shoreline extraction:
      <cwd>/logs/shoreline_extraction/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'shoreline_extraction')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"shoreline_extraction.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def extract_shorelines_after_segmentation_section_old2(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in GeoTIFFs.

    Inputs:
      g (str): global region
      c (str): coastal area
      rr (str): subregion
      sss (str): shoreline section
      r_home (str): path/to/g#/c#/rr##/

    Logging:
      <cwd>/logs/shoreline_extraction/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_shoreline_extraction_logger(section_string)

    # Reference shoreline and polygon (reproject to UTM)
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    except Exception:
        logger.exception("Failed to read reference shoreline → %s", reference_shoreline)
        return
    try:
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    except Exception:
        logger.exception("Failed to reproject reference shoreline to UTM.")
        return
    crs = reference_shoreline_gdf.crs

    # Select CSV paths
    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path):
            sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
        else:
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    # Load segmentation CSV
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    except Exception:
        logger.exception("Failed to read segmentation CSV → %s", sat_image_list_df_path)
        return

    # Output directories
    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
    for d in (shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
        os.makedirs(d, exist_ok=True)

    num_images = len(sat_image_list_df)
    if num_images == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    # Reset handling
    if reset is True:
        sat_image_list_df['shoreline_done'] = [None] * num_images
        for d in (zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
            try:
                shutil.rmtree(d)
            except Exception:
                # directory may not exist; ignore
                pass
        logger.info("Reset: cleared shoreline_done and removed output directories.")
    else:
        if 'shoreline_done' not in sat_image_list_df.columns:
            # fallback to 'done' if present; else initialize
            if 'done' in sat_image_list_df.columns:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['done']
            else:
                sat_image_list_df['shoreline_done'] = [None] * num_images

    # Cache of existing zoo geojsons to skip re-write
    zoo_shorelines_list = set(glob.glob(os.path.join(zoo_shoreline_dir, '*.geojson')))

    processed = 0
    skipped_score = 0
    skipped_missing = 0
    already_done = 0
    errors = 0

    for i in tqdm(range(len(sat_image_list_df['analysis_image'])),
                  desc=f"Extracting {section_string}",
                  unit="img",
                  total=len(sat_image_list_df['analysis_image'])):
        try:
            image = sat_image_list_df['analysis_image'].iloc[i]
            roi_folder = sat_image_list_df['roi_folder'].iloc[i]

            # Already marked done
            if sat_image_list_df['shoreline_done'].iloc[i] is True:
                already_done += 1
                # progressive write
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip low suitability
            try:
                image_suitability_score = float(sat_image_list_df['model_scores'].iloc[i])
            except Exception:
                image_suitability_score = 0.0
            if image_suitability_score < 0.335:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_score += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip missing analysis image
            if image is None or (isinstance(image, float) and np.isnan(image)):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_missing += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            satname = sat_image_list_df['satnames'].iloc[i]
            date = sat_image_list_df['datetimes_utc'].iloc[i]
            # Standardize date for filename
            try:
                check_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d-%H-%M-%S")
            except Exception:
                # Fallback: use raw date string with unsafe chars replaced
                check_date = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")

            # If output exists, mark done
            if shoreline_path in zoo_shorelines_list or os.path.isfile(shoreline_path):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Load raster bands (segmented results saved in prior step)
            with rasterio.open(image) as src:
                # Bands layout:
                # non-PS: [blue,green,red,nir,swir,seg_lab,binary_nir,binary_swir]
                # PS:     [blue,green,red,nir,seg_lab,binary_nir]
                if satname != 'PS':
                    nir = src.read(4)
                    seg_lab = src.read(6)
                    binary_image_nir = src.read(7)
                    binary_image_swir = src.read(8)
                else:
                    nir = src.read(4)
                    seg_lab = src.read(5)
                    binary_image_nir = src.read(6)
                    binary_image_swir = None

                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                bounds = src.bounds
                crs_raster = src.crs

                # Pixel size from transform
                x_res = transform.a
                y_res = -transform.e
                xmin = bounds.left
                ymax = bounds.top

                # Valid-data mask
                if mask_value is not None:
                    mask = nir != mask_value
                else:
                    mask = None

                # Build polygons for valid/no-data
                data_polygon = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for _, (s, v) in enumerate(shapes(nir, mask=mask, transform=src.transform))
                )
                data_polygon = gpd.GeoDataFrame.from_features(list(data_polygon), crs=src.crs)

                if mask_value is not None:
                    mask = nir == mask_value
                else:
                    mask = None

                no_data_polygon = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for _, (s, v) in enumerate(shapes(nir, mask=mask, transform=src.transform))
                )
                try:
                    no_data_polygon = gpd.GeoDataFrame.from_features(list(no_data_polygon), crs=src.crs)
                except Exception:
                    no_data_polygon = None

            # Buffer/difference polygons (avoid errors if no_data_polygon is None)
            try:
                if no_data_polygon is not None and len(no_data_polygon) > 0:
                    no_data_union = no_data_polygon.buffer(x_res * 2).unary_union
                else:
                    no_data_union = None

                data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                if data_union is not None and no_data_union is not None:
                    data_polygon_final = data_union.difference(no_data_union)
                else:
                    data_polygon_final = data_union
            except Exception:
                logger.exception("Polygon buffer/difference failed; proceeding with data_union as-is.")
                data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None

            # Build masks for arrays
            data_mask = nir != mask_value if mask_value is not None else np.ones_like(nir, dtype=bool)
            no_data_mask = nir == mask_value if mask_value is not None else np.zeros_like(nir, dtype=bool)

            # Ensure segmentation has nodata cleared
            seg_lab = seg_lab.astype(np.float32)
            seg_lab[no_data_mask] = np.nan

            # Extract contours for each source
            try:
                linestrings = get_contours(
                    seg_lab, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for seg_lab → %s", image)
                linestrings = []

            try:
                linestrings_nir = get_contours(
                    binary_image_nir, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for binary NIR → %s", image)
                linestrings_nir = []

            if satname != 'PS' and binary_image_swir is not None:
                try:
                    linestrings_swir = get_contours(
                        binary_image_swir, satname, xmin, ymax, x_res, y_res,
                        data_polygon_final, reference_shoreline_gdf,
                        ref_shore_buffer, reference_polygon,
                        data_mask, crs
                    )
                except Exception:
                    logger.exception("get_contours failed for binary SWIR → %s", image)
                    linestrings_swir = []
            else:
                linestrings_swir = []

            # Scores
            image_score = sat_image_list_df['model_scores'].iloc[i]
            seg_score = sat_image_list_df['seg_scores'].iloc[i] if 'seg_scores' in sat_image_list_df.columns else None

            # Filter by minimum coordinate count
            waterlines = [None] * len(linestrings)
            shoreline_seg_scores = [None] * len(linestrings)
            shoreline_image_scores = [None] * len(linestrings)

            k = 0
            for ls in linestrings:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines[k] = ls
                    shoreline_seg_scores[k] = seg_score
                    shoreline_image_scores[k] = image_score
                k += 1
            remove_nones(waterlines)
            remove_nones(shoreline_seg_scores)
            remove_nones(shoreline_image_scores)

            waterlines_nir = [None] * len(linestrings_nir)
            shoreline_seg_scores_nir = [None] * len(linestrings_nir)
            shoreline_image_scores_nir = [None] * len(linestrings_nir)

            k = 0
            for ls in linestrings_nir:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines_nir[k] = ls
                    shoreline_seg_scores_nir[k] = seg_score
                    shoreline_image_scores_nir[k] = image_score
                k += 1
            remove_nones(waterlines_nir)
            remove_nones(shoreline_seg_scores_nir)
            remove_nones(shoreline_image_scores_nir)

            waterlines_swir = []
            shoreline_seg_scores_swir = []
            shoreline_image_scores_swir = []
            if satname != 'PS':
                waterlines_swir = [None] * len(linestrings_swir)
                shoreline_seg_scores_swir = [None] * len(linestrings_swir)
                shoreline_image_scores_swir = [None] * len(linestrings_swir)
                k = 0
                for ls in linestrings_swir:
                    coords = LineString_to_arr(ls)
                    if len(coords) > 5:
                        waterlines_swir[k] = ls
                        shoreline_seg_scores_swir[k] = seg_score
                        shoreline_image_scores_swir[k] = image_score
                    k += 1
                remove_nones(waterlines_swir)
                remove_nones(shoreline_seg_scores_swir)
                remove_nones(shoreline_image_scores_swir)

            # If none extracted, mark done and continue
            if len(waterlines) == 0 or len(waterlines_nir) == 0:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                logger.info("No shorelines extracted for %s", image)
                continue

            # Save RGB-based shorelines (zoo)
            try:
                shorelines_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines),
                        'image_suitability_score': shoreline_image_scores,
                        'segmentation_suitability_score': shoreline_seg_scores,
                        'satname': [satname] * len(waterlines)
                    },
                    geometry=waterlines, crs=crs
                )
                shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_gdf = utm_to_wgs84_df(shorelines_gdf).reset_index(drop=True)
                shorelines_gdf = split_line(shorelines_gdf, 'LineString', smooth=True)
                shorelines_gdf.to_file(shoreline_path)
                logger.info("Saved RGB shorelines → %s", shoreline_path)
            except Exception:
                logger.exception("Failed saving RGB shorelines → %s", image)

            # Save NIR threshold shorelines
            try:
                nir_shoreline_path = os.path.join(nir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_nir_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines_nir),
                        'image_suitability_score': shoreline_image_scores_nir,
                        'segmentation_suitability_score': shoreline_seg_scores_nir,
                        'satname': [satname] * len(waterlines_nir),
                    },
                    geometry=waterlines_nir, crs=crs
                ).reset_index(drop=True)
                shorelines_nir_gdf = utm_to_wgs84_df(shorelines_nir_gdf)
                shorelines_nir_gdf = split_line(shorelines_nir_gdf, 'LineString', smooth=True)
                shorelines_nir_gdf.to_file(nir_shoreline_path)
                logger.info("Saved NIR shorelines → %s", nir_shoreline_path)
            except Exception:
                logger.exception("Failed saving NIR shorelines → %s", image)

            # Save SWIR threshold shorelines (non-PS only)
            if satname != 'PS' and len(waterlines_swir) > 0:
                try:
                    swir_shoreline_path = os.path.join(swir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                    shorelines_swir_gdf = gpd.GeoDataFrame(
                        {
                            'dates': [check_date] * len(waterlines_swir),
                            'image_suitability_score': shoreline_image_scores_swir,
                            'segmentation_suitability_score': shoreline_seg_scores_swir,
                            'satname': [satname] * len(waterlines_swir),
                        },
                        geometry=waterlines_swir, crs=crs
                    ).reset_index(drop=True)
                    shorelines_swir_gdf = utm_to_wgs84_df(shorelines_swir_gdf)
                    shorelines_swir_gdf = split_line(shorelines_swir_gdf, 'LineString', smooth=True)
                    shorelines_swir_gdf.to_file(swir_shoreline_path)
                    logger.info("Saved SWIR shorelines → %s", swir_shoreline_path)
                except Exception:
                    logger.exception("Failed saving SWIR shorelines → %s", image)

            # Mark done and persist
            sat_image_list_df.at[i, 'shoreline_done'] = True
            processed += 1
            try:
                tmp_path = sat_image_list_df_path_shore + ".tmp"
                sat_image_list_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, sat_image_list_df_path_shore)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception at index=%d", i)

        # GC
        try:
            gc.collect()
        except Exception:
            logger.exception("gc.collect() failed")

    # Final write and summary
    try:
        sat_image_list_df.to_csv(sat_image_list_df_path_shore, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path_shore)

    logger.info(
        "Summary: rows=%d, processed=%d, already_done=%d, skipped_low_score=%d, "
        "skipped_missing_image=%d, errors=%d, CSV → %s",
        num_images, processed, already_done, skipped_score, skipped_missing, errors, sat_image_list_df_path_shore
    )


def extract_shorelines_after_segmentation_section_old3(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in GeoTIFFs.

    Inputs:
      g (str): global region
      c (str): coastal area
      rr (str): subregion
      sss (str): shoreline section
      r_home (str): path/to/g#/c#/rr##/

    Logging:
      <cwd>/logs/shoreline_extraction/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_shoreline_extraction_logger(section_string)

    # Reference shoreline and polygon (reproject to UTM)
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    except Exception:
        logger.exception("Failed to read reference shoreline → %s", reference_shoreline)
        return
    try:
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    except Exception:
        logger.exception("Failed to reproject reference shoreline to UTM.")
        return
    crs = reference_shoreline_gdf.crs

    # Select CSV paths
    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path):
            sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
        else:
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    # Load segmentation CSV
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    except Exception:
        logger.exception("Failed to read segmentation CSV → %s", sat_image_list_df_path)
        return

    # Output directories
    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
    for d in (shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
        os.makedirs(d, exist_ok=True)

    num_images = len(sat_image_list_df)
    if num_images == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    # Reset handling
    if reset is True:
        sat_image_list_df['shoreline_done'] = [None] * num_images
        for d in (zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
            try:
                shutil.rmtree(d)
            except Exception:
                # directory may not exist; ignore
                pass
        logger.info("Reset: cleared shoreline_done and removed output directories.")
    else:
        if 'shoreline_done' not in sat_image_list_df.columns:
            # fallback to 'done' if present; else initialize
            if 'done' in sat_image_list_df.columns:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['done']
            else:
                sat_image_list_df['shoreline_done'] = [None] * num_images

    # Cache of existing zoo geojsons to skip re-write
    zoo_shorelines_list = set(glob.glob(os.path.join(zoo_shoreline_dir, '*.geojson')))

    processed = 0
    skipped_score = 0
    skipped_missing = 0
    already_done = 0
    errors = 0

    # Main loop
    for i in tqdm(range(len(sat_image_list_df['analysis_image'])),
                  desc=f"Extracting {section_string}",
                  unit="img",
                  total=len(sat_image_list_df['analysis_image'])):
        try:
            # Row-local variables to reduce iloc overhead
            row = sat_image_list_df.iloc[i]
            image = row['analysis_image']
            roi_folder = row['roi_folder']

            # Already marked done
            if sat_image_list_df['shoreline_done'].iloc[i] is True:
                already_done += 1
                # progressive write
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False, lineterminator='\n')
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip low suitability
            try:
                image_suitability_score = float(row['model_scores'])
            except Exception:
                image_suitability_score = 0.0
            if image_suitability_score < 0.335:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_score += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False, lineterminator='\n')
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip missing analysis image
            if image is None or (isinstance(image, float) and np.isnan(image)):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_missing += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False, lineterminator='\n')
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            satname = row['satnames']
            date = row['datetimes_utc']

            # Standardize date for filename
            try:
                check_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d-%H-%M-%S")
            except Exception:
                # Fallback: use raw date string with unsafe chars replaced
                check_date = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")

            # If output exists, mark done
            if shoreline_path in zoo_shorelines_list or os.path.isfile(shoreline_path):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False, lineterminator='\n')
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Load raster bands (segmented results saved in prior step)
            with rasterio.open(image) as src:
                # Bands layout:
                # non-PS: [blue,green,red,nir,swir,seg_lab,binary_nir,binary_swir]
                # PS:     [blue,green,red,nir,seg_lab,binary_nir]
                if satname != 'PS':
                    nir = src.read(4, masked=True)
                    seg_lab = src.read(6, masked=True)
                    binary_image_nir = src.read(7, masked=True)
                    binary_image_swir = src.read(8, masked=True)
                else:
                    nir = src.read(4, masked=True)
                    seg_lab = src.read(5, masked=True)
                    binary_image_nir = src.read(6, masked=True)
                    binary_image_swir = None

                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                bounds = src.bounds
                crs_raster = src.crs

                # Pixel size from transform
                x_res = transform.a
                y_res = -transform.e
                xmin = bounds.left
                ymax = bounds.top

                # Build masks based on masked read (identical semantics)
                data_mask = ~nir.mask
                no_data_mask = nir.mask

                # ---- Polygonization: single pass on a binary valid/nodata label (same union/diff area) ----
                try:
                    valid_label = np.where(nir.mask, 0, 1).astype(np.uint8)
                    polys = list(raster_shapes(valid_label, mask=None, transform=src.transform, connectivity=8))
                    valid_geoms = [s for (s, v) in polys if v == 1]
                    nodata_geoms = [s for (s, v) in polys if v == 0]

                    data_polygon = gpd.GeoDataFrame(geometry=valid_geoms, crs=src.crs)
                    no_data_polygon = gpd.GeoDataFrame(geometry=nodata_geoms, crs=src.crs) if nodata_geoms else None
                except Exception:
                    # Fallback to original two-pass polygonization if any issue occurs
                    try:
                        # Valid-data polygons
                        mask = nir.data != mask_value if mask_value is not None else None
                        data_features = (
                            {'properties': {'raster_val': v}, 'geometry': s}
                            for _, (s, v) in enumerate(raster_shapes(nir.data, mask=mask, transform=src.transform))
                        )
                        data_polygon = gpd.GeoDataFrame.from_features(list(data_features), crs=src.crs)

                        # No-data polygons
                        mask = nir.data == mask_value if mask_value is not None else None
                        nd_features = (
                            {'properties': {'raster_val': v}, 'geometry': s}
                            for _, (s, v) in enumerate(raster_shapes(nir.data, mask=mask, transform=src.transform))
                        )
                        no_data_polygon = gpd.GeoDataFrame.from_features(list(nd_features), crs=src.crs) if mask is not None else None
                    except Exception:
                        logger.exception("Fallback polygonization failed")
                        data_polygon = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                        no_data_polygon = None

            # Buffer/difference polygons (unchanged semantics)
            try:
                if no_data_polygon is not None and len(no_data_polygon) > 0:
                    no_data_union = no_data_polygon.buffer(x_res * 2).unary_union
                else:
                    no_data_union = None

                data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                if data_union is not None and no_data_union is not None:
                    data_polygon_final = data_union.difference(no_data_union)
                else:
                    data_polygon_final = data_union
            except Exception:
                logger.exception("Polygon buffer/difference failed; proceeding with data_union as-is.")
                data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None

            # Ensure segmentation arrays for contours (keep methodology: pass mask to find_contours)
            seg_lab_arr = seg_lab.data          # no dtype cast / NaN fill needed; mask controls nodata
            binary_nir_arr = binary_image_nir.data
            binary_swir_arr = binary_image_swir.data if binary_image_swir is not None else None

            # Extract contours for each source (same calls, same args)
            try:
                linestrings = get_contours(
                    seg_lab_arr, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for seg_lab → %s", image)
                linestrings = []

            try:
                linestrings_nir = get_contours(
                    binary_nir_arr, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for binary NIR → %s", image)
                linestrings_nir = []

            if satname != 'PS' and binary_swir_arr is not None:
                try:
                    linestrings_swir = get_contours(
                        binary_swir_arr, satname, xmin, ymax, x_res, y_res,
                        data_polygon_final, reference_shoreline_gdf,
                        ref_shore_buffer, reference_polygon,
                        data_mask, crs
                    )
                except Exception:
                    logger.exception("get_contours failed for binary SWIR → %s", image)
                    linestrings_swir = []
            else:
                linestrings_swir = []

            # Scores
            image_score = row['model_scores']
            seg_score = row['seg_scores'] if 'seg_scores' in sat_image_list_df.columns else None

            # Filter by minimum coordinate count (direct list buildup; same filter)
            waterlines = []
            shoreline_seg_scores = []
            shoreline_image_scores = []
            for ls in linestrings:
                try:
                    if len(ls.coords) > 5:
                        waterlines.append(ls)
                        shoreline_seg_scores.append(seg_score)
                        shoreline_image_scores.append(image_score)
                except Exception:
                    continue

            waterlines_nir = []
            shoreline_seg_scores_nir = []
            shoreline_image_scores_nir = []
            for ls in linestrings_nir:
                try:
                    if len(ls.coords) > 5:
                        waterlines_nir.append(ls)
                        shoreline_seg_scores_nir.append(seg_score)
                        shoreline_image_scores_nir.append(image_score)
                except Exception:
                    continue

            waterlines_swir = []
            shoreline_seg_scores_swir = []
            shoreline_image_scores_swir = []
            if satname != 'PS':
                for ls in linestrings_swir:
                    try:
                        if len(ls.coords) > 5:
                            waterlines_swir.append(ls)
                            shoreline_seg_scores_swir.append(seg_score)
                            shoreline_image_scores_swir.append(image_score)
                    except Exception:
                        continue

            # If none extracted, mark done and continue
            if len(waterlines) == 0 or len(waterlines_nir) == 0:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False, lineterminator='\n')
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                logger.info("No shorelines extracted for %s", image)
                continue

            # Save RGB-based shorelines (zoo) — unchanged downstream processing
            try:
                shorelines_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines),
                        'image_suitability_score': shoreline_image_scores,
                        'segmentation_suitability_score': shoreline_seg_scores,
                        'satname': [satname] * len(waterlines)
                    },
                    geometry=waterlines, crs=crs
                )
                shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_gdf = utm_to_wgs84_df(shorelines_gdf).reset_index(drop=True)
                shorelines_gdf = split_line(shorelines_gdf, 'LineString', smooth=True)
                shorelines_gdf.to_file(shoreline_path)
                logger.info("Saved RGB shorelines → %s", shoreline_path)
            except Exception:
                logger.exception("Failed saving RGB shorelines → %s", image)

            # Save NIR threshold shorelines — unchanged
            try:
                nir_shoreline_path = os.path.join(nir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_nir_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines_nir),
                        'image_suitability_score': shoreline_image_scores_nir,
                        'segmentation_suitability_score': shoreline_seg_scores_nir,
                        'satname': [satname] * len(waterlines_nir),
                    },
                    geometry=waterlines_nir, crs=crs
                ).reset_index(drop=True)
                shorelines_nir_gdf = utm_to_wgs84_df(shorelines_nir_gdf)
                shorelines_nir_gdf = split_line(shorelines_nir_gdf, 'LineString', smooth=True)
                shorelines_nir_gdf.to_file(nir_shoreline_path)
                logger.info("Saved NIR shorelines → %s", nir_shoreline_path)
            except Exception:
                logger.exception("Failed saving NIR shorelines → %s", image)

            # Save SWIR threshold shorelines (non-PS only) — unchanged
            if satname != 'PS' and len(waterlines_swir) > 0:
                try:
                    swir_shoreline_path = os.path.join(swir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                    shorelines_swir_gdf = gpd.GeoDataFrame(
                        {
                            'dates': [check_date] * len(waterlines_swir),
                            'image_suitability_score': shoreline_image_scores_swir,
                            'segmentation_suitability_score': shoreline_seg_scores_swir,
                            'satname': [satname] * len(waterlines_swir),
                        },
                        geometry=waterlines_swir, crs=crs
                    ).reset_index(drop=True)
                    shorelines_swir_gdf = utm_to_wgs84_df(shorelines_swir_gdf)
                    shorelines_swir_gdf = split_line(shorelines_swir_gdf, 'LineString', smooth=True)
                    shorelines_swir_gdf.to_file(swir_shoreline_path)
                    logger.info("Saved SWIR shorelines → %s", swir_shoreline_path)
                except Exception:
                    logger.exception("Failed saving SWIR shorelines → %s", image)

            # Mark done and persist
            sat_image_list_df.at[i, 'shoreline_done'] = True
            processed += 1
            try:
                tmp_path = sat_image_list_df_path_shore + ".tmp"
                sat_image_list_df.to_csv(tmp_path, index=False, lineterminator='\n')
                os.replace(tmp_path, sat_image_list_df_path_shore)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception at index=%d", i)

        # (Optional) GC — remove to avoid per-iteration pauses; kept as try/except if you want it
        # try:
        #     gc.collect()
        # except Exception:
        #     logger.exception("gc.collect() failed")

    # Final write and summary
    try:
        sat_image_list_df.to_csv(sat_image_list_df_path_shore, index=False, lineterminator='\n')
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path_shore)

    logger.info(
        "Summary: rows=%d, processed=%d, already_done=%d, skipped_low_score=%d, "
        "skipped_missing_image=%d, errors=%d, CSV → %s",
        num_images, processed, already_done, skipped_score, skipped_missing, errors, sat_image_list_df_path_shore
    )


def extract_shorelines_after_segmentation_section_old4(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in GeoTIFFs.

    Inputs:
      g (str): global region
      c (str): coastal area
      rr (str): subregion
      sss (str): shoreline section
      r_home (str): path/to/g#/c#/rr##/

    Logging:
      <cwd>/logs/shoreline_extraction/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_shoreline_extraction_logger(section_string)

    # Reference shoreline and polygon (reproject to UTM)
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    except Exception:
        logger.exception("Failed to read reference shoreline → %s", reference_shoreline)
        return
    try:
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    except Exception:
        logger.exception("Failed to reproject reference shoreline to UTM.")
        return
    crs = reference_shoreline_gdf.crs

    # Select CSV paths
    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path):
            sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
        else:
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    # Load segmentation CSV
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    except Exception:
        logger.exception("Failed to read segmentation CSV → %s", sat_image_list_df_path)
        return

    # Output directories
    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
    for d in (shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
        os.makedirs(d, exist_ok=True)

    num_images = len(sat_image_list_df)
    if num_images == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    # Reset handling
    if reset is True:
        sat_image_list_df['shoreline_done'] = [None] * num_images
        for d in (zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
            try:
                shutil.rmtree(d)
            except Exception:
                # directory may not exist; ignore
                pass
        logger.info("Reset: cleared shoreline_done and removed output directories.")
    else:
        if 'shoreline_done' not in sat_image_list_df.columns:
            # fallback to 'done' if present; else initialize
            if 'done' in sat_image_list_df.columns:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['done']
            else:
                sat_image_list_df['shoreline_done'] = [None] * num_images

    # Cache of existing zoo geojsons to skip re-write
    zoo_shorelines_list = set(glob.glob(os.path.join(zoo_shoreline_dir, '*.geojson')))

    processed = 0
    skipped_score = 0
    skipped_missing = 0
    already_done = 0
    errors = 0

    for i in tqdm(range(len(sat_image_list_df['analysis_image'])),
                  desc=f"Extracting {section_string}",
                  unit="img",
                  total=len(sat_image_list_df['analysis_image'])):
        try:
            image = sat_image_list_df['analysis_image'].iloc[i]
            roi_folder = sat_image_list_df['roi_folder'].iloc[i]

            # Already marked done
            if sat_image_list_df['shoreline_done'].iloc[i] is True:
                already_done += 1
                # progressive write
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip low suitability
            try:
                image_suitability_score = float(sat_image_list_df['model_scores'].iloc[i])
            except Exception:
                image_suitability_score = 0.0
            if image_suitability_score < 0.335:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_score += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip missing analysis image
            if image is None or (isinstance(image, float) and np.isnan(image)):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_missing += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            satname = sat_image_list_df['satnames'].iloc[i]
            date = sat_image_list_df['datetimes_utc'].iloc[i]
            # Standardize date for filename
            try:
                check_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d-%H-%M-%S")
            except Exception:
                # Fallback: use raw date string with unsafe chars replaced
                check_date = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")

            # If output exists, mark done
            if shoreline_path in zoo_shorelines_list or os.path.isfile(shoreline_path):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Load raster bands (segmented results saved in prior step)
            with rasterio.open(image) as src:
                # Bands layout:
                # non-PS: [blue,green,red,nir,swir,seg_lab,binary_nir,binary_swir]
                # PS:     [blue,green,red,nir,seg_lab,binary_nir]
                if satname != 'PS':
                    nir = src.read(4)
                    seg_lab = src.read(6)
                    binary_image_nir = src.read(7)
                    binary_image_swir = src.read(8)
                else:
                    nir = src.read(4)
                    seg_lab = src.read(5)
                    binary_image_nir = src.read(6)
                    binary_image_swir = None

                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                bounds = src.bounds
                crs_raster = src.crs

                # Pixel size from transform
                x_res = transform.a
                y_res = -transform.e
                xmin = bounds.left
                ymax = bounds.top

                # Valid-data mask (base)
                if mask_value is not None:
                    data_mask = nir != mask_value
                else:
                    data_mask = np.ones_like(nir, dtype=bool)

                # ---- L7-specific mask hardening to avoid scanline stripes (no clipping) ----
                # Apply only when satname is 'L7'; leaves other sensors unchanged.
                if str(satname).upper() == 'L7':
                    try:
                        # Exclude zero-valued NIR pixels (common in SLC-off gaps/seams)
                        data_mask = data_mask & (nir != 0)

                        # If NIR binary is present, exclude zeros there too (reinforces gaps)
                        if binary_image_nir is not None:
                            data_mask = data_mask & (binary_image_nir != 0)

                        # Exclude non-finite segmentation values (NaN/Inf) to avoid artifacts
                        if seg_lab is not None:
                            data_mask = data_mask & np.isfinite(seg_lab)

                        # If SWIR binary present (non-PS), exclude zeros there too
                        if binary_image_swir is not None:
                            data_mask = data_mask & (binary_image_swir != 0)
                    except Exception:
                        logger.exception("L7 mask hardening failed; proceeding with original mask.")

                # Build no-data mask for NaN assignment (unchanged)
                if mask_value is not None:
                    no_data_mask = nir == mask_value
                else:
                    no_data_mask = np.zeros_like(nir, dtype=bool)

                # Build polygons for valid/no-data (original approach)
                data_polygon = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for _, (s, v) in enumerate(shapes(nir, mask=(nir != mask_value) if mask_value is not None else None, transform=src.transform))
                )
                data_polygon = gpd.GeoDataFrame.from_features(list(data_polygon), crs=src.crs)

                no_data_polygon = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for _, (s, v) in enumerate(shapes(nir, mask=(nir == mask_value) if mask_value is not None else None, transform=src.transform))
                )
                try:
                    no_data_polygon = gpd.GeoDataFrame.from_features(list(no_data_polygon), crs=src.crs)
                except Exception:
                    no_data_polygon = None

            # Buffer/difference polygons (unchanged semantics)
            try:
                if no_data_polygon is not None and len(no_data_polygon) > 0:
                    no_data_union = no_data_polygon.buffer(x_res * 2).unary_union
                else:
                    no_data_union = None

                data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                if data_union is not None and no_data_union is not None:
                    data_polygon_final = data_union.difference(no_data_union)
                else:
                    data_polygon_final = data_union
            except Exception:
                logger.exception("Polygon buffer/difference failed; proceeding with data_union as-is.")
                data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None

            # Ensure segmentation has nodata cleared (unchanged)
            seg_lab = seg_lab.astype(np.float32)
            seg_lab[no_data_mask] = np.nan

            # Extract contours for each source (unchanged calls; no extra clipping here)
            try:
                linestrings = get_contours(
                    seg_lab, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for seg_lab → %s", image)
                linestrings = []

            try:
                linestrings_nir = get_contours(
                    binary_image_nir, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for binary NIR → %s", image)
                linestrings_nir = []

            if satname != 'PS' and binary_image_swir is not None:
                try:
                    linestrings_swir = get_contours(
                        binary_image_swir, satname, xmin, ymax, x_res, y_res,
                        data_polygon_final, reference_shoreline_gdf,
                        ref_shore_buffer, reference_polygon,
                        data_mask, crs
                    )
                except Exception:
                    logger.exception("get_contours failed for binary SWIR → %s", image)
                    linestrings_swir = []
            else:
                linestrings_swir = []

            # Scores
            image_score = sat_image_list_df['model_scores'].iloc[i]
            seg_score = sat_image_list_df['seg_scores'].iloc[i] if 'seg_scores' in sat_image_list_df.columns else None

            # Filter by minimum coordinate count (unchanged behavior)
            waterlines = []
            shoreline_seg_scores = []
            shoreline_image_scores = []

            for ls in linestrings:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines.append(ls)
                    shoreline_seg_scores.append(seg_score)
                    shoreline_image_scores.append(image_score)

            waterlines_nir = []
            shoreline_seg_scores_nir = []
            shoreline_image_scores_nir = []

            for ls in linestrings_nir:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines_nir.append(ls)
                    shoreline_seg_scores_nir.append(seg_score)
                    shoreline_image_scores_nir.append(image_score)

            waterlines_swir = []
            shoreline_seg_scores_swir = []
            shoreline_image_scores_swir = []
            if satname != 'PS':
                for ls in linestrings_swir:
                    coords = LineString_to_arr(ls)
                    if len(coords) > 5:
                        waterlines_swir.append(ls)
                        shoreline_seg_scores_swir.append(seg_score)
                        shoreline_image_scores_swir.append(image_score)

            # If none extracted, mark done and continue
            if len(waterlines) == 0 or len(waterlines_nir) == 0:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                logger.info("No shorelines extracted for %s", image)
                continue

            # Save RGB-based shorelines (zoo)
            try:
                shorelines_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines),
                        'image_suitability_score': shoreline_image_scores,
                        'segmentation_suitability_score': shoreline_seg_scores,
                        'satname': [satname] * len(waterlines)
                    },
                    geometry=waterlines, crs=crs
                )
                shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_gdf = utm_to_wgs84_df(shorelines_gdf).reset_index(drop=True)
                shorelines_gdf = split_line(shorelines_gdf, 'LineString', smooth=True)
                shorelines_gdf.to_file(shoreline_path)
                logger.info("Saved RGB shorelines → %s", shoreline_path)
            except Exception:
                logger.exception("Failed saving RGB shorelines → %s", image)

            # Save NIR threshold shorelines
            try:
                nir_shoreline_path = os.path.join(nir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_nir_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines_nir),
                        'image_suitability_score': shoreline_image_scores_nir,
                        'segmentation_suitability_score': shoreline_seg_scores_nir,
                        'satname': [satname] * len(waterlines_nir),
                    },
                    geometry=waterlines_nir, crs=crs
                ).reset_index(drop=True)
                shorelines_nir_gdf = utm_to_wgs84_df(shorelines_nir_gdf)
                shorelines_nir_gdf = split_line(shorelines_nir_gdf, 'LineString', smooth=True)
                shorelines_nir_gdf.to_file(nir_shoreline_path)
                logger.info("Saved NIR shorelines → %s", nir_shoreline_path)
            except Exception:
                logger.exception("Failed saving NIR shorelines → %s", image)

            # Save SWIR threshold shorelines (non-PS only)
            if satname != 'PS' and len(waterlines_swir) > 0:
                try:
                    swir_shoreline_path = os.path.join(swir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                    shorelines_swir_gdf = gpd.GeoDataFrame(
                        {
                            'dates': [check_date] * len(waterlines_swir),
                            'image_suitability_score': shoreline_image_scores_swir,
                            'segmentation_suitability_score': shoreline_seg_scores_swir,
                            'satname': [satname] * len(waterlines_swir),
                        },
                        geometry=waterlines_swir, crs=crs
                    ).reset_index(drop=True)
                    shorelines_swir_gdf = utm_to_wgs84_df(shorelines_swir_gdf)
                    shorelines_swir_gdf = split_line(shorelines_swir_gdf, 'LineString', smooth=True)
                    shorelines_swir_gdf.to_file(swir_shoreline_path)
                    logger.info("Saved SWIR shorelines → %s", swir_shoreline_path)
                except Exception:
                    logger.exception("Failed saving SWIR shorelines → %s", image)

            # Mark done and persist
            sat_image_list_df.at[i, 'shoreline_done'] = True
            processed += 1
            try:
                tmp_path = sat_image_list_df_path_shore + ".tmp"
                sat_image_list_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, sat_image_list_df_path_shore)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception at index=%d", i)

        # GC
        try:
            gc.collect()
        except Exception:
            logger.exception("gc.collect() failed")

    # Final write and summary
    try:
        sat_image_list_df.to_csv(sat_image_list_df_path_shore, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path_shore)

    logger.info(
        "Summary: rows=%d, processed=%d, already_done=%d, skipped_low_score=%d, "
        "skipped_missing_image=%d, errors=%d, CSV → %s",
        num_images, processed, already_done, skipped_score, skipped_missing, errors, sat_image_list_df_path_shore
    )


def extract_shorelines_after_segmentation_section_old5(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in GeoTIFFs.

    Inputs:
      g (str): global region
      c (str): coastal area
      rr (str): subregion
      sss (str): shoreline section
      r_home (str): path/to/g#/c#/rr##/

    Logging:
      <cwd>/logs/shoreline_extraction/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_shoreline_extraction_logger(section_string)

    # Reference shoreline and polygon (reproject to UTM)
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    except Exception:
        logger.exception("Failed to read reference shoreline → %s", reference_shoreline)
        return
    try:
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    except Exception:
        logger.exception("Failed to reproject reference shoreline to UTM.")
        return
    crs = reference_shoreline_gdf.crs

    # Select CSV paths
    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path):
            sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
        else:
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    # Load segmentation CSV
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    except Exception:
        logger.exception("Failed to read segmentation CSV → %s", sat_image_list_df_path)
        return

    # Output directories
    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
    for d in (shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
        os.makedirs(d, exist_ok=True)

    num_images = len(sat_image_list_df)
    if num_images == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    # Reset handling
    if reset is True:
        sat_image_list_df['shoreline_done'] = [None] * num_images
        for d in (zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
            try:
                shutil.rmtree(d)
            except Exception:
                # directory may not exist; ignore
                pass
        logger.info("Reset: cleared shoreline_done and removed output directories.")
    else:
        if 'shoreline_done' not in sat_image_list_df.columns:
            # fallback to 'done' if present; else initialize
            if 'done' in sat_image_list_df.columns:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['done']
            else:
                sat_image_list_df['shoreline_done'] = [None] * num_images

    # Cache of existing zoo geojsons to skip re-write
    zoo_shorelines_list = set(glob.glob(os.path.join(zoo_shoreline_dir, '*.geojson')))

    processed = 0
    skipped_score = 0
    skipped_missing = 0
    already_done = 0
    errors = 0

    for i in tqdm(range(len(sat_image_list_df['analysis_image'])),
                  desc=f"Extracting {section_string}",
                  unit="img",
                  total=len(sat_image_list_df['analysis_image'])):
        try:
            image = sat_image_list_df['analysis_image'].iloc[i]
            roi_folder = sat_image_list_df['roi_folder'].iloc[i]

            # Already marked done
            if sat_image_list_df['shoreline_done'].iloc[i] is True:
                already_done += 1
                # progressive write
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip low suitability
            try:
                image_suitability_score = float(sat_image_list_df['model_scores'].iloc[i])
            except Exception:
                image_suitability_score = 0.0
            if image_suitability_score < 0.335:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_score += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip missing analysis image
            if image is None or (isinstance(image, float) and np.isnan(image)):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_missing += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            satname = sat_image_list_df['satnames'].iloc[i]
            date = sat_image_list_df['datetimes_utc'].iloc[i]
            # Standardize date for filename
            try:
                check_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d-%H-%M-%S")
            except Exception:
                # Fallback: use raw date string with unsafe chars replaced
                check_date = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")

            # If output exists, mark done
            if shoreline_path in zoo_shorelines_list or os.path.isfile(shoreline_path):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Load raster bands (segmented results saved in prior step)
            with rasterio.open(image) as src:
                # Bands layout:
                # non-PS: [blue,green,red,nir,swir,seg_lab,binary_nir,binary_swir]
                # PS:     [blue,green,red,nir,seg_lab,binary_nir]
                if satname != 'PS':
                    nir = src.read(4)
                    seg_lab = src.read(6)
                    binary_image_nir = src.read(7)
                    binary_image_swir = src.read(8)
                else:
                    nir = src.read(4)
                    seg_lab = src.read(5)
                    binary_image_nir = src.read(6)
                    binary_image_swir = None

                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                bounds = src.bounds
                crs_raster = src.crs

                # Pixel size from transform
                x_res = transform.a
                y_res = -transform.e
                xmin = bounds.left
                ymax = bounds.top

                # Base valid-data mask (bool contiguous)
                if mask_value is not None:
                    data_mask = (nir != mask_value).astype(bool, copy=False)
                else:
                    data_mask = np.ones(nir.shape, dtype=bool)

                # ---- L7-only mask hardening (fast, in-place; no clipping here) ----
                if str(satname).upper() == 'L7':
                    try:
                        nir_arr = np.asarray(nir)  # plain ndarray view

                        # Only apply if zeros actually exist; in-place to avoid temporaries
                        if (nir_arr == 0).any():
                            np.logical_and(data_mask, nir_arr != 0, out=data_mask)

                        # Reinforce with binary NIR if it has zeros
                        if binary_image_nir is not None:
                            bin_nir = np.asarray(binary_image_nir)
                            if (bin_nir == 0).any():
                                np.logical_and(data_mask, bin_nir != 0, out=data_mask)

                        # Apply isfinite only if NaNs present (you set NaNs via no_data_mask)
                        if np.isnan(seg_lab).any():
                            seg_arr = np.asarray(seg_lab)
                            np.logical_and(data_mask, np.isfinite(seg_arr), out=data_mask)

                        # Reinforce with binary SWIR if present and has zeros
                        if binary_image_swir is not None:
                            bin_swir = np.asarray(binary_image_swir)
                            if (bin_swir == 0).any():
                                np.logical_and(data_mask, bin_swir != 0, out=data_mask)
                    except Exception:
                        logger.exception("L7 mask hardening failed; proceeding with original mask.")

                # Build no-data mask for NaN assignment (unchanged)
                if mask_value is not None:
                    no_data_mask = (nir == mask_value)
                else:
                    no_data_mask = np.zeros_like(nir, dtype=bool)

                # Build polygons for valid/no-data (original approach, unchanged)
                data_polygon = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for _, (s, v) in enumerate(shapes(nir, mask=(nir != mask_value) if mask_value is not None else None, transform=src.transform))
                )
                data_polygon = gpd.GeoDataFrame.from_features(list(data_polygon), crs=src.crs)

                no_data_polygon = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for _, (s, v) in enumerate(shapes(nir, mask=(nir == mask_value) if mask_value is not None else None, transform=src.transform))
                )
                try:
                    no_data_polygon = gpd.GeoDataFrame.from_features(list(no_data_polygon), crs=src.crs)
                except Exception:
                    no_data_polygon = None

            # Buffer/difference polygons (unchanged semantics)
            try:
                if no_data_polygon is not None and len(no_data_polygon) > 0:
                    no_data_union = no_data_polygon.buffer(x_res * 2).unary_union
                else:
                    no_data_union = None

                data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                if data_union is not None and no_data_union is not None:
                    data_polygon_final = data_union.difference(no_data_union)
                else:
                    data_polygon_final = data_union
            except Exception:
                logger.exception("Polygon buffer/difference failed; proceeding with data_union as-is.")
                data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None

            # Ensure segmentation has nodata cleared (unchanged)
            seg_lab = seg_lab.astype(np.float32)
            seg_lab[no_data_mask] = np.nan

            # Extract contours for each source (unchanged calls; no clipping in get_contours)
            try:
                linestrings = get_contours(
                    seg_lab, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for seg_lab → %s", image)
                linestrings = []

            try:
                linestrings_nir = get_contours(
                    binary_image_nir, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for binary NIR → %s", image)
                linestrings_nir = []

            if satname != 'PS' and binary_image_swir is not None:
                try:
                    linestrings_swir = get_contours(
                        binary_image_swir, satname, xmin, ymax, x_res, y_res,
                        data_polygon_final, reference_shoreline_gdf,
                        ref_shore_buffer, reference_polygon,
                        data_mask, crs
                    )
                except Exception:
                    logger.exception("get_contours failed for binary SWIR → %s", image)
                    linestrings_swir = []
            else:
                linestrings_swir = []

            # Scores
            image_score = sat_image_list_df['model_scores'].iloc[i]
            seg_score = sat_image_list_df['seg_scores'].iloc[i] if 'seg_scores' in sat_image_list_df.columns else None

            # Filter by minimum coordinate count (unchanged behavior)
            waterlines = []
            shoreline_seg_scores = []
            shoreline_image_scores = []

            for ls in linestrings:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines.append(ls)
                    shoreline_seg_scores.append(seg_score)
                    shoreline_image_scores.append(image_score)

            waterlines_nir = []
            shoreline_seg_scores_nir = []
            shoreline_image_scores_nir = []

            for ls in linestrings_nir:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines_nir.append(ls)
                    shoreline_seg_scores_nir.append(seg_score)
                    shoreline_image_scores_nir.append(image_score)

            waterlines_swir = []
            shoreline_seg_scores_swir = []
            shoreline_image_scores_swir = []
            if satname != 'PS':
                for ls in linestrings_swir:
                    coords = LineString_to_arr(ls)
                    if len(coords) > 5:
                        waterlines_swir.append(ls)
                        shoreline_seg_scores_swir.append(seg_score)
                        shoreline_image_scores_swir.append(image_score)

            # If none extracted, mark done and continue
            if len(waterlines) == 0 or len(waterlines_nir) == 0:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                logger.info("No shorelines extracted for %s", image)
                continue

            # Save RGB-based shorelines (zoo)
            try:
                shorelines_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines),
                        'image_suitability_score': shoreline_image_scores,
                        'segmentation_suitability_score': shoreline_seg_scores,
                        'satname': [satname] * len(waterlines)
                    },
                    geometry=waterlines, crs=crs
                )
                shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_gdf = utm_to_wgs84_df(shorelines_gdf).reset_index(drop=True)
                shorelines_gdf = split_line(shorelines_gdf, 'LineString', smooth=True)
                shorelines_gdf.to_file(shoreline_path)
                logger.info("Saved RGB shorelines → %s", shoreline_path)
            except Exception:
                logger.exception("Failed saving RGB shorelines → %s", image)

            # Save NIR threshold shorelines
            try:
                nir_shoreline_path = os.path.join(nir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_nir_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines_nir),
                        'image_suitability_score': shoreline_image_scores_nir,
                        'segmentation_suitability_score': shoreline_seg_scores_nir,
                        'satname': [satname] * len(waterlines_nir),
                    },
                    geometry=waterlines_nir, crs=crs
                ).reset_index(drop=True)
                shorelines_nir_gdf = utm_to_wgs84_df(shorelines_nir_gdf)
                shorelines_nir_gdf = split_line(shorelines_nir_gdf, 'LineString', smooth=True)
                shorelines_nir_gdf.to_file(nir_shoreline_path)
                logger.info("Saved NIR shorelines → %s", nir_shoreline_path)
            except Exception:
                logger.exception("Failed saving NIR shorelines → %s", image)

            # Save SWIR threshold shorelines (non-PS only)
            if satname != 'PS' and len(waterlines_swir) > 0:
                try:
                    swir_shoreline_path = os.path.join(swir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                    shorelines_swir_gdf = gpd.GeoDataFrame(
                        {
                            'dates': [check_date] * len(waterlines_swir),
                            'image_suitability_score': shoreline_image_scores_swir,
                            'segmentation_suitability_score': shoreline_seg_scores_swir,
                            'satname': [satname] * len(waterlines_swir),
                        },
                        geometry=waterlines_swir, crs=crs
                    ).reset_index(drop=True)
                    shorelines_swir_gdf = utm_to_wgs84_df(shorelines_swir_gdf)
                    shorelines_swir_gdf = split_line(shorelines_swir_gdf, 'LineString', smooth=True)
                    shorelines_swir_gdf.to_file(swir_shoreline_path)
                    logger.info("Saved SWIR shorelines → %s", swir_shoreline_path)
                except Exception:
                    logger.exception("Failed saving SWIR shorelines → %s", image)

            # Mark done and persist
            sat_image_list_df.at[i, 'shoreline_done'] = True
            processed += 1
            try:
                tmp_path = sat_image_list_df_path_shore + ".tmp"
                sat_image_list_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, sat_image_list_df_path_shore)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception at index=%d", i)

        # GC (optional)
        try:
            gc.collect()
        except Exception:
            logger.exception("gc.collect() failed")

    # Final write and summary
    try:
        sat_image_list_df.to_csv(sat_image_list_df_path_shore, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path_shore)

    logger.info(
        "Summary: rows=%d, processed=%d, already_done=%d, skipped_low_score=%d, "
        "skipped_missing_image=%d, errors=%d, CSV → %s",
        num_images, processed, already_done, skipped_score, skipped_missing, errors, sat_image_list_df_path_shore
    )


def extract_shorelines_after_segmentation_section_old6(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in GeoTIFFs.

    Logging:
      <cwd>/logs/shoreline_extraction/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_shoreline_extraction_logger(section_string)

    # Reference shoreline and polygon (reproject to UTM)
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    except Exception:
        logger.exception("Failed to read reference shoreline → %s", reference_shoreline)
        return
    try:
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    except Exception:
        logger.exception("Failed to reproject reference shoreline to UTM.")
        return
    crs = reference_shoreline_gdf.crs

    # Select CSV paths
    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path):
            sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
        else:
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    # Load segmentation CSV
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    except Exception:
        logger.exception("Failed to read segmentation CSV → %s", sat_image_list_df_path)
        return

    # Output directories
    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
    for d in (shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
        os.makedirs(d, exist_ok=True)

    num_images = len(sat_image_list_df)
    if num_images == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    # Reset handling
    if reset is True:
        sat_image_list_df['shoreline_done'] = [None] * num_images
        for d in (zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        logger.info("Reset: cleared shoreline_done and removed output directories.")
    else:
        if 'shoreline_done' not in sat_image_list_df.columns:
            if 'done' in sat_image_list_df.columns:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['done']
            else:
                sat_image_list_df['shoreline_done'] = [None] * num_images

    # Cache of existing zoo geojsons to skip re-write
    zoo_shorelines_list = set(glob.glob(os.path.join(zoo_shoreline_dir, '*.geojson')))

    processed = 0
    skipped_score = 0
    skipped_missing = 0
    already_done = 0
    errors = 0

    for i in tqdm(range(len(sat_image_list_df['analysis_image'])),
                  desc=f"Extracting {section_string}",
                  unit="img",
                  total=len(sat_image_list_df['analysis_image'])):
        try:
            image = sat_image_list_df['analysis_image'].iloc[i]
            roi_folder = sat_image_list_df['roi_folder'].iloc[i]

            # Already marked done
            if sat_image_list_df['shoreline_done'].iloc[i] is True:
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip low suitability
            try:
                image_suitability_score = float(sat_image_list_df['model_scores'].iloc[i])
            except Exception:
                image_suitability_score = 0.0
            if image_suitability_score < 0.335:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_score += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip missing analysis image
            if image is None or (isinstance(image, float) and np.isnan(image)):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_missing += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            satname = sat_image_list_df['satnames'].iloc[i]
            date = sat_image_list_df['datetimes_utc'].iloc[i]
            # Standardize date for filename
            try:
                check_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d-%H-%M-%S")
            except Exception:
                check_date = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")

            # If output exists, mark done
            if shoreline_path in zoo_shorelines_list or os.path.isfile(shoreline_path):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Load raster bands (segmented results saved in prior step)
            with rasterio.open(image) as src:
                # Bands layout:
                # non-PS: [blue,green,red,nir,swir,seg_lab,binary_nir,binary_swir]
                # PS:     [blue,green,red,nir,seg_lab,binary_nir]
                if satname != 'PS':
                    nir = src.read(4)
                    seg_lab = src.read(6)
                    binary_image_nir = src.read(7)
                    binary_image_swir = src.read(8)
                else:
                    nir = src.read(4)
                    seg_lab = src.read(5)
                    binary_image_nir = src.read(6)
                    binary_image_swir = None

                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                bounds = src.bounds
                crs_raster = src.crs

                # Pixel size from transform
                x_res = transform.a
                y_res = -transform.e
                xmin = bounds.left
                ymax = bounds.top

                # Base valid-data mask (bool contiguous)
                if mask_value is not None:
                    data_mask = (nir != mask_value).astype(bool, copy=False)
                    no_data_mask = (nir == mask_value)
                else:
                    data_mask = np.ones(nir.shape, dtype=bool)
                    no_data_mask = np.zeros_like(nir, dtype=bool)

                # ---- L7-only mask hardening (fast, in-place; no extra clipping) ----
                if str(satname).upper() == 'L7':
                    try:
                        nir_arr = np.asarray(nir)
                        if (nir_arr == 0).any():
                            # In-place combine to avoid temporaries
                            np.logical_and(data_mask, nir_arr != 0, out=data_mask)

                        if binary_image_nir is not None:
                            bin_nir = np.asarray(binary_image_nir)
                            if (bin_nir == 0).any():
                                np.logical_and(data_mask, bin_nir != 0, out=data_mask)

                        # Only apply isfinite if NaNs exist
                        if np.isnan(seg_lab).any():
                            seg_arr = np.asarray(seg_lab)
                            np.logical_and(data_mask, np.isfinite(seg_arr), out=data_mask)

                        if binary_image_swir is not None:
                            bin_swir = np.asarray(binary_image_swir)
                            if (bin_swir == 0).any():
                                np.logical_and(data_mask, bin_swir != 0, out=data_mask)
                    except Exception:
                        logger.exception("L7 mask hardening failed; proceeding with original mask.")

                # ---- Polygonization: single pass over a binary valid/nodata label (faster; same final area) ----
                try:
                    valid_label = np.where(no_data_mask, 0, 1).astype(np.uint8)
                    polys = list(raster_shapes(valid_label, mask=None, transform=src.transform, connectivity=8))
                    valid_geoms  = [s for (s, v) in polys if v == 1]
                    nodata_geoms = [s for (s, v) in polys if v == 0]

                    data_polygon = gpd.GeoDataFrame(geometry=valid_geoms, crs=src.crs)
                    no_data_polygon = gpd.GeoDataFrame(geometry=nodata_geoms, crs=src.crs) if nodata_geoms else None
                except Exception:
                    # Fallback: original two-pass polygonization on NIR (keeps outputs identical)
                    logger.exception("Binary mask polygonization failed; falling back to two-pass polygonization.")
                    try:
                        # Valid regions
                        mask_valid = (nir != mask_value) if mask_value is not None else None
                        data_features = (
                            {'properties': {'raster_val': v}, 'geometry': s}
                            for _, (s, v) in enumerate(raster_shapes(nir, mask=mask_valid, transform=src.transform))
                        )
                        data_polygon = gpd.GeoDataFrame.from_features(list(data_features), crs=src.crs)

                        # No-data regions
                        mask_nodata = (nir == mask_value) if mask_value is not None else None
                        nd_features = (
                            {'properties': {'raster_val': v}, 'geometry': s}
                            for _, (s, v) in enumerate(raster_shapes(nir, mask=mask_nodata, transform=src.transform))
                        )
                        no_data_polygon = gpd.GeoDataFrame.from_features(list(nd_features), crs=src.crs) if mask_nodata is not None else None
                    except Exception:
                        logger.exception("Fallback polygonization failed completely; proceeding with empty polygons.")
                        data_polygon = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                        no_data_polygon = None

            # Buffer/difference polygons (unchanged semantics)
            try:
                if no_data_polygon is not None and len(no_data_polygon) > 0:
                    no_data_union = no_data_polygon.buffer(x_res * 2).unary_union
                else:
                    no_data_union = None

                data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                if data_union is not None and no_data_union is not None:
                    data_polygon_final = data_union.difference(no_data_union)
                else:
                    data_polygon_final = data_union
            except Exception:
                logger.exception("Polygon buffer/difference failed; proceeding with data_union as-is.")
                data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None

            # Ensure segmentation has nodata cleared (unchanged)
            seg_lab = seg_lab.astype(np.float32, copy=False)
            seg_lab[no_data_mask] = np.nan

            # Extract contours for each source (unchanged calls; no extra clipping here)
            try:
                linestrings = get_contours(
                    seg_lab, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for seg_lab → %s", image)
                linestrings = []

            try:
                linestrings_nir = get_contours(
                    binary_image_nir, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for binary NIR → %s", image)
                linestrings_nir = []

            if satname != 'PS' and binary_image_swir is not None:
                try:
                    linestrings_swir = get_contours(
                        binary_image_swir, satname, xmin, ymax, x_res, y_res,
                        data_polygon_final, reference_shoreline_gdf,
                        ref_shore_buffer, reference_polygon,
                        data_mask, crs
                    )
                except Exception:
                    logger.exception("get_contours failed for binary SWIR → %s", image)
                    linestrings_swir = []
            else:
                linestrings_swir = []

            # Scores
            image_score = sat_image_list_df['model_scores'].iloc[i]
            seg_score = sat_image_list_df['seg_scores'].iloc[i] if 'seg_scores' in sat_image_list_df.columns else None

            # Filter by minimum coordinate count (unchanged)
            waterlines = []
            shoreline_seg_scores = []
            shoreline_image_scores = []
            for ls in linestrings:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines.append(ls)
                    shoreline_seg_scores.append(seg_score)
                    shoreline_image_scores.append(image_score)

            waterlines_nir = []
            shoreline_seg_scores_nir = []
            shoreline_image_scores_nir = []
            for ls in linestrings_nir:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines_nir.append(ls)
                    shoreline_seg_scores_nir.append(seg_score)
                    shoreline_image_scores_nir.append(image_score)

            waterlines_swir = []
            shoreline_seg_scores_swir = []
            shoreline_image_scores_swir = []
            if satname != 'PS':
                for ls in linestrings_swir:
                    coords = LineString_to_arr(ls)
                    if len(coords) > 5:
                        waterlines_swir.append(ls)
                        shoreline_seg_scores_swir.append(seg_score)
                        shoreline_image_scores_swir.append(image_score)

            # If none extracted, mark done and continue
            if len(waterlines) == 0 or len(waterlines_nir) == 0:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                logger.info("No shorelines extracted for %s", image)
                continue

            # Save RGB-based shorelines (zoo)
            try:
                shorelines_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines),
                        'image_suitability_score': shoreline_image_scores,
                        'segmentation_suitability_score': shoreline_seg_scores,
                        'satname': [satname] * len(waterlines)
                    },
                    geometry=waterlines, crs=crs
                )
                shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_gdf = utm_to_wgs84_df(shorelines_gdf).reset_index(drop=True)
                shorelines_gdf = split_line(shorelines_gdf, 'LineString', smooth=True)
                shorelines_gdf.to_file(shoreline_path)
                logger.info("Saved RGB shorelines → %s", shoreline_path)
            except Exception:
                logger.exception("Failed saving RGB shorelines → %s", image)

            # Save NIR threshold shorelines
            try:
                nir_shoreline_path = os.path.join(nir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_nir_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines_nir),
                        'image_suitability_score': shoreline_image_scores_nir,
                        'segmentation_suitability_score': shoreline_seg_scores_nir,
                        'satname': [satname] * len(waterlines_nir),
                    },
                    geometry=waterlines_nir, crs=crs
                ).reset_index(drop=True)
                shorelines_nir_gdf = utm_to_wgs84_df(shorelines_nir_gdf)
                shorelines_nir_gdf = split_line(shorelines_nir_gdf, 'LineString', smooth=True)
                shorelines_nir_gdf.to_file(nir_shoreline_path)
                logger.info("Saved NIR shorelines → %s", nir_shoreline_path)
            except Exception:
                logger.exception("Failed saving NIR shorelines → %s", image)

            # Save SWIR threshold shorelines (non-PS only)
            if satname != 'PS' and len(waterlines_swir) > 0:
                try:
                    swir_shoreline_path = os.path.join(swir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                    shorelines_swir_gdf = gpd.GeoDataFrame(
                        {
                            'dates': [check_date] * len(waterlines_swir),
                            'image_suitability_score': shoreline_image_scores_swir,
                            'segmentation_suitability_score': shoreline_seg_scores_swir,
                            'satname': [satname] * len(waterlines_swir),
                        },
                        geometry=waterlines_swir, crs=crs
                    ).reset_index(drop=True)
                    shorelines_swir_gdf = utm_to_wgs84_df(shorelines_swir_gdf)
                    shorelines_swir_gdf = split_line(shorelines_swir_gdf, 'LineString', smooth=True)
                    shorelines_swir_gdf.to_file(swir_shoreline_path)
                    logger.info("Saved SWIR shorelines → %s", swir_shoreline_path)
                except Exception:
                    logger.exception("Failed saving SWIR shorelines → %s", image)

            # Mark done and persist
            sat_image_list_df.at[i, 'shoreline_done'] = True
            processed += 1
            try:
                tmp_path = sat_image_list_df_path_shore + ".tmp"
                sat_image_list_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, sat_image_list_df_path_shore)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception at index=%d", i)

        # GC
        try:
            gc.collect()
        except Exception:
            logger.exception("gc.collect() failed")

    # Final write and summary
    try:
        sat_image_list_df.to_csv(sat_image_list_df_path_shore, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path_shore)

    logger.info(
        "Summary: rows=%d, processed=%d, already_done=%d, skipped_low_score=%d, "
        "skipped_missing_image=%d, errors=%d, CSV → %s",
        num_images, processed, already_done, skipped_score, skipped_missing, errors, sat_image_list_df_path_shore
    )


def extract_shorelines_after_segmentation_section_old7(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in GeoTIFFs.

    Logging:
      <cwd>/logs/shoreline_extraction/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_shoreline_extraction_logger(section_string)

    # Reference shoreline and polygon (reproject to UTM)
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    except Exception:
        logger.exception("Failed to read reference shoreline → %s", reference_shoreline)
        return
    try:
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    except Exception:
        logger.exception("Failed to reproject reference shoreline to UTM.")
        return
    crs = reference_shoreline_gdf.crs

    # Select CSV paths
    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path):
            sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
        else:
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    # Load segmentation CSV
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    except Exception:
        logger.exception("Failed to read segmentation CSV → %s", sat_image_list_df_path)
        return

    # Output directories
    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
    for d in (shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
        os.makedirs(d, exist_ok=True)

    num_images = len(sat_image_list_df)
    if num_images == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    # Reset handling
    if reset is True:
        sat_image_list_df['shoreline_done'] = [None] * num_images
        for d in (zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        logger.info("Reset: cleared shoreline_done and removed output directories.")
    else:
        if 'shoreline_done' not in sat_image_list_df.columns:
            if 'done' in sat_image_list_df.columns:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['done']
            else:
                sat_image_list_df['shoreline_done'] = [None] * num_images

    # Cache existing zoo geojsons to skip re-write
    zoo_shorelines_list = set(glob.glob(os.path.join(zoo_shoreline_dir, '*.geojson')))

    processed = 0
    skipped_score = 0
    skipped_missing = 0
    already_done = 0
    errors = 0

    for i in tqdm(range(len(sat_image_list_df['analysis_image'])),
                  desc=f"Extracting {section_string}",
                  unit="img",
                  total=len(sat_image_list_df['analysis_image'])):
        try:
            image = sat_image_list_df['analysis_image'].iloc[i]
            roi_folder = sat_image_list_df['roi_folder'].iloc[i]

            # Already marked done
            if sat_image_list_df['shoreline_done'].iloc[i] is True:
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip low suitability
            try:
                image_suitability_score = float(sat_image_list_df['model_scores'].iloc[i])
            except Exception:
                image_suitability_score = 0.0
            if image_suitability_score < 0.335:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_score += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip missing analysis image
            if image is None or (isinstance(image, float) and np.isnan(image)):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_missing += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            satname = sat_image_list_df['satnames'].iloc[i]
            date = sat_image_list_df['datetimes_utc'].iloc[i]
            # Standardize date for filename
            try:
                check_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d-%H-%M-%S")
            except Exception:
                check_date = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")

            # If output exists, mark done
            if shoreline_path in zoo_shorelines_list or os.path.isfile(shoreline_path):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Load raster bands (segmented results saved in prior step)
            with rasterio.open(image) as src:
                # non-PS: [blue,green,red,nir,swir,seg_lab,binary_nir,binary_swir]
                # PS:     [blue,green,red,nir,seg_lab,binary_nir]
                if satname != 'PS':
                    nir = src.read(4)
                    seg_lab = src.read(6)
                    binary_image_nir = src.read(7)
                    binary_image_swir = src.read(8)
                else:
                    nir = src.read(4)
                    seg_lab = src.read(5)
                    binary_image_nir = src.read(6)
                    binary_image_swir = None

                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                bounds = src.bounds
                crs_raster = src.crs

                # Pixel size from transform
                x_res = transform.a
                y_res = -transform.e
                xmin = bounds.left
                ymax = bounds.top

                # Base valid-data mask (contiguous bool)
                if mask_value is not None:
                    data_mask = (nir != mask_value).astype(bool, copy=False)
                    no_data_mask = (nir == mask_value)
                else:
                    data_mask = np.ones(nir.shape, dtype=bool)
                    no_data_mask = np.zeros_like(nir, dtype=bool)

                # ---- L7-only mask hardening (fast, in-place; no clipping added) ----
                if str(satname).upper() == 'L7':
                    try:
                        nir_arr = np.asarray(nir)
                        if (nir_arr == 0).any():
                            # In-place combine to avoid temporaries for large frames
                            np.logical_and(data_mask, nir_arr != 0, out=data_mask)

                        if binary_image_nir is not None:
                            bin_nir = np.asarray(binary_image_nir)
                            if (bin_nir == 0).any():
                                np.logical_and(data_mask, bin_nir != 0, out=data_mask)

                        # Convert seg_lab to float and set NaNs on nodata
                        seg_lab = seg_lab.astype(np.float32, copy=False)
                        seg_lab[no_data_mask] = np.nan

                        # Apply isfinite only if NaNs exist (fast guard)
                        if np.isnan(seg_lab).any():
                            seg_arr = np.asarray(seg_lab)
                            np.logical_and(data_mask, np.isfinite(seg_arr), out=data_mask)

                        if binary_image_swir is not None:
                            bin_swir = np.asarray(binary_image_swir)
                            if (bin_swir == 0).any():
                                np.logical_and(data_mask, bin_swir != 0, out=data_mask)

                        # Safety: revert if mask collapses
                        if not data_mask.any():
                            logger.warning("L7: data_mask empty after hardening; reverting to base nodata mask.")
                            data_mask = (nir != mask_value).astype(bool, copy=False) if mask_value is not None else np.ones(nir.shape, dtype=bool)
                            seg_lab[no_data_mask] = np.nan
                    except Exception:
                        logger.exception("L7 mask hardening failed; proceeding with base mask.")
                        seg_lab = seg_lab.astype(np.float32, copy=False)
                        seg_lab[no_data_mask] = np.nan
                else:
                    # Non-L7: set NaNs on nodata (unchanged)
                    seg_lab = seg_lab.astype(np.float32, copy=False)
                    seg_lab[no_data_mask] = np.nan

                # ---- Polygonization: single pass over binary valid/nodata label (faster; same final area) ----
                try:
                    # Build a simple label image: 1 = valid, 0 = nodata
                    valid_label = np.where(no_data_mask, 0, 1).astype(np.uint8)

                    # Iterate once; convert GeoJSON-like mappings to Shapely
                    valid_geoms = []
                    nodata_geoms = []
                    for mapping, val in rio_shapes(valid_label, mask=None, transform=transform, connectivity=8):
                        if val == 1:
                            valid_geoms.append(shapely_shape(mapping))
                        else:
                            nodata_geoms.append(shapely_shape(mapping))

                    # Unions/difference are done directly in Shapely (faster, identical result)
                    data_union = unary_union(valid_geoms) if valid_geoms else None
                    if nodata_geoms:
                        no_data_union = unary_union(nodata_geoms).buffer(x_res * 2)
                    else:
                        no_data_union = None

                    if data_union is not None and no_data_union is not None:
                        data_polygon_final = data_union.difference(no_data_union)
                    else:
                        data_polygon_final = data_union
                except Exception:
                    # Fallback: original two-pass polygonization on NIR (keeps outputs identical)
                    logger.exception("Binary mask polygonization failed; falling back to two-pass polygonization.")
                    try:
                        mask_valid = (nir != mask_value) if mask_value is not None else None
                        data_features = (
                            {'properties': {'raster_val': v}, 'geometry': s}
                            for _, (s, v) in enumerate(rio_shapes(nir, mask=mask_valid, transform=transform))
                        )
                        data_polygon = gpd.GeoDataFrame.from_features(list(data_features), crs=crs_raster)

                        mask_nodata = (nir == mask_value) if mask_value is not None else None
                        nd_features = (
                            {'properties': {'raster_val': v}, 'geometry': s}
                            for _, (s, v) in enumerate(rio_shapes(nir, mask=mask_nodata, transform=transform))
                        )
                        no_data_polygon = gpd.GeoDataFrame.from_features(list(nd_features), crs=crs_raster) if mask_nodata is not None else None

                        # Match original semantics
                        try:
                            if no_data_polygon is not None and len(no_data_polygon) > 0:
                                no_data_union = no_data_polygon.buffer(x_res * 2).unary_union
                            else:
                                no_data_union = None
                            data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                            if data_union is not None and no_data_union is not None:
                                data_polygon_final = data_union.difference(no_data_union)
                            else:
                                data_polygon_final = data_union
                        except Exception:
                            logger.exception("Polygon buffer/difference failed; proceeding with data_union as-is.")
                            data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None
                    except Exception:
                        logger.exception("Fallback polygonization failed completely; proceeding with empty polygons.")
                        data_polygon_final = None  # safe: get_contours handles None internally / clips to None

            # Extract contours for each source (unchanged calls; no clipping added)
            try:
                linestrings = get_contours(
                    seg_lab, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for seg_lab → %s", image)
                linestrings = []

            try:
                linestrings_nir = get_contours(
                    binary_image_nir, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for binary NIR → %s", image)
                linestrings_nir = []

            if satname != 'PS' and binary_image_swir is not None:
                try:
                    linestrings_swir = get_contours(
                        binary_image_swir, satname, xmin, ymax, x_res, y_res,
                        data_polygon_final, reference_shoreline_gdf,
                        ref_shore_buffer, reference_polygon,
                        data_mask, crs
                    )
                except Exception:
                    logger.exception("get_contours failed for binary SWIR → %s", image)
                    linestrings_swir = []
            else:
                linestrings_swir = []

            # Scores
            image_score = sat_image_list_df['model_scores'].iloc[i]
            seg_score = sat_image_list_df['seg_scores'].iloc[i] if 'seg_scores' in sat_image_list_df.columns else None

            # Filter by minimum coordinate count (unchanged)
            waterlines = []
            shoreline_seg_scores = []
            shoreline_image_scores = []
            for ls in linestrings:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines.append(ls)
                    shoreline_seg_scores.append(seg_score)
                    shoreline_image_scores.append(image_score)

            waterlines_nir = []
            shoreline_seg_scores_nir = []
            shoreline_image_scores_nir = []
            for ls in linestrings_nir:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines_nir.append(ls)
                    shoreline_seg_scores_nir.append(seg_score)
                    shoreline_image_scores_nir.append(image_score)

            waterlines_swir = []
            shoreline_seg_scores_swir = []
            shoreline_image_scores_swir = []
            if satname != 'PS':
                for ls in linestrings_swir:
                    coords = LineString_to_arr(ls)
                    if len(coords) > 5:
                        waterlines_swir.append(ls)
                        shoreline_seg_scores_swir.append(seg_score)
                        shoreline_image_scores_swir.append(image_score)

            # If none extracted, mark done and continue
            if len(waterlines) == 0 or len(waterlines_nir) == 0:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                logger.info("No shorelines extracted for %s", image)
                continue

            # Save RGB-based shorelines (zoo)
            try:
                shorelines_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines),
                        'image_suitability_score': shoreline_image_scores,
                        'segmentation_suitability_score': shoreline_seg_scores,
                        'satname': [satname] * len(waterlines)
                    },
                    geometry=waterlines, crs=crs
                )
                shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_gdf = utm_to_wgs84_df(shorelines_gdf).reset_index(drop=True)
                shorelines_gdf = split_line(shorelines_gdf, 'LineString', smooth=True)
                shorelines_gdf.to_file(shoreline_path)
                logger.info("Saved RGB shorelines → %s", shoreline_path)
            except Exception:
                logger.exception("Failed saving RGB shorelines → %s", image)

            # Save NIR threshold shorelines
            try:
                nir_shoreline_path = os.path.join(nir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_nir_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines_nir),
                        'image_suitability_score': shoreline_image_scores_nir,
                        'segmentation_suitability_score': shoreline_seg_scores_nir,
                        'satname': [satname] * len(waterlines_nir),
                    },
                    geometry=waterlines_nir, crs=crs
                ).reset_index(drop=True)
                shorelines_nir_gdf = utm_to_wgs84_df(shorelines_nir_gdf)
                shorelines_nir_gdf = split_line(shorelines_nir_gdf, 'LineString', smooth=True)
                shorelines_nir_gdf.to_file(nir_shoreline_path)
                logger.info("Saved NIR shorelines → %s", nir_shoreline_path)
            except Exception:
                logger.exception("Failed saving NIR shorelines → %s", image)

            # Save SWIR threshold shorelines (non-PS only)
            if satname != 'PS' and len(waterlines_swir) > 0:
                try:
                    swir_shoreline_path = os.path.join(swir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                    shorelines_swir_gdf = gpd.GeoDataFrame(
                        {
                            'dates': [check_date] * len(waterlines_swir),
                            'image_suitability_score': shoreline_image_scores_swir,
                            'segmentation_suitability_score': shoreline_seg_scores_swir,
                            'satname': [satname] * len(waterlines_swir),
                        },
                        geometry=waterlines_swir, crs=crs
                    ).reset_index(drop=True)
                    shorelines_swir_gdf = utm_to_wgs84_df(shorelines_swir_gdf)
                    shorelines_swir_gdf = split_line(shorelines_swir_gdf, 'LineString', smooth=True)
                    shorelines_swir_gdf.to_file(swir_shoreline_path)
                    logger.info("Saved SWIR shorelines → %s", swir_shoreline_path)
                except Exception:
                    logger.exception("Failed saving SWIR shorelines → %s", image)

            # Mark done and persist
            sat_image_list_df.at[i, 'shoreline_done'] = True
            processed += 1
            try:
                tmp_path = sat_image_list_df_path_shore + ".tmp"
                sat_image_list_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, sat_image_list_df_path_shore)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception at index=%d", i)

        # GC (optional)
        try:
            gc.collect()
        except Exception:
            logger.exception("gc.collect() failed")

    # Final write and summary
    try:
        sat_image_list_df.to_csv(sat_image_list_df_path_shore, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path_shore)

    logger.info(
        "Summary: rows=%d, processed=%d, already_done=%d, skipped_low_score=%d, "
        "skipped_missing_image=%d, errors=%d, CSV → %s",
        num_images, processed, already_done, skipped_score, errors, sat_image_list_df_path_shore)


def extract_shorelines_after_segmentation_section_old8(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in GeoTIFFs.

    Logging:
      <cwd>/logs/shoreline_extraction/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_shoreline_extraction_logger(section_string)

    # Reference shoreline and polygon (reproject to UTM)
    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    except Exception:
        logger.exception("Failed to read reference shoreline → %s", reference_shoreline)
        return
    try:
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    except Exception:
        logger.exception("Failed to reproject reference shoreline to UTM.")
        return
    crs = reference_shoreline_gdf.crs  # UTM CRS used downstream

    # Select CSV paths
    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path):
            sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
        else:
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_shore = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    # Load segmentation CSV
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
    except Exception:
        logger.exception("Failed to read segmentation CSV → %s", sat_image_list_df_path)
        return

    # Output directories
    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')
    for d in (shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
        os.makedirs(d, exist_ok=True)

    num_images = len(sat_image_list_df)
    if num_images == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    # Reset handling
    if reset is True:
        sat_image_list_df['shoreline_done'] = [None] * num_images
        for d in (zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        logger.info("Reset: cleared shoreline_done and removed output directories.")
    else:
        if 'shoreline_done' not in sat_image_list_df.columns:
            if 'done' in sat_image_list_df.columns:
                sat_image_list_df['shoreline_done'] = sat_image_list_df['done']
            else:
                sat_image_list_df['shoreline_done'] = [None] * num_images

    # Cache existing zoo geojsons to skip re-write
    zoo_shorelines_list = set(glob.glob(os.path.join(zoo_shoreline_dir, '*.geojson')))

    processed = 0
    skipped_score = 0
    skipped_missing = 0
    already_done = 0
    errors = 0

    for i in tqdm(range(len(sat_image_list_df['analysis_image'])),
                  desc=f"Extracting {section_string}",
                  unit="img",
                  total=len(sat_image_list_df['analysis_image'])):
        try:
            image = sat_image_list_df['analysis_image'].iloc[i]
            roi_folder = sat_image_list_df['roi_folder'].iloc[i]

            # Already marked done
            if sat_image_list_df['shoreline_done'].iloc[i] is True:
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip low suitability
            try:
                image_suitability_score = float(sat_image_list_df['model_scores'].iloc[i])
            except Exception:
                image_suitability_score = 0.0
            if image_suitability_score < 0.335:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_score += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Skip missing analysis image
            if image is None or (isinstance(image, float) and np.isnan(image)):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                skipped_missing += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            satname = sat_image_list_df['satnames'].iloc[i]  # exact value, 'L7' for Landsat 7
            date = sat_image_list_df['datetimes_utc'].iloc[i]
            # Standardize date for filename
            try:
                check_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d-%H-%M-%S")
            except Exception:
                check_date = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")

            # If output exists, mark done
            if shoreline_path in zoo_shorelines_list or os.path.isfile(shoreline_path):
                sat_image_list_df.at[i, 'shoreline_done'] = True
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                continue

            # Load raster bands (segmented results saved in prior step)
            with rasterio.open(image) as src:
                # non-PS: [blue,green,red,nir,swir,seg_lab,binary_nir,binary_swir]
                # PS:     [blue,green,red,nir,seg_lab,binary_nir]
                if satname != 'PS':
                    nir = src.read(4)
                    seg_lab = src.read(6)
                    binary_image_nir = src.read(7)
                    binary_image_swir = src.read(8)
                else:
                    nir = src.read(4)
                    seg_lab = src.read(5)
                    binary_image_nir = src.read(6)
                    binary_image_swir = None

                mask_value = src.meta.get('nodata', None)
                transform = src.transform
                bounds = src.bounds
                crs_raster = src.crs

                # Pixel size from transform
                x_res = transform.a
                y_res = -transform.e
                xmin = bounds.left
                ymax = bounds.top

                # Base valid-data mask
                if mask_value is not None:
                    data_mask = (nir != mask_value).astype(bool, copy=False)
                    no_data_mask = (nir == mask_value)
                else:
                    data_mask = np.ones(nir.shape, dtype=bool)
                    no_data_mask = np.zeros_like(nir, dtype=bool)

                # ---- L7-only mask hardening (fast, in-place; no clipping added) ----
                if satname == 'L7':
                    try:
                        # Exclude obvious zero stripes (only if present)
                        nir_arr = np.asarray(nir)
                        if (nir_arr == 0).any():
                            np.logical_and(data_mask, nir_arr != 0, out=data_mask)

                        # Reinforce with binary bands (only if zeros present)
                        if binary_image_nir is not None:
                            bin_nir = np.asarray(binary_image_nir)
                            if (bin_nir == 0).any():
                                np.logical_and(data_mask, bin_nir != 0, out=data_mask)
                        if binary_image_swir is not None:
                            bin_swir = np.asarray(binary_image_swir)
                            if (bin_swir == 0).any():
                                np.logical_and(data_mask, bin_swir != 0, out=data_mask)

                        # seg_lab to float and set NaNs on nodata
                        seg_lab = seg_lab.astype(np.float32, copy=False)
                        seg_lab[no_data_mask] = np.nan

                        # Apply isfinite only if NaNs exist
                        if np.isnan(seg_lab).any():
                            seg_arr = np.asarray(seg_lab)
                            np.logical_and(data_mask, np.isfinite(seg_arr), out=data_mask)

                        # Safety: revert if mask collapses
                        if not data_mask.any():
                            logger.warning("L7: data_mask empty after hardening; reverting to base nodata mask.")
                            data_mask = (nir != mask_value).astype(bool, copy=False) if mask_value is not None else np.ones(nir.shape, dtype=bool)
                            seg_lab[no_data_mask] = np.nan
                    except Exception:
                        logger.exception("L7 mask hardening failed; proceeding with base mask.")
                        seg_lab = seg_lab.astype(np.float32, copy=False)
                        seg_lab[no_data_mask] = np.nan
                else:
                    # Non-L7: set NaNs on nodata (unchanged)
                    seg_lab = seg_lab.astype(np.float32, copy=False)
                    seg_lab[no_data_mask] = np.nan

                # ---- Polygonization: single pass over binary valid/nodata label (faster; same final area) ----
                try:
                    # Label image: 1 = valid, 0 = nodata
                    valid_label = np.where(no_data_mask, 0, 1).astype(np.uint8)

                    # Convert GeoJSON-like mappings to Shapely once
                    valid_geoms = []
                    nodata_geoms = []
                    for mapping, val in rio_shapes(valid_label, mask=None, transform=transform, connectivity=8):
                        if val == 1:
                            valid_geoms.append(shapely_shape(mapping))
                        else:
                            nodata_geoms.append(shapely_shape(mapping))

                    # Shapely unions/buffer/difference (faster; identical semantics)
                    data_union = unary_union(valid_geoms) if valid_geoms else None
                    no_data_union = unary_union(nodata_geoms).buffer(x_res * 2) if nodata_geoms else None

                    if data_union is not None and no_data_union is not None:
                        data_polygon_final = data_union.difference(no_data_union)
                    else:
                        data_polygon_final = data_union
                except Exception:
                    # Fallback: original two-pass polygonization on NIR (keeps outputs identical)
                    logger.exception("Binary mask polygonization failed; falling back to two-pass polygonization.")
                    try:
                        mask_valid = (nir != mask_value) if mask_value is not None else None
                        data_features = (
                            {'properties': {'raster_val': v}, 'geometry': s}
                            for _, (s, v) in enumerate(rio_shapes(nir, mask=mask_valid, transform=transform))
                        )
                        data_polygon = gpd.GeoDataFrame.from_features(list(data_features), crs=crs_raster)

                        mask_nodata = (nir == mask_value) if mask_value is not None else None
                        nd_features = (
                            {'properties': {'raster_val': v}, 'geometry': s}
                            for _, (s, v) in enumerate(rio_shapes(nir, mask=mask_nodata, transform=transform))
                        )
                        no_data_polygon = gpd.GeoDataFrame.from_features(list(nd_features), crs=crs_raster) if mask_nodata is not None else None

                        # Match original semantics
                        try:
                            if no_data_polygon is not None and len(no_data_polygon) > 0:
                                no_data_union = no_data_polygon.buffer(x_res * 2).unary_union
                            else:
                                no_data_union = None
                            data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                            if data_union is not None and no_data_union is not None:
                                data_polygon_final = data_union.difference(no_data_union)
                            else:
                                data_polygon_final = data_union
                        except Exception:
                            logger.exception("Polygon buffer/difference failed; proceeding with data_union as-is.")
                            data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None
                    except Exception:
                        logger.exception("Fallback polygonization failed completely; proceeding with empty polygons.")
                        data_polygon_final = None  # handled below

            # If polygonization yielded no valid area, mark done and continue
            if data_polygon_final is None:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                logger.info("No valid polygonized area for %s; skipping.", image)
                continue

            # Extract contours for each source (unchanged calls; no clipping inside get_contours)
            try:
                linestrings = get_contours(
                    seg_lab, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for seg_lab → %s", image)
                linestrings = []

            try:
                linestrings_nir = get_contours(
                    binary_image_nir, satname, xmin, ymax, x_res, y_res,
                    data_polygon_final, reference_shoreline_gdf,
                    ref_shore_buffer, reference_polygon,
                    data_mask, crs
                )
            except Exception:
                logger.exception("get_contours failed for binary NIR → %s", image)
                linestrings_nir = []

            if satname != 'PS' and binary_image_swir is not None:
                try:
                    linestrings_swir = get_contours(
                        binary_image_swir, satname, xmin, ymax, x_res, y_res,
                        data_polygon_final, reference_shoreline_gdf,
                        ref_shore_buffer, reference_polygon,
                        data_mask, crs
                    )
                except Exception:
                    logger.exception("get_contours failed for binary SWIR → %s", image)
                    linestrings_swir = []
            else:
                linestrings_swir = []

            # Scores
            image_score = sat_image_list_df['model_scores'].iloc[i]
            seg_score = sat_image_list_df['seg_scores'].iloc[i] if 'seg_scores' in sat_image_list_df.columns else None

            # Filter by minimum coordinate count (unchanged)
            waterlines = []
            shoreline_seg_scores = []
            shoreline_image_scores = []
            for ls in linestrings:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines.append(ls)
                    shoreline_seg_scores.append(seg_score)
                    shoreline_image_scores.append(image_score)

            waterlines_nir = []
            shoreline_seg_scores_nir = []
            shoreline_image_scores_nir = []
            for ls in linestrings_nir:
                coords = LineString_to_arr(ls)
                if len(coords) > 5:
                    waterlines_nir.append(ls)
                    shoreline_seg_scores_nir.append(seg_score)
                    shoreline_image_scores_nir.append(image_score)

            waterlines_swir = []
            shoreline_seg_scores_swir = []
            shoreline_image_scores_swir = []
            if satname != 'PS':
                for ls in linestrings_swir:
                    coords = LineString_to_arr(ls)
                    if len(coords) > 5:
                        waterlines_swir.append(ls)
                        shoreline_seg_scores_swir.append(seg_score)
                        shoreline_image_scores_swir.append(image_score)

            # If none extracted, mark done and continue
            if len(waterlines) == 0 or len(waterlines_nir) == 0:
                sat_image_list_df.at[i, 'shoreline_done'] = True
                try:
                    tmp_path = sat_image_list_df_path_shore + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_shore)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)
                logger.info("No shorelines extracted for %s", image)
                continue

            # Save RGB-based shorelines (zoo)
            try:
                shorelines_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines),
                        'image_suitability_score': shoreline_image_scores,
                        'segmentation_suitability_score': shoreline_seg_scores,
                        'satname': [satname] * len(waterlines)
                    },
                    geometry=waterlines, crs=crs
                )
                shoreline_path = os.path.join(zoo_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_gdf = utm_to_wgs84_df(shorelines_gdf).reset_index(drop=True)
                shorelines_gdf = split_line(shorelines_gdf, 'LineString', smooth=True)
                shorelines_gdf.to_file(shoreline_path)
                logger.info("Saved RGB shorelines → %s", shoreline_path)
            except Exception:
                logger.exception("Failed saving RGB shorelines → %s", image)

            # Save NIR threshold shorelines
            try:
                nir_shoreline_path = os.path.join(nir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                shorelines_nir_gdf = gpd.GeoDataFrame(
                    {
                        'dates': [check_date] * len(waterlines_nir),
                        'image_suitability_score': shoreline_image_scores_nir,
                        'segmentation_suitability_score': shoreline_seg_scores_nir,
                        'satname': [satname] * len(waterlines_nir),
                    },
                    geometry=waterlines_nir, crs=crs
                ).reset_index(drop=True)
                shorelines_nir_gdf = utm_to_wgs84_df(shorelines_nir_gdf)
                shorelines_nir_gdf = split_line(shorelines_nir_gdf, 'LineString', smooth=True)
                shorelines_nir_gdf.to_file(nir_shoreline_path)
                logger.info("Saved NIR shorelines → %s", nir_shoreline_path)
            except Exception:
                logger.exception("Failed saving NIR shorelines → %s", image)

            # Save SWIR threshold shorelines (non-PS only)
            if satname != 'PS' and len(waterlines_swir) > 0:
                try:
                    swir_shoreline_path = os.path.join(swir_shoreline_dir, f"{check_date}_{satname}_{roi_folder}.geojson")
                    shorelines_swir_gdf = gpd.GeoDataFrame(
                        {
                            'dates': [check_date] * len(waterlines_swir),
                            'image_suitability_score': shoreline_image_scores_swir,
                            'segmentation_suitability_score': shoreline_seg_scores_swir,
                            'satname': [satname] * len(waterlines_swir),
                        },
                        geometry=waterlines_swir, crs=crs
                    ).reset_index(drop=True)
                    shorelines_swir_gdf = utm_to_wgs84_df(shorelines_swir_gdf)
                    shorelines_swir_gdf = split_line(shorelines_swir_gdf, 'LineString', smooth=True)
                    shorelines_swir_gdf.to_file(swir_shoreline_path)
                    logger.info("Saved SWIR shorelines → %s", swir_shoreline_path)
                except Exception:
                    logger.exception("Failed saving SWIR shorelines → %s", image)

            # Mark done and persist
            sat_image_list_df.at[i, 'shoreline_done'] = True
            processed += 1
            try:
                tmp_path = sat_image_list_df_path_shore + ".tmp"
                sat_image_list_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, sat_image_list_df_path_shore)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_shore)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception at index=%d", i)

    # Final write and summary
    try:
        sat_image_list_df.to_csv(sat_image_list_df_path_shore, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path_shore)

    logger.info(
        "Summary: rows=%d, processed=%d, already_done=%d, skipped_low_score=%d, "
        "skipped_missing_image=%d, errors=%d, CSV → %s",
        num_images, processed, already_done, skipped_score, skipped_missing, errors, sat_image_list_df_path_shore)


def extract_shorelines_after_segmentation_section(g, c, rr, sss, r_home, reset=False, planet=False):
    """
    Extracts shorelines for single section from segmented bands in GeoTIFFs.
    writes log to wd/logs/shoreline_extraction/

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/G#/C#/RR##
    reset (bool): optional, resets lookup table
    planet (bool): optional, process planet imagery

    returns:
    nothing
    """
    import os, glob, shutil, datetime
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    from tqdm import tqdm
    from shapely.ops import unary_union
    from shapely.geometry import shape as shapely_shape
    from rasterio.features import shapes as rio_shapes

    # external helpers assumed available in your environment:
    # - get_shoreline_extraction_logger
    # - wgs84_to_utm_df
    # - split_line
    # - LineString_to_arr
    # - get_contours  (updated version provided below)

    # --------------------------------------------------------------------------------------
    # SETUP
    # --------------------------------------------------------------------------------------

    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_shoreline_extraction_logger(section_string)

    # Reference shoreline (→UTM)
    ref_shl_path = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')

    try:
        reference_shoreline_gdf = gpd.read_file(ref_shl_path)
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
        crs_ref = reference_shoreline_gdf.crs
    except Exception:
        logger.exception("Reference shoreline load/reproject failed → %s", ref_shl_path)
        return

    # CSV
    if planet:
        csv_in = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        csv_out = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented_shoreline_extracted.csv')
    else:
        csv_in = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        csv_out = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented_shoreline_extracted.csv')

    try:
        df = pd.read_csv(csv_in)
    except Exception:
        logger.exception("Failed to read segmentation CSV → %s", csv_in)
        return

    # Output dirs
    out_dir = os.path.join(section_dir, 'shorelines')
    zoo_dir  = os.path.join(out_dir, 'zoo_rgb')
    nir_dir  = os.path.join(out_dir, 'nir_thresh')
    swir_dir = os.path.join(out_dir, 'swir_thresh')
    for d in (out_dir, zoo_dir, nir_dir, swir_dir):
        os.makedirs(d, exist_ok=True)

    nrows = len(df)
    if nrows == 0:
        logger.info("Empty CSV → %s", csv_in)
        return

    # Reset
    if reset:
        df['shoreline_done'] = [None] * nrows
        for d in (zoo_dir, nir_dir, swir_dir):
            try:
                shutil.rmtree(d)
            except:
                pass
        logger.info("Reset: cleared shoreline_done and deleted outputs.")
    else:
        if 'shoreline_done' not in df.columns:
            df['shoreline_done'] = df.get('done', [None] * nrows)

    # Files already written
    existing = set(glob.glob(os.path.join(zoo_dir, '*.geojson')))

    processed = skipped_score = skipped_missing = already_done = errors = 0

    # --------------------------------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------------------------------

    for i in tqdm(range(nrows), desc=f"Extracting {section_string}", unit="img"):

        try:
            image = df['analysis_image'].iloc[i]
            roi   = df['roi_folder'].iloc[i]

            # Already done?
            if df['shoreline_done'].iloc[i] is True:
                already_done += 1
                # progressive write
                try:
                    df.to_csv(csv_out + ".tmp", index=False)
                    os.replace(csv_out + ".tmp", csv_out)
                except:
                    logger.exception("CSV write failed → %s", csv_out)
                continue

            # Low suitability
            try:
                score = float(df['model_scores'].iloc[i])
            except:
                score = 0.0
            if score < 0.335:
                df.at[i, 'shoreline_done'] = True
                skipped_score += 1
                try:
                    df.to_csv(csv_out + ".tmp", index=False)
                    os.replace(csv_out + ".tmp", csv_out)
                except:
                    logger.exception("CSV write failed → %s", csv_out)
                continue

            # Missing image?
            if image is None or (isinstance(image, float) and np.isnan(image)):
                df.at[i, 'shoreline_done'] = True
                skipped_missing += 1
                try:
                    df.to_csv(csv_out + ".tmp", index=False)
                    os.replace(csv_out + ".tmp", csv_out)
                except:
                    logger.exception("CSV write failed → %s", csv_out)
                continue

            satname = df['satnames'].iloc[i]
            date    = df['datetimes_utc'].iloc[i]

            # Filename-safe date
            try:
                fn_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00") \
                        .strftime("%Y-%m-%d-%H-%M-%S")
            except:
                fn_date = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            out_file = os.path.join(zoo_dir, f"{fn_date}_{satname}_{roi}.geojson")

            # Already exists?
            if out_file in existing or os.path.isfile(out_file):
                df.at[i, 'shoreline_done'] = True
                already_done += 1
                try:
                    df.to_csv(csv_out + ".tmp", index=False)
                    os.replace(csv_out + ".tmp", csv_out)
                except:
                    logger.exception("CSV write failed → %s", csv_out)
                continue

            # ----------------------------------------------------------------------------------
            # LOAD RASTER
            # ----------------------------------------------------------------------------------
            with rasterio.open(image) as src:
                if satname != 'PS':
                    nir      = src.read(4)
                    seg_lab  = src.read(6)
                    bin_nir  = src.read(7)
                    bin_swir = src.read(8)
                else:
                    nir      = src.read(4)
                    seg_lab  = src.read(5)
                    bin_nir  = src.read(6)
                    bin_swir = None

                mask_value = src.meta.get('nodata', None)
                transform  = src.transform
                bounds     = src.bounds
                crs_raster = src.crs

                x_res = transform.a
                y_res = -transform.e  # positive pixel height for north-up rasters
                xmin  = bounds.left
                ymax  = bounds.top

            # ----------------------------------------------------------------------------------
            # BUILD BASE MASKS
            # ----------------------------------------------------------------------------------

            # Valid-data mask from NIR nodata
            if mask_value is not None:
                data_mask    = (nir != mask_value).astype(bool, copy=False)
                no_data_mask = (nir == mask_value)
            else:
                data_mask    = np.ones(nir.shape, dtype=bool)
                no_data_mask = np.zeros_like(nir, dtype=bool)

            # seg_lab → float + NaN on nodata
            seg_lab = seg_lab.astype(np.float32, copy=False)
            seg_lab[no_data_mask] = np.nan

            # Require segmentation to be finite (stops L7 stripe following)
            np.logical_and(data_mask, np.isfinite(seg_lab), out=data_mask)

            # ----------------------------------------------------------------------------------
            # POLYGONIZATION (fast shapely version)
            # ----------------------------------------------------------------------------------

            try:
                # Simple label: 1 = valid, 0 = nodata
                valid_label = np.where(no_data_mask, 0, 1).astype(np.uint8)

                valid_geoms  = []
                nodata_geoms = []
                for mapping, val in rio_shapes(valid_label, mask=None, transform=transform, connectivity=8):
                    if val == 1:
                        valid_geoms.append(shapely_shape(mapping))
                    else:
                        nodata_geoms.append(shapely_shape(mapping))

                data_union = unary_union(valid_geoms) if valid_geoms else None
                nd_union   = unary_union(nodata_geoms).buffer(x_res * 2) if nodata_geoms else None

                if data_union is not None and nd_union is not None:
                    data_polygon_final = data_union.difference(nd_union)
                else:
                    data_polygon_final = data_union

            except Exception:
                logger.exception("Binary polygonization failed; trying fallback.")

                try:
                    mask_valid = (nir != mask_value) if mask_value is not None else None
                    df_valid = (
                        {'properties': {'val': v}, 'geometry': s}
                        for _, (s, v) in enumerate(rio_shapes(nir, mask=mask_valid, transform=transform))
                    )
                    data_polygon = gpd.GeoDataFrame.from_features(list(df_valid), crs=crs_raster)

                    mask_nd = (nir == mask_value) if mask_value is not None else None
                    df_nd = (
                        {'properties': {'val': v}, 'geometry': s}
                        for _, (s, v) in enumerate(rio_shapes(nir, mask=mask_nd, transform=transform))
                    )
                    no_data_polygon = gpd.GeoDataFrame.from_features(list(df_nd), crs=crs_raster) if mask_nd is not None else None

                    try:
                        nd_union = no_data_polygon.buffer(x_res * 2).unary_union if no_data_polygon is not None and len(no_data_polygon) > 0 else None
                        data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                        if data_union is not None and nd_union is not None:
                            data_polygon_final = data_union.difference(nd_union)
                        else:
                            data_polygon_final = data_union
                    except:
                        data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None

                except Exception:
                    logger.exception("Fallback polygonization failed completely.")
                    data_polygon_final = None

            if data_polygon_final is None:
                df.at[i, 'shoreline_done'] = True
                try:
                    df.to_csv(csv_out + ".tmp", index=False)
                    os.replace(csv_out + ".tmp", csv_out)
                except:
                    logger.exception("CSV write failed → %s", csv_out)
                logger.info("No valid polygonized area for %s", image)
                continue

            # ----------------------------------------------------------------------------------
            # CONTOUR EXTRACTION (affine-aware)
            # ----------------------------------------------------------------------------------

            try:
                lines = get_contours(seg_lab, satname, xmin, ymax, x_res, y_res,
                                     data_polygon_final, reference_shoreline_gdf,
                                     ref_shore_buffer, None, data_mask, crs_ref,
                                     transform=transform)
            except Exception:
                logger.exception("get_contours failed for seg_lab → %s", image)
                lines = []

            try:
                lines_nir = get_contours(bin_nir, satname, xmin, ymax, x_res, y_res,
                                         data_polygon_final, reference_shoreline_gdf,
                                         ref_shore_buffer, None, data_mask, crs_ref,
                                         transform=transform)
            except Exception:
                logger.exception("get_contours failed for binary NIR → %s", image)
                lines_nir = []

            if satname != 'PS' and bin_swir is not None:
                try:
                    lines_swir = get_contours(bin_swir, satname, xmin, ymax, x_res, y_res,
                                              data_polygon_final, reference_shoreline_gdf,
                                              ref_shore_buffer, None, data_mask, crs_ref,
                                              transform=transform)
                except Exception:
                    logger.exception("get_contours failed for binary SWIR → %s", image)
                    lines_swir = []
            else:
                lines_swir = []

            # Scores
            img_score  = df['model_scores'].iloc[i]
            seg_score  = df['seg_scores'].iloc[i] if 'seg_scores' in df.columns else None

            # Filter min coordinate count (>5)
            waterlines, shore_seg, shore_img = [], [], []
            for ls in lines:
                if len(LineString_to_arr(ls)) > 5:
                    waterlines.append(ls)
                    shore_seg.append(seg_score)
                    shore_img.append(img_score)

            waterlines_nir, shore_seg_nir, shore_img_nir = [], [], []
            for ls in lines_nir:
                if len(LineString_to_arr(ls)) > 5:
                    waterlines_nir.append(ls)
                    shore_seg_nir.append(seg_score)
                    shore_img_nir.append(img_score)

            waterlines_swir, shore_seg_swir, shore_img_swir = [], [], []
            if satname != 'PS':
                for ls in lines_swir:
                    if len(LineString_to_arr(ls)) > 5:
                        waterlines_swir.append(ls)
                        shore_seg_swir.append(seg_score)
                        shore_img_swir.append(img_score)

            # If none, mark done
            if len(waterlines) == 0 or len(waterlines_nir) == 0:
                df.at[i, 'shoreline_done'] = True
                try:
                    df.to_csv(csv_out + ".tmp", index=False)
                    os.replace(csv_out + ".tmp", csv_out)
                except:
                    logger.exception("CSV write failed → %s", csv_out)
                logger.info("No shorelines extracted for %s", image)
                continue

            # ----------------------------------------------------------------------------------
            # SAVE OUTPUTS (CRS-safe: tag with raster CRS, then to WGS84)
            # ----------------------------------------------------------------------------------

            # Zoo RGB
            try:
                gdf = gpd.GeoDataFrame(
                    {
                        'dates': [fn_date] * len(waterlines),
                        'image_suitability_score': shore_img,
                        'segmentation_suitability_score': shore_seg,
                        'satname': [satname] * len(waterlines)
                    },
                    geometry=waterlines,
                    crs=crs_raster   # <-- use raster CRS
                )
                out_rgb = os.path.join(zoo_dir, f"{fn_date}_{satname}_{roi}.geojson")
                gdf = gdf.to_crs("EPSG:4326")  # generic reprojection
                gdf = split_line(gdf, 'LineString', smooth=True)
                gdf.to_file(out_rgb)
                logger.info("Saved RGB shorelines → %s", out_rgb)
            except Exception:
                logger.exception("Failed saving RGB → %s", image)

            # NIR
            try:
                gdf_n = gpd.GeoDataFrame(
                    {
                        'dates': [fn_date] * len(waterlines_nir),
                        'image_suitability_score': shore_img_nir,
                        'segmentation_suitability_score': shore_seg_nir,
                        'satname': [satname] * len(waterlines_nir)
                    },
                    geometry=waterlines_nir,
                    crs=crs_raster
                )
                out_nir = os.path.join(nir_dir, f"{fn_date}_{satname}_{roi}.geojson")
                gdf_n = gdf_n.to_crs("EPSG:4326")
                gdf_n = split_line(gdf_n, 'LineString', smooth=True)
                gdf_n.to_file(out_nir)
                logger.info("Saved NIR shorelines → %s", out_nir)
            except Exception:
                logger.exception("Failed saving NIR → %s", image)

            # SWIR
            if satname != 'PS' and len(waterlines_swir) > 0:
                try:
                    gdf_s = gpd.GeoDataFrame(
                        {
                            'dates': [fn_date] * len(waterlines_swir),
                            'image_suitability_score': shore_img_swir,
                            'segmentation_suitability_score': shore_seg_swir,
                            'satname': [satname] * len(waterlines_swir)
                        },
                        geometry=waterlines_swir,
                        crs=crs_raster
                    )
                    out_s = os.path.join(swir_dir, f"{fn_date}_{satname}_{roi}.geojson")
                    gdf_s = gdf_s.to_crs("EPSG:4326")
                    gdf_s = split_line(gdf_s, 'LineString', smooth=True)
                    gdf_s.to_file(out_s)
                    logger.info("Saved SWIR shorelines → %s", out_s)
                except Exception:
                    logger.exception("Failed saving SWIR → %s", image)

            # Mark processed + write CSV
            df.at[i, 'shoreline_done'] = True
            processed += 1
            try:
                df.to_csv(csv_out + ".tmp", index=False)
                os.replace(csv_out + ".tmp", csv_out)
            except:
                logger.exception("CSV write failed → %s", csv_out)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception index=%d", i)

    # --------------------------------------------------------------------------------------
    # END LOOP
    # --------------------------------------------------------------------------------------



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

def merge_shorelines_section_old(g, c, rr, sss, r_home):
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

def get_merging_logger(section_string: str) -> logging.Logger:
    """
    Logger for shoreline merging:
      <cwd>/logs/merging/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'merging')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"merging.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def merge_shorelines_section(g, c, rr, sss, r_home):
    """
    Merges individual shorelines into one geojson for an entire shoreline section.

    Inputs:
      g (str): global region
      c (str): coastal area
      rr (str): subregion
      sss (str): shoreline section
      r_home (str): path/to/g#/c#/rr##/

    Logging:
      <cwd>/logs/merging/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_merging_logger(section_string)

    # Establish file paths
    extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
    extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
    extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')
    reference_polygon_path = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline_path = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')
    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')

    logger.info("Start merge: section_dir=%s", section_dir)

    # Concatenate individual shorelines
    try:
        zoo_gdf = concat_gdfs_in_folder(zoo_shoreline_dir)
        logger.info("Concatenated zoo RGB: %s → count=%d", zoo_shoreline_dir, 0 if zoo_gdf is None else len(zoo_gdf))
    except Exception:
        logger.exception("Failed to concatenate zoo RGB from %s", zoo_shoreline_dir)
        zoo_gdf = None

    try:
        nir_gdf = concat_gdfs_in_folder(nir_shoreline_dir)
        logger.info("Concatenated NIR: %s → count=%d", nir_shoreline_dir, 0 if nir_gdf is None else len(nir_gdf))
    except Exception:
        logger.exception("Failed to concatenate NIR from %s", nir_shoreline_dir)
        nir_gdf = None

    try:
        swir_gdf = concat_gdfs_in_folder(swir_shoreline_dir)
        logger.info("Concatenated SWIR: %s → count=%d", swir_shoreline_dir, 0 if swir_gdf is None else len(swir_gdf))
    except Exception:
        logger.exception("Failed to concatenate SWIR from %s", swir_shoreline_dir)
        swir_gdf = None

    # Load reference files
    try:
        reference_polygon_gdf = gpd.read_file(reference_polygon_path)
        reference_shoreline_gdf = gpd.read_file(reference_shoreline_path)
        logger.info("Loaded reference polygon and shoreline.")
    except Exception:
        logger.exception("Failed to read reference files: polygon=%s, shoreline=%s", reference_polygon_path, reference_shoreline_path)
        return

    # Clip/explode helpers
    def _clip_explode(gdf, name):
        if gdf is None or len(gdf) == 0:
            logger.warning("Empty %s GDF; skipping clip/explode.", name)
            return None
        try:
            gdf = gpd.clip(gdf, reference_shoreline_gdf['geometry'].iloc[0].buffer(ref_shore_buffer))
            gdf = gpd.clip(gdf, reference_polygon_gdf)
            gdf = gdf.explode(ignore_index=True)
            logger.info("Clipped/exploded %s → count=%d", name, len(gdf))
            return gdf
        except Exception:
            logger.exception("Clip/explode failed for %s", name)
            return None

    zoo_gdf = _clip_explode(zoo_gdf, "zoo")
    nir_gdf = _clip_explode(nir_gdf, "nir")
    swir_gdf = _clip_explode(swir_gdf, "swir")

    # Save outputs
    try:
        if zoo_gdf is not None and len(zoo_gdf) > 0:
            zoo_gdf.to_file(extracted_shorelines_path)
            logger.info("Saved merged zoo RGB → %s (count=%d)", extracted_shorelines_path, len(zoo_gdf))
        else:
            logger.warning("Zoo RGB empty; not writing %s", extracted_shorelines_path)
    except Exception:
        logger.exception("Failed writing merged zoo RGB → %s", extracted_shorelines_path)

    try:
        if nir_gdf is not None and len(nir_gdf) > 0:
            nir_gdf.to_file(extracted_shorelines_nir_path)
            logger.info("Saved merged NIR → %s (count=%d)", extracted_shorelines_nir_path, len(nir_gdf))
        else:
            logger.warning("NIR empty; not writing %s", extracted_shorelines_nir_path)
    except Exception:
        logger.exception("Failed writing merged NIR → %s", extracted_shorelines_nir_path)

    try:
        if swir_gdf is not None and len(swir_gdf) > 0:
            swir_gdf.to_file(extracted_shorelines_swir_path)
            logger.info("Saved merged SWIR → %s (count=%d)", extracted_shorelines_swir_path, len(swir_gdf))
        else:
            logger.warning("SWIR empty; not writing %s", extracted_shorelines_swir_path)
    except Exception:
        logger.exception("Failed writing merged SWIR → %s", extracted_shorelines_swir_path)


def resample_shorelines_section_old(g, c, rr, sss, r_home):
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

def get_resampling_logger(section_string: str) -> logging.Logger:
    """
    Logger for shoreline resampling:
      <cwd>/logs/resampling/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'resampling')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"resampling.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def resample_shorelines_section(g, c, rr, sss, r_home):
    """
    Resamples shorelines to a point every 10 m for a shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/

    Logging:
      <cwd>/logs/resampling/<section_string>_<timestamp>.log
    """
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_resampling_logger(section_string)

    # Establish file paths
    extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
    extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
    extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')

    logger.info("Resampling start: section_dir=%s", section_dir)

    # Load in gdfs and convert to UTM (so we can resample in units of meters)
    try:
        extracted_shorelines_gdf = wgs84_to_utm_df(gpd.read_file(extracted_shorelines_path))
        logger.info("Loaded RGB shorelines: %s (count=%d)", extracted_shorelines_path, len(extracted_shorelines_gdf))
    except Exception:
        logger.exception("Failed loading RGB shorelines → %s", extracted_shorelines_path)
        extracted_shorelines_gdf = gpd.GeoDataFrame(geometry=[], crs=None)

    try:
        extracted_shorelines_nir_gdf = wgs84_to_utm_df(gpd.read_file(extracted_shorelines_nir_path))
        logger.info("Loaded NIR shorelines: %s (count=%d)", extracted_shorelines_nir_path, len(extracted_shorelines_nir_gdf))
    except Exception:
        logger.exception("Failed loading NIR shorelines → %s", extracted_shorelines_nir_path)
        extracted_shorelines_nir_gdf = gpd.GeoDataFrame(geometry=[], crs=None)

    try:
        extracted_shorelines_swir_gdf = wgs84_to_utm_df(gpd.read_file(extracted_shorelines_swir_path))
        logger.info("Loaded SWIR shorelines: %s (count=%d)", extracted_shorelines_swir_path, len(extracted_shorelines_swir_gdf))
    except Exception:
        logger.exception("Failed loading SWIR shorelines → %s", extracted_shorelines_swir_path)
        extracted_shorelines_swir_gdf = gpd.GeoDataFrame(geometry=[], crs=None)

    # RGB zoo output
    try:
        for idx in tqdm(extracted_shorelines_gdf.index, desc=f"Resampling RGB {section_string}", unit="line", total=len(extracted_shorelines_gdf)):
            geom = extracted_shorelines_gdf.at[idx, 'geometry']
            new_geom = resample_line_by_distance(geom, 10)
            extracted_shorelines_gdf.at[idx, 'geometry'] = new_geom
        logger.info("Resampled RGB shorelines (10 m spacing).")
    except Exception:
        logger.exception("Resampling failed for RGB shorelines.")

    # NIR threshold output
    try:
        for idx in tqdm(extracted_shorelines_nir_gdf.index, desc=f"Resampling NIR {section_string}", unit="line", total=len(extracted_shorelines_nir_gdf)):
            geom = extracted_shorelines_nir_gdf.at[idx, 'geometry']
            new_geom = resample_line_by_distance(geom, 10)
            extracted_shorelines_nir_gdf.at[idx, 'geometry'] = new_geom
        logger.info("Resampled NIR shorelines (10 m spacing).")
    except Exception:
        logger.exception("Resampling failed for NIR shorelines.")

    # SWIR threshold output
    try:
        for idx in tqdm(extracted_shorelines_swir_gdf.index, desc=f"Resampling SWIR {section_string}", unit="line", total=len(extracted_shorelines_swir_gdf)):
            geom = extracted_shorelines_swir_gdf.at[idx, 'geometry']
            new_geom = resample_line_by_distance(geom, 10)
            extracted_shorelines_swir_gdf.at[idx, 'geometry'] = new_geom
        logger.info("Resampled SWIR shorelines (10 m spacing).")
    except Exception:
        logger.exception("Resampling failed for SWIR shorelines.")

    # Drop nans from the gdfs
    try:
        before = len(extracted_shorelines_gdf)
        extracted_shorelines_gdf = extracted_shorelines_gdf.dropna(subset=['geometry'])
        logger.info("RGB dropna: %d → %d", before, len(extracted_shorelines_gdf))
    except Exception:
        logger.exception("dropna failed for RGB.")

    try:
        before = len(extracted_shorelines_nir_gdf)
        extracted_shorelines_nir_gdf = extracted_shorelines_nir_gdf.dropna(subset=['geometry'])
        logger.info("NIR dropna: %d → %d", before, len(extracted_shorelines_nir_gdf))
    except Exception:
        logger.exception("dropna failed for NIR.")

    try:
        before = len(extracted_shorelines_swir_gdf)
        extracted_shorelines_swir_gdf = extracted_shorelines_swir_gdf.dropna(subset=['geometry'])
        logger.info("SWIR dropna: %d → %d", before, len(extracted_shorelines_swir_gdf))
    except Exception:
        logger.exception("dropna failed for SWIR.")

    # Convert back to WGS84
    try:
        extracted_shorelines_gdf = utm_to_wgs84_df(extracted_shorelines_gdf)
        logger.info("Converted RGB to WGS84.")
    except Exception:
        logger.exception("UTM→WGS84 conversion failed for RGB.")

    try:
        extracted_shorelines_nir_gdf = utm_to_wgs84_df(extracted_shorelines_nir_gdf)
        logger.info("Converted NIR to WGS84.")
    except Exception:
        logger.exception("UTM→WGS84 conversion failed for NIR.")

    try:
        extracted_shorelines_swir_gdf = utm_to_wgs84_df(extracted_shorelines_swir_gdf)
        logger.info("Converted SWIR to WGS84.")
    except Exception:
        logger.exception("UTM→WGS84 conversion failed for SWIR.")

    # Save each file (with defensive checks)
    try:
        if len(extracted_shorelines_gdf) > 0:
            extracted_shorelines_gdf.to_file(extracted_shorelines_path)
            logger.info("Saved RGB resampled shorelines → %s (count=%d)", extracted_shorelines_path, len(extracted_shorelines_gdf))
        else:
            logger.warning("RGB resampled GDF empty; not writing %s", extracted_shorelines_path)
    except Exception:
        logger.exception("Failed writing RGB resampled → %s", extracted_shorelines_path)

    try:
        if len(extracted_shorelines_nir_gdf) > 0:
            extracted_shorelines_nir_gdf.to_file(extracted_shorelines_nir_path)
            logger.info("Saved NIR resampled shorelines → %s (count=%d)", extracted_shorelines_nir_path, len(extracted_shorelines_nir_gdf))
        else:
            logger.warning("NIR resampled GDF empty; not writing %s", extracted_shorelines_nir_path)
    except Exception:
        logger.exception("Failed writing NIR resampled → %s", extracted_shorelines_nir_path)

    try:
        if len(extracted_shorelines_swir_gdf) > 0:
            extracted_shorelines_swir_gdf.to_file(extracted_shorelines_swir_path)
            logger.info("Saved SWIR resampled shorelines → %s (count=%d)", extracted_shorelines_swir_path, len(extracted_shorelines_swir_gdf))
        else:
            logger.warning("SWIR resampled GDF empty; not writing %s", extracted_shorelines_swir_path)
    except Exception:
        logger.exception("Failed writing SWIR resampled → %s", extracted_shorelines_swir_path)


def record_stats_shoreline_section_old(g, c, rr, sss, r_home):
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



def _get_logger_record_statistics(section_string: str, level: int = logging.INFO) -> logging.Logger:
    """
    File-only logger for shoreline record statistics:
      <cwd>/logs/record_statistics/<section_string>_<YYYYMMDD_%H%M%S>.log

    - Creates the log directory if missing.
    - Uses a timestamped file per invocation (no rotation).
    - Prevents duplicate handlers if requested multiple times.
    - No console handler (no stdout/stderr output).
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'record_statistics')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"record_statistics.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs via root

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler only
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def record_stats_shoreline_section(g, c, rr, sss, r_home):
    section_str = g + c + rr + sss
    section_dir = os.path.join(r_home, 'SSS' + sss)

    # File-only logger: <cwd>/logs/record_statistics/<section_str>_<YYYYMMDD_%H%M%S>.log
    logger = _get_logger_record_statistics(section_str, level=logging.INFO)

    logger.info("=== Record Statistics Start ===")
    logger.info("Inputs | G=%s | C=%s | RR=%s | SSS=%s | section_dir=%s", g, c, rr, sss, section_dir)

    reprojected_points = os.path.join(section_dir, section_str + '_reprojected_points.geojson')
    transects = os.path.join(section_dir, section_str + '_transects.geojson')

    logger.info("Reading files | reprojected_points=%s | transects=%s", reprojected_points, transects)

    try:
        transects_gdf = gpd.read_file(transects)
    except Exception as e:
        logger.exception("Failed to read transects: %s", transects)
        raise

    try:
        points_gdf = gpd.read_file(reprojected_points)
    except Exception as e:
        logger.exception("Failed to read reprojected points: %s", reprojected_points)
        raise

    org_crs = transects_gdf.crs
    utm_crs = transects_gdf.estimate_utm_crs()

    logger.info("CRS | org_crs=%s | utm_crs=%s", org_crs, utm_crs)

    transects_utm = transects_gdf.to_crs(utm_crs)
    points_utm = points_gdf.to_crs(utm_crs)

    transect_ids = sorted(np.unique(points_gdf['transect_id']))
    logger.info("Counts | transects=%d | points=%d | unique_transect_ids=%d",
                len(transects_gdf), len(points_gdf), len(transect_ids))

    min_vals = [None] * len(transect_ids)
    q1_vals = [None] * len(transect_ids)
    median_vals = [None] * len(transect_ids)
    mean_vals = [None] * len(transect_ids)
    q3_vals = [None] * len(transect_ids)
    max_vals = [None] * len(transect_ids)

    iqr_vals = [None] * len(transect_ids)
    mad_vals = [None] * len(transect_ids)
    std_vals = [None] * len(transect_ids)
    cv_vals = [None] * len(transect_ids)
    skew_vals = [None] * len(transect_ids)
    kurt_vals = [None] * len(transect_ids)

    geometry_from_centroids = [None] * len(transect_ids)
    transect_ids_ = [None] * len(transect_ids)

    geom_min_utm = [None] * len(transect_ids)
    geom_q1_utm = [None] * len(transect_ids)
    geom_median_utm = [None] * len(transect_ids)
    geom_mean_utm = [None] * len(transect_ids)
    geom_q3_utm = [None] * len(transect_ids)
    geom_max_utm = [None] * len(transect_ids)

    skipped = 0

    for i in range(len(transect_ids)):
        try:
            transect_id = transect_ids[i]

            transect = transects_utm[transects_utm['transect_id'] == transect_id].reset_index(drop=True).iloc[0]
            first = transect.geometry.coords[0]
            last = transect.geometry.coords[1]
            angle = np.arctan2(last[1] - first[1], last[0] - first[0])

            pts_utm = points_utm[points_utm['transect_id'] == transect_id]
            pts_orig = points_gdf[points_gdf['transect_id'] == transect_id]

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

            geom_min_utm[i] = shapely.Point(first[0] + d_min * np.cos(angle),
                                            first[1] + d_min * np.sin(angle))
            geom_q1_utm[i] = shapely.Point(first[0] + q1 * np.cos(angle),
                                           first[1] + q1 * np.sin(angle))
            geom_median_utm[i] = shapely.Point(first[0] + med * np.cos(angle),
                                               first[1] + med * np.sin(angle))
            geom_mean_utm[i] = shapely.Point(first[0] + mean_val * np.cos(angle),
                                             first[1] + mean_val * np.sin(angle))
            geom_q3_utm[i] = shapely.Point(first[0] + q3 * np.cos(angle),
                                           first[1] + q3 * np.sin(angle))
            geom_max_utm[i] = shapely.Point(first[0] + d_max * np.cos(angle),
                                            first[1] + d_max * np.sin(angle))

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

        except Exception as e:
            skipped += 1
            logger.debug("Skipped transect_id=%s due to error: %s", transect_ids[i], str(e))
            continue

    logger.info("Loop summary | processed=%d | skipped=%d", len(transect_ids) - skipped, skipped)

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
        try:
            gdf = gpd.GeoDataFrame(
                {
                    'transect_id': transect_ids_,
                    **vals,
                    'geometry_from_centroids': geometry_from_centroids,
                },
                geometry=geom,
                crs=org_crs
            )
            points_out = os.path.join(section_dir, f"{section_str}_{name}_shoreline_points.geojson")
            gdf.to_file(points_out)
            logger.info("Saved points (%s): %s | rows=%d", name, points_out, len(gdf))

            line = shapely.geometry.LineString(geom)
            line_gdf = gpd.GeoDataFrame(
                {'G': [g], 'C': [c], 'RR': [rr], 'SSS': [sss]},
                geometry=[line],
                crs=org_crs
            )
            line_out = os.path.join(section_dir, f"{section_str}_{name}_shoreline.geojson")
            line_gdf.to_file(line_out)
            logger.info("Saved line (%s): %s", name, line_out)

        except Exception as e:
            logger.exception("Failed to save %s outputs: %s", name, str(e))
            raise

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

    logger.info("record stats computed")
    logger.info("=== Record Statistics End ===")

    print('record stats computed')

def spatial_kde_section_old(g, c, rr, sss, r_home):
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

def get_spatial_kde_logger(section_string: str) -> logging.Logger:
    """
    Logger for spatial KDE:
      <cwd>/logs/spatial_kde/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'spatial_kde')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"spatial_kde.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def spatial_kde_section(g, c, rr, sss, r_home):
    """
    Computes spatial KDE by shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##/

    Logging:
      <cwd>/logs/spatial_kde/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_spatial_kde_logger(section_string)

    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')

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

    # Load reference shoreline and project to UTM
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
        crs = reference_shoreline_gdf.crs
        logger.info("Loaded reference shoreline and reprojected to UTM.")
    except Exception:
        logger.exception("Failed to read or reproject reference shoreline → %s", reference_shoreline)
        return

    # Create KDE if needed
    if (not os.path.isfile(kde_path)
        and os.path.isfile(extracted_shorelines_path)
        and os.path.isfile(extracted_shorelines_nir_path)
        and os.path.isfile(extracted_shorelines_swir_path)):
        try:
            shorelines_concat = pd.concat([
                gpd.read_file(extracted_shorelines_path),
                gpd.read_file(extracted_shorelines_nir_path),
                gpd.read_file(extracted_shorelines_swir_path)
            ])
            logger.info("Concatenated shorelines for KDE: count=%d", len(shorelines_concat))
            shoreline_change_envelope.point_density_grid(
                shorelines_concat,
                kde_path,
                15
            )
            logger.info("Computed point density KDE → %s", kde_path)
        except Exception:
            logger.exception("KDE computation failed.")

    # Otsu thresholding and masking
    if os.path.isfile(kde_path):
        try:
            shoreline_change_envelope.compute_otsu_threshold(kde_path, otsu_path)
            shoreline_change_envelope.binary_raster_to_vector(otsu_path, otsu_geojson)
            logger.info("Computed Otsu threshold and vectorized mask → %s", otsu_geojson)
        except Exception:
            logger.exception("Otsu threshold or vectorization failed.")
            return

        try:
            otsu_geojson_gdf = gpd.read_file(otsu_geojson)
            utm_crs = otsu_geojson_gdf.crs

            zoo_shorelines_gdf = gpd.read_file(extracted_shorelines_path)
            swir_shorelines_gdf = gpd.read_file(extracted_shorelines_swir_path)
            nir_shorelines_gdf = gpd.read_file(extracted_shorelines_nir_path)
            wgs84_crs = zoo_shorelines_gdf.crs

            # Reproject inputs to UTM of Otsu mask
            zoo_shorelines_gdf = zoo_shorelines_gdf.to_crs(utm_crs)
            swir_shorelines_gdf = swir_shorelines_gdf.to_crs(utm_crs)
            nir_shorelines_gdf = nir_shorelines_gdf.to_crs(utm_crs)

            # Clip by Otsu mask and convert back to WGS84
            zoo_shorelines_gdf = gpd.clip(zoo_shorelines_gdf, otsu_geojson_gdf).to_crs(wgs84_crs)
            swir_shorelines_gdf = gpd.clip(swir_shorelines_gdf, otsu_geojson_gdf).to_crs(wgs84_crs)
            nir_shorelines_gdf = gpd.clip(nir_shorelines_gdf, otsu_geojson_gdf).to_crs(wgs84_crs)

            # Explode, keep only LineString, trim ends, drop short segments, add year, vertex filter
            def _process_and_filter(gdf, name):
                if gdf is None or len(gdf) == 0:
                    logger.warning("%s: empty after clip; skipping.", name)
                    return gdf
                try:
                    gdf = gdf.explode(ignore_index=True)
                    gdf['type'] = gdf['geometry'].type
                    gdf = gdf[gdf['type'] == 'LineString'].reset_index(drop=True)

                    # Trim 5 vertices from both ends (progress bar for visibility)
                    for idx, row in tqdm(gdf.iterrows(), desc=f"Trim {name} {section_string}", unit="line", total=len(gdf)):
                        geom_arr = LineString_to_arr(row['geometry'])
                        filtered_geom = geom_arr[5:-5]
                        if len(filtered_geom) < 5:
                            new_geom = None
                        else:
                            new_geom = arr_to_LineString(filtered_geom)
                        gdf.at[idx, 'geometry'] = new_geom

                    gdf = gdf.dropna(subset=['geometry']).reset_index(drop=True)
                    # Add year from dates (expects format '%Y-%m-%d-%H-%M-%S')
                    gdf['year'] = pd.to_datetime(gdf['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True, errors='coerce').dt.year
                    gdf = vertex_filter(gdf)
                    logger.info("%s: processed count=%d", name, len(gdf))
                    return gdf
                except Exception:
                    logger.exception("%s: processing failed.", name)
                    return gdf

            zoo_shorelines_gdf = _process_and_filter(zoo_shorelines_gdf, "RGB")
            swir_shorelines_gdf = _process_and_filter(swir_shorelines_gdf, "SWIR")
            nir_shorelines_gdf = _process_and_filter(nir_shorelines_gdf, "NIR")

            # Save filtered outputs
            try:
                if len(zoo_shorelines_gdf) > 0:
                    zoo_shorelines_gdf.to_file(extracted_shorelines_path_filter)
                    logger.info("Saved filtered RGB shorelines → %s (count=%d)", extracted_shorelines_path_filter, len(zoo_shorelines_gdf))
                else:
                    logger.warning("RGB filtered GDF empty; not writing %s", extracted_shorelines_path_filter)
            except Exception:
                logger.exception("Failed writing RGB filtered → %s", extracted_shorelines_path_filter)

            try:
                if len(swir_shorelines_gdf) > 0:
                    swir_shorelines_gdf.to_file(extracted_shorelines_swir_path_filter)
                    logger.info("Saved filtered SWIR shorelines → %s (count=%d)", extracted_shorelines_swir_path_filter, len(swir_shorelines_gdf))
                else:
                    logger.warning("SWIR filtered GDF empty; not writing %s", extracted_shorelines_swir_path_filter)
            except Exception:
                logger.exception("Failed writing SWIR filtered → %s", extracted_shorelines_swir_path_filter)

            try:
                if len(nir_shorelines_gdf) > 0:
                    nir_shorelines_gdf.to_file(extracted_shorelines_nir_path_filter)
                    logger.info("Saved filtered NIR shorelines → %s (count=%d)", extracted_shorelines_nir_path_filter, len(nir_shorelines_gdf))
                else:
                    logger.warning("NIR filtered GDF empty; not writing %s", extracted_shorelines_nir_path_filter)
            except Exception:
                logger.exception("Failed writing NIR filtered → %s", extracted_shorelines_nir_path_filter)

        except Exception:
            logger.exception("Filtering by Otsu mask failed.")
            return

    # Sample spatial KDE for each shoreline set
    if os.path.isfile(kde_path) and os.path.isfile(extracted_shorelines_path):
        for pth, label in [
            (extracted_shorelines_path, "RGB"),
            (extracted_shorelines_nir_path, "NIR"),
            (extracted_shorelines_swir_path, "SWIR"),
        ]:
            try:
                sample_spatial_kde(kde_path, pth, crs)
                logger.info("Sampled KDE for %s shorelines.", label)
            except Exception:
                logger.exception("Sampling KDE failed for %s → %s", label, pth)


def get_trends_section_old(g, c, rr, sss, r_home):
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
 
def get_trend_computation_logger(section_string: str) -> logging.Logger:
    """
    Logger for trend computation:
        <cwd>/logs/trend_computation/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'trend_computation')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"trend_computation.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_trends_section(g, c, rr, sss, r_home):
    """
    Computes trends for shoreline section.

    inputs:
      g (str): global region
      c (str): coastal area
      rr (str): subregion
      sss (str): shoreline section
      r_home (str): path/to/g#/c#/rr##/

    Logging:
      <cwd>/logs/trend_computation/<section_string>_<timestamp>.log
    """
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_trend_computation_logger(section_string)

    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')

    # Load reference shoreline and project to UTM (for consistency)
    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
        crs = reference_shoreline_gdf.crs
        logger.info("Loaded reference shoreline and reprojected to UTM.")
    except Exception:
        logger.exception("Failed to read/reproject reference shoreline → %s", reference_shoreline)
        return

    extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
    extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
    extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')
    transects = os.path.join(section_dir, section_string + '_transects.geojson')

    # Only compute if all extracted shoreline files exist
    if (os.path.isfile(extracted_shorelines_path)
        and os.path.isfile(extracted_shorelines_nir_path)
        and os.path.isfile(extracted_shorelines_swir_path)):

        transect_trends = os.path.join(section_dir, section_string + '_transects_trends.geojson')
        resampled_csv = os.path.join(section_dir, section_string + '_resampled_tidally_corrected_transect_time_series_merged.csv')

        # Call get_trends (pure function: no logging inside)
        try:
            logger.info("Computing trends with get_trends → input=%s, transects=%s, out=%s",
                        resampled_csv, transects, transect_trends)
            get_trends(
                resampled_csv,
                transects,
                transect_trends,
            )
            logger.info("get_trends completed → %s", transect_trends)
        except Exception:
            logger.exception("get_trends failed.")
            return

        # Post-process: mark significance and save back
        try:
            gdf = gpd.read_file(transect_trends)
            if 'linear_trend' not in gdf.columns or 'linear_trend_95_confidence' not in gdf.columns:
                logger.warning("Expected trend columns missing in %s; columns present: %s",
                               transect_trends, list(gdf.columns))
            else:
                gdf['significant'] = (np.abs(gdf['linear_trend']) > gdf['linear_trend_95_confidence']).astype(int)

            try:
                gdf.to_file(transect_trends)
                logger.info("Saved trends (with significance flag) → %s (rows=%d)", transect_trends, len(gdf))
            except Exception:
                logger.exception("Failed writing trend geojson → %s", transect_trends)
        except Exception:
            logger.exception("Failed reading trend geojson → %s", transect_trends)
            return
    else:
        logger.warning(
            "Missing extracted shorelines; skipping trend computation. "
            "RGB=%s exists=%s, NIR=%s exists=%s, SWIR=%s exists=%s",
            extracted_shorelines_path, os.path.isfile(extracted_shorelines_path),
            extracted_shorelines_nir_path, os.path.isfile(extracted_shorelines_nir_path),
            extracted_shorelines_swir_path, os.path.isfile(extracted_shorelines_swir_path)
        )
        return

    logger.info("Trend computation completed for section=%s", section_string)

def transect_timeseries_section_old(g, c, rr, sss, r_home, waterline_filter=False):
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

def get_computing_intersections_logger(section_string: str) -> logging.Logger:
    """
    Logger for computing transect/shoreline intersections:
      <cwd>/logs/computing_intersections/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'computing_intersections')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"computing_intersections.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def transect_timeseries_section(g, c, rr, sss, r_home, waterline_filter=False):
    """
    Computes transect and shoreline intersections by shoreline section.

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/g#/c#/rr##
    
    Logging:
      <cwd>/logs/computing_intersections/<section_string>_<timestamp>.log
    """
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_computing_intersections_logger(section_string)

    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')

    try:
        reference_shoreline_gdf = gpd.read_file(reference_shoreline)
        reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
        crs = reference_shoreline_gdf.crs
    except Exception:
        logger.exception("Failed to read or reproject reference shoreline → %s", reference_shoreline)
        return

    if waterline_filter:
        extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines_filter.geojson')
        extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh_filter.geojson')
        extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh_filter.geojson')
        logger.info("Using waterline filter.")
    else:
        extracted_shorelines_path = os.path.join(section_dir, section_string + '_extracted_shorelines.geojson')
        extracted_shorelines_nir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_nir_thresh.geojson')
        extracted_shorelines_swir_path = os.path.join(section_dir, section_string + '_extracted_shorelines_swir_thresh.geojson')
        logger.info("Not using waterline filter.")

    transects = os.path.join(section_dir, section_string + '_transects.geojson')

    # Process RGB shorelines
    if os.path.isfile(extracted_shorelines_path):
        try:
            merged_csv_rgb = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged.csv')
            matrix_csv_rgb = os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix.csv')
            transect_timeseries(extracted_shorelines_path, transects, reference_polygon, crs, merged_csv_rgb, matrix_csv_rgb)
            logger.info("Computed intersections (RGB). Outputs: merged=%s, matrix=%s", merged_csv_rgb, matrix_csv_rgb)
        except Exception:
            logger.exception("Failed computing intersections for RGB → %s", extracted_shorelines_path)
    else:
        logger.warning("RGB waterline file missing → %s", extracted_shorelines_path)

    # Process NIR shorelines
    if os.path.isfile(extracted_shorelines_nir_path):
        try:
            merged_csv_nir = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_nir_thresh.csv')
            matrix_csv_nir = os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix_nir_thresh.csv')
            transect_timeseries(extracted_shorelines_nir_path, transects, reference_polygon, crs, merged_csv_nir, matrix_csv_nir)
            logger.info("Computed intersections (NIR). Outputs: merged=%s, matrix=%s", merged_csv_nir, matrix_csv_nir)
        except Exception:
            logger.exception("Failed computing intersections for NIR → %s", extracted_shorelines_nir_path)
    else:
        logger.warning("NIR waterline file missing → %s", extracted_shorelines_nir_path)

    # Process SWIR shorelines
    if os.path.isfile(extracted_shorelines_swir_path):
        try:
            merged_csv_swir = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_swir_thresh.csv')
            matrix_csv_swir = os.path.join(section_dir, section_string + '_raw_transect_time_series_matrix_swir_thresh.csv')
            transect_timeseries(extracted_shorelines_swir_path, transects, reference_polygon, crs, merged_csv_swir, matrix_csv_swir)
            logger.info("Computed intersections (SWIR). Outputs: merged=%s, matrix=%s", merged_csv_swir, matrix_csv_swir)
        except Exception:
            logger.exception("Failed computing intersections for SWIR → %s", extracted_shorelines_swir_path)
    else:
        logger.warning("SWIR waterline file missing → %s", extracted_shorelines_swir_path)


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

def get_tide_data_section_old(g, c, rr, sss, r_home):
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

def get_tide_data_logger(section_string: str) -> logging.Logger:
    """
    Logger for tide data retrieval:
      <cwd>/logs/tide_data_retrieval/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'tide_data_retrieval')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"tide_data_retrieval.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


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
    
    Logging:
      <cwd>/logs/tide_data_retrieval/<section_string>_<timestamp>.log
    """
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_tide_data_logger(section_string)

    transects = os.path.join(section_dir, section_string + '_transects.geojson')
    tide_model_dir = os.path.join(get_script_path(), 'CoastSeg', 'tide_model')
    timeseries_zoo = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged.csv')
    timeseries_swir = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_swir_thresh.csv')
    timeseries_nir = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_nir_thresh.csv')
    tide_data_path = os.path.join(section_dir, section_string + '_tides.csv')

    logger.info("Starting tide data retrieval for section=%s", section_string)
    logger.info("Tide model dir: %s", tide_model_dir)

    # Read time series CSVs
    try:
        timeseries_zoo_df = pd.read_csv(timeseries_zoo)
        logger.info("Loaded RGB timeseries: %s (rows=%d)", timeseries_zoo, len(timeseries_zoo_df))
    except Exception:
        logger.exception("Failed to read RGB timeseries → %s", timeseries_zoo)
        return

    try:
        timeseries_swir_df = pd.read_csv(timeseries_swir)
        logger.info("Loaded SWIR timeseries: %s (rows=%d)", timeseries_swir, len(timeseries_swir_df))
    except Exception:
        logger.exception("Failed to read SWIR timeseries → %s", timeseries_swir)
        return

    try:
        timeseries_nir_df = pd.read_csv(timeseries_nir)
        logger.info("Loaded NIR timeseries: %s (rows=%d)", timeseries_nir, len(timeseries_nir_df))
    except Exception:
        logger.exception("Failed to read NIR timeseries → %s", timeseries_nir)
        return

    # Collect unique dates
    try:
        zoo_dates = timeseries_zoo_df['dates']
        swir_dates = timeseries_swir_df['dates']
        nir_dates = timeseries_nir_df['dates']
        dates_concat = pd.concat([zoo_dates, swir_dates, nir_dates])
        unique_dates = np.unique(dates_concat)
        unique_dates = pd.to_datetime(unique_dates, format='%Y-%m-%d-%H-%M-%S', utc=True).values
        logger.info("Unique timestamps gathered: %d", len(unique_dates))
    except Exception:
        logger.exception("Failed to parse unique dates.")
        return

    # Load transects and compute end points
    try:
        transects_gdf = gpd.read_file(transects)
        logger.info("Loaded transects: %s (rows=%d)", transects, len(transects_gdf))
    except Exception:
        logger.exception("Failed to read transects → %s", transects)
        return

    try:
        end_coords = transects_gdf['geometry'].apply(lambda line: Point(line.coords[-1]))
        transect_ids = transects_gdf['transect_id'] if 'transect_id' in transects_gdf.columns else pd.Series(range(len(transects_gdf)))
        x = end_coords.x
        y = end_coords.y
        transects_gdf['x'] = x
        transects_gdf['y'] = y
        logger.info("Computed transect endpoints (count=%d).", len(transects_gdf))
    except Exception:
        logger.exception("Failed to compute transect endpoint coordinates.")
        return

    # Tide model inference
    try:
        tide_data = tide_correction.model_tides(
            x, y, unique_dates,
            model="FES2022",
            directory=tide_model_dir,
            epsg=4326,
            method="bilinear",
            extrapolate=True,
            cutoff=10.0
        )
        logger.info("Tide model computed: rows=%d", len(tide_data))
    except Exception:
        logger.exception("Tide model computation failed.")
        return

    # Merge to include transect metadata
    try:
        tides = pd.merge(tide_data, transects_gdf, how='left', on=['x', 'y'])
    except Exception:
        logger.exception("Failed to merge tide data with transects.")
        return

    # Save tides CSV (atomic replace)
    try:
        tmp_path = tide_data_path + ".tmp"
        tides.to_csv(tmp_path, index=False)
        os.replace(tmp_path, tide_data_path)
        logger.info("Saved tide data → %s (rows=%d)", tide_data_path, len(tides))
    except Exception:
        logger.exception("Failed to write tide data CSV → %s", tide_data_path)
        return


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
        
def tidally_correct_section_old(g, 
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

def get_tidal_correction_logger(section_string: str) -> logging.Logger:
    """
    Logger for tidal correction:
      <cwd>/logs/tidal_correction/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'tidal_correction')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"tidal_correction.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def tidally_correct_section(
    g, c, rr, sss, r_home, dem='', reference_elevation=0
):
    """
    Tidally corrects shoreline data within a shoreline section

    inputs:
    g (str): global region
    c (str): coastal area
    rr (str): subregion
    sss (str): shoreline section
    r_home (str): path/to/subregion
    reference elevation (float): reference elevation for correction

    Logging:
      <cwd>/logs/tidal_correction/<section_string>_<timestamp>.log
    """
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_tidal_correction_logger(section_string)

    zoo_path = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged.csv')
    swir_path = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_swir_thresh.csv')
    nir_path = os.path.join(section_dir, section_string + '_raw_transect_time_series_merged_nir_thresh.csv')
    tide_path = os.path.join(section_dir, section_string + '_tides.csv')
    slope_data_path = os.path.join(section_dir, section_string + f'_transects_slopes_{dem}.geojson')

    out_zoo = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged.csv')
    out_swir = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_swir_thresh.csv')
    out_nir = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_nir_thresh.csv')

    logger.info("Start tidal correction: section=%s, dem='%s', ref_elev=%.3f", section_string, dem, reference_elevation)

    # Load inputs
    try:
        timeseries_zoo = pd.read_csv(zoo_path)
        timeseries_zoo['dates'] = pd.to_datetime(timeseries_zoo['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        logger.info("Loaded zoo timeseries (%d rows) → %s", len(timeseries_zoo), zoo_path)
    except Exception:
        logger.exception("Failed reading zoo timeseries → %s", zoo_path)
        return

    try:
        timeseries_swir = pd.read_csv(swir_path)
        timeseries_swir['dates'] = pd.to_datetime(timeseries_swir['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        logger.info("Loaded swir timeseries (%d rows) → %s", len(timeseries_swir), swir_path)
    except Exception:
        logger.exception("Failed reading swir timeseries → %s", swir_path)
        return

    try:
        timeseries_nir = pd.read_csv(nir_path)
        timeseries_nir['dates'] = pd.to_datetime(timeseries_nir['dates'], format='%Y-%m-%d-%H-%M-%S', utc=True)
        logger.info("Loaded nir timeseries (%d rows) → %s", len(timeseries_nir), nir_path)
    except Exception:
        logger.exception("Failed reading nir timeseries → %s", nir_path)
        return

    try:
        tide_data = pd.read_csv(tide_path)
        tide_data['dates'] = pd.to_datetime(tide_data['dates'], utc=True)
        logger.info("Loaded tide data (%d rows) → %s", len(tide_data), tide_path)
    except Exception:
        logger.exception("Failed reading tide data → %s", tide_path)
        return

    try:
        slope_data = gpd.read_file(slope_data_path)
        logger.info("Loaded slope data (%d rows) → %s", len(slope_data), slope_data_path)
    except Exception:
        logger.exception("Failed reading slope data → %s", slope_data_path)
        return

    # Merge tide data
    try:
        timeseries_zoo_merge = pd.merge(timeseries_zoo, tide_data, how='left', on=['transect_id', 'dates'])
        timeseries_swir_merge = pd.merge(timeseries_swir, tide_data, how='left', on=['transect_id', 'dates'])
        timeseries_nir_merge = pd.merge(timeseries_nir, tide_data, how='left', on=['transect_id', 'dates'])
        logger.info("Merged tide data: zoo=%d, swir=%d, nir=%d",
                    len(timeseries_zoo_merge), len(timeseries_swir_merge), len(timeseries_nir_merge))
    except Exception:
        logger.exception("Merging tide data failed.")
        return

    # First column filter set
    keep_cols_1 = [
        'dates', 'image_suitability_score', 'segmentation_suitability_score',
        'satname', 'kde_value', 'transect_id', 'intersect_x', 'intersect_y',
        'cross_distance', 'x', 'y', 'tide', 'G', 'C', 'RR', 'SSS'
    ]
    try:
        for df_merge in (timeseries_zoo_merge, timeseries_swir_merge, timeseries_nir_merge):
            drop_cols = [c for c in df_merge.columns if c not in keep_cols_1]
            if drop_cols:
                df_merge.drop(columns=drop_cols, inplace=True, errors='ignore')
        logger.info("Dropped non-required columns (phase 1).")
    except Exception:
        logger.exception("Dropping columns (phase 1) failed.")

    # Cast IDs to strings for joins
    try:
        for df_merge in (timeseries_zoo_merge, timeseries_swir_merge, timeseries_nir_merge):
            for col in ['G', 'C', 'RR', 'SSS', 'transect_id']:
                if col in df_merge.columns:
                    df_merge[col] = df_merge[col].astype(str)
        # Slope data casting
        for col in ['G', 'C', 'RR', 'SSS', 'transect_id']:
            if col in slope_data.columns:
                slope_data[col] = slope_data[col].astype(str)
        logger.info("Standardized ID columns to string.")
    except Exception:
        logger.exception("ID casting failed.")

    # Merge slope data
    try:
        timeseries_zoo_merge = pd.merge(timeseries_zoo_merge, slope_data, how='left', on=['transect_id'])
        timeseries_swir_merge = pd.merge(timeseries_swir_merge, slope_data, how='left', on=['transect_id'])
        timeseries_nir_merge = pd.merge(timeseries_nir_merge, slope_data, how='left', on=['transect_id'])
        logger.info("Merged slope data to timeseries.")
    except Exception:
        logger.exception("Merging slope data failed.")
        return

    # Second column filter set
    keep_cols_2 = [
        'dates', 'image_suitability_score', 'segmentation_suitability_score',
        'satname', 'transect_id', 'intersect_x', 'intersect_y',
        'cross_distance', 'x', 'y', 'tide', 'avg_slope_cleaned', 'kde_value'
    ]
    try:
        for df_merge in (timeseries_zoo_merge, timeseries_swir_merge, timeseries_nir_merge):
            drop_cols = [c for c in df_merge.columns if c not in keep_cols_2]
            if drop_cols:
                df_merge.drop(columns=drop_cols, inplace=True, errors='ignore')
        logger.info("Dropped non-required columns (phase 2).")
    except Exception:
        logger.exception("Dropping columns (phase 2) failed.")

    # Apply tidal correction: cross_distance - (tide - ref_elev) / slope
    try:
        for df_merge, name in (
            (timeseries_zoo_merge, "zoo"),
            (timeseries_swir_merge, "swir"),
            (timeseries_nir_merge, "nir"),
        ):
            df_merge['cross_distance_tidally_corrected'] = (
                df_merge['cross_distance'] - (df_merge['tide'] - reference_elevation) / df_merge['avg_slope_cleaned']
            )
            logger.info("Applied tidal correction for %s (%d rows).", name, len(df_merge))
    except Exception:
        logger.exception("Applying tidal correction failed.")
        return

    # Save outputs (atomic replace)
    def _safe_csv_write(df: pd.DataFrame, path: str, label: str):
        try:
            tmp = path + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
            logger.info("Saved %s → %s (rows=%d)", label, path, len(df))
        except Exception:
            logger.exception("Writing %s failed → %s", label, path)

    _safe_csv_write(timeseries_zoo_merge, out_zoo, "tidally corrected RGB")
    _safe_csv_write(timeseries_swir_merge, out_swir, "tidally corrected SWIR")
    _safe_csv_write(timeseries_nir_merge, out_nir, "tidally corrected NIR")



def clip_extracted_shoreline_section_old(g, c, rr, sss, r_home, planet=False):
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

def get_clipping_logger(section_string: str) -> logging.Logger:
    """
    Logger for clipping:
      <cwd>/logs/clipping/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'clipping')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"clipping.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def clip_extracted_shoreline_section(g, c, rr, sss, r_home, planet=False):
    ref_shore_buffer = 400
    section_dir = os.path.join(r_home, 'SSS' + sss)
    section_string = g + c + rr + sss
    logger = get_clipping_logger(section_string)

    reference_polygon = os.path.join(section_dir, section_string + '_reference_polygon.geojson')
    reference_shoreline = os.path.join(section_dir, section_string + '_reference_shoreline.geojson')

    if planet is True:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_segmented.csv')
        if os.path.isfile(sat_image_list_df_path):
            sat_image_list_df_path_clipped = os.path.join(section_dir, section_string + '_ms_lists', 'planet_ms_paths_scored_clip.csv')
        else:
            logger.warning("Planet: segmented CSV not found → %s", sat_image_list_df_path)
            return
    else:
        sat_image_list_df_path = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_segmented.csv')
        sat_image_list_df_path_clipped = os.path.join(section_dir, section_string + '_ms_lists', 'landsat_sentinel_ms_paths_scored_clip.csv')

    # Load clipped CSV if present; else base CSV and initialize 'clipped'
    try:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path_clipped)
        logger.info("Loaded clipped CSV → %s", sat_image_list_df_path_clipped)
    except Exception:
        sat_image_list_df = pd.read_csv(sat_image_list_df_path)
        if 'clipped' not in sat_image_list_df.columns:
            sat_image_list_df['clipped'] = [False] * len(sat_image_list_df)
        logger.info("Loaded base CSV and initialized 'clipped' → %s", sat_image_list_df_path)

    shorelines_dir = os.path.join(section_dir, 'shorelines')
    zoo_shoreline_dir = os.path.join(shorelines_dir, 'zoo_rgb')
    nir_shoreline_dir = os.path.join(shorelines_dir, 'nir_thresh')
    swir_shoreline_dir = os.path.join(shorelines_dir, 'swir_thresh')

    for d in (shorelines_dir, zoo_shoreline_dir, nir_shoreline_dir, swir_shoreline_dir):
        os.makedirs(d, exist_ok=True)

    num_images = len(sat_image_list_df)
    if num_images == 0:
        logger.info("No rows to process. CSV → %s", sat_image_list_df_path)
        return

    processed = 0
    skipped_score = 0
    skipped_missing = 0
    already_done = 0
    errors = 0

    for i in tqdm(range(len(sat_image_list_df['analysis_image'])),
                  desc=f"Clipping {section_string}", unit="img",
                  total=len(sat_image_list_df['analysis_image'])):
        try:
            image = sat_image_list_df['analysis_image'].iloc[i]
            roi_folder = sat_image_list_df['roi_folder'].iloc[i]
            try:
                image_suitability_score = float(sat_image_list_df['model_scores'].iloc[i])
            except Exception:
                image_suitability_score = 0.0

            # Skip low score
            if image_suitability_score < 0.335:
                sat_image_list_df.at[i, 'clipped'] = True
                skipped_score += 1
                # Persist progressively (atomic)
                try:
                    tmp_path = sat_image_list_df_path_clipped + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_clipped)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_clipped)
                continue

            # Skip missing image
            if image is None or (isinstance(image, float) and np.isnan(image)):
                sat_image_list_df.at[i, 'clipped'] = True
                skipped_missing += 1
                try:
                    tmp_path = sat_image_list_df_path_clipped + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_clipped)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_clipped)
                continue

            # Already clipped
            if bool(sat_image_list_df['clipped'].iloc[i]) is True:
                already_done += 1
                try:
                    tmp_path = sat_image_list_df_path_clipped + ".tmp"
                    sat_image_list_df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, sat_image_list_df_path_clipped)
                except Exception:
                    logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_clipped)
                continue

            satname = sat_image_list_df['satnames'].iloc[i]
            date = sat_image_list_df['datetimes_utc'].iloc[i]

            # Read bands
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
                mask_value = src.meta.get('nodata', None)
                transform = src.transform

                xmin = bounds.left
                ymax = bounds.top
                x_res = resolution[0]
                y_res = resolution[1]

                # Build valid/nodata polygons
                mask = nir != mask_value if mask_value is not None else None
                data_polygon = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for _, (s, v) in enumerate(shapes(nir, mask=mask, transform=src.transform))
                )
                data_polygon = gpd.GeoDataFrame.from_features(list(data_polygon), crs=src.crs)

                mask = nir == mask_value if mask_value is not None else None
                no_data_polygon = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for _, (s, v) in enumerate(shapes(nir, mask=mask, transform=src.transform))
                )
                try:
                    no_data_polygon = gpd.GeoDataFrame.from_features(list(no_data_polygon), crs=src.crs)
                except Exception:
                    no_data_polygon = None

            # Buffer/difference
            try:
                if no_data_polygon is not None and len(no_data_polygon) > 0:
                    no_data_union = no_data_polygon.buffer(x_res * 2).unary_union
                else:
                    no_data_union = None
                data_union = data_polygon.unary_union if len(data_polygon) > 0 else None
                data_polygon_final = data_union.difference(no_data_union) if (data_union is not None and no_data_union is not None) else data_union
            except Exception:
                logger.exception("Polygon buffer/difference failed; proceeding with data_union as-is.")
                data_polygon_final = data_polygon.unary_union if len(data_polygon) > 0 else None

            # Compose shoreline file names
            try:
                dt = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S+00:00")
                date_str = dt.strftime("%Y-%m-%d-%H-%M-%S")
            except Exception:
                date_str = str(date).replace(" ", "_").replace(":", "-").replace("+", "_")

            zoo_shoreline_path = os.path.join(zoo_shoreline_dir, f"{date_str}_{satname}_{roi_folder}.geojson")
            nir_shoreline_path = os.path.join(nir_shoreline_dir, f"{date_str}_{satname}_{roi_folder}.geojson")
            swir_shoreline_path = os.path.join(swir_shoreline_dir, f"{date_str}_{satname}_{roi_folder}.geojson")

            # Clip if files exist
            try:
                if os.path.isfile(zoo_shoreline_path):
                    clip_extracted_shoreline(zoo_shoreline_path, data_polygon_final, reference_shoreline, reference_polygon, ref_shore_buffer)
                    logger.info("Clipped zoo shoreline → %s", zoo_shoreline_path)
                if os.path.isfile(nir_shoreline_path):
                    clip_extracted_shoreline(nir_shoreline_path, data_polygon_final, reference_shoreline, reference_polygon, ref_shore_buffer)
                    logger.info("Clipped NIR shoreline → %s", nir_shoreline_path)
                if os.path.isfile(swir_shoreline_path):
                    clip_extracted_shoreline(swir_shoreline_path, data_polygon_final, reference_shoreline, reference_polygon, ref_shore_buffer)
                    logger.info("Clipped SWIR shoreline → %s", swir_shoreline_path)
            except Exception:
                logger.exception("Clipping failed for roi=%s, date=%s", roi_folder, date_str)

            # Mark clipped and persist
            sat_image_list_df.at[i, 'clipped'] = True
            processed += 1
            try:
                tmp_path = sat_image_list_df_path_clipped + ".tmp"
                sat_image_list_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, sat_image_list_df_path_clipped)
            except Exception:
                logger.exception("CSV write failed (atomic) → %s", sat_image_list_df_path_clipped)

        except Exception:
            errors += 1
            logger.exception("Unhandled exception at index=%d", i)

        # GC
        try:
            gc.collect()
        except Exception:
            logger.exception("gc.collect() failed")

    # Final write and summary
    try:
        sat_image_list_df.to_csv(sat_image_list_df_path_clipped, index=False)
    except Exception:
        logger.exception("Final CSV write failed → %s", sat_image_list_df_path_clipped)

    logger.info(
        "Summary: rows=%d, processed=%d, already_done=%d, skipped_low_score=%d, "
        "skipped_missing_image=%d, errors=%d, CSV → %s",
        num_images, processed, already_done, skipped_score, skipped_missing, errors, sat_image_list_df_path_clipped)

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

def filter_timeseries_section_old(g, c, rr, sss, r_home, resample_freq='365D'):
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

def get_timeseries_filtering_logger(section_string: str) -> logging.Logger:
    """
    Logger for timeseries filtering:
      <cwd>/logs/timeseries_filtering/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'timeseries_filtering')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"timeseries_filtering.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def filter_timeseries_section(g, c, rr, sss, r_home, resample_freq='365D'):
    section = 'SSS' + sss
    section_dir = os.path.join(r_home, section)
    section_string = g + c + rr + sss
    transects = os.path.join(section_dir, section_string + '_transects.geojson')

    logger = get_timeseries_filtering_logger(section_string)
    logger.info("Timeseries filtering start: section_dir=%s, resample_freq=%s", section_dir, resample_freq)

    # establish merged files
    timeseries_zoo_merge = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged.csv')
    timeseries_swir_merge = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_swir_thresh.csv')
    timeseries_nir_merge = os.path.join(section_dir, section_string + '_tidally_corrected_transect_time_series_merged_nir_thresh.csv')

    try:
        make_transect_csvs.save_csv_per_id(
            timeseries_zoo_merge,
            timeseries_nir_merge,
            timeseries_swir_merge,
            'ensemble',
            transects,
            section_string,
            resample_freq=resample_freq
        )
        logger.info("Timeseries filtering completed for section=%s", section_string)
    except Exception:
        logger.exception("Timeseries filtering failed for section=%s", section_string)


def get_slopes_section_old(g, c, rr, sss, r_home, arctic_dem, custom_dem):
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


def get_slope_logger(section_string: str) -> logging.Logger:
    """
    Logger for slope computation:
      <cwd>/logs/slope_computation/<section_string>_<YYYYMMDD_HHMMSS>.log
    """
    log_root = os.path.join(os.getcwd(), 'logs', 'slope_computation')
    os.makedirs(log_root, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f"{section_string}_{ts}.log")

    logger_name = f"slope_computation.{section_string}.{ts}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_slopes_section(g, c, rr, sss, r_home, arctic_dem, custom_dem):
    """
    Wrapper for profile_shoreline_section with logging.
    Logs to: <cwd>/logs/slope_computation/<section_string>_YYYYMMDD_HHMMSS.log
    """

    import os
    import glob
    import datetime
    import logging
    import geopandas as gpd
    import pandas as pd
    from osgeo import gdal

    section = 'SSS' + sss
    section_dir = os.path.join(r_home, section)
    section_string = g + c + rr + sss

    # -------------------------
    # Initialize logger
    # -------------------------
    logger = get_slope_logger(section_string)
    logger.info("Starting slope computation for section %s", section_string)
    logger.info("Section directory: %s", section_dir)
    logger.info("Arctic DEM: %s | Custom DEM: %s", arctic_dem, custom_dem)

    try:
        logger.info("Calling profile_shoreline_section()")
        dem_to_beach_slope.profile_shoreline_section(
            g,
            c,
            rr,
            '0',
            r_home,
            all_sections=False,
            custom_sections=['SSS' + sss],
            arctic_dem=arctic_dem,
            custom_dem=custom_dem,
            vertical_datum=''
        )
        logger.info("Completed profile_shoreline_section() successfully for %s", section_string)

    except Exception as e:
        logger.exception("Error in profile_shoreline_section for %s", section_string)

    logger.info("Finished slope computation wrapper for %s", section_string)

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


#loading config and args
cfg = args.config

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
        


