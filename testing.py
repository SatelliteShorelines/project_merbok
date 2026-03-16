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

from transformers import TFSegformerForSemanticSegmentation, SegformerImageProcessor
import tensorflow as tf
import numpy as np
from PIL import Image
from osgeo import gdal
import os

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