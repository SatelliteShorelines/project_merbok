"""
This script contains the functionality to compute annual or decadal composites from Alaska satellite imagery 
and compute land masks and shorelines via the composites.
"""

###basic utilities
import os
import shutil
from datetime import datetime

###numerical and data 
import numpy as np
import pandas as pd
import geopandas as gpd

###counting
from collections import Counter

##gdal and rasterio routines
from osgeo import gdal
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.mask import mask
from rasterio.windows import Window

###shapely routines
from shapely.geometry import box
from shapely.ops import linemerge

##progress bar
from tqdm import tqdm

###exception tracking
import traceback

###skimage routines
from skimage.filters import threshold_otsu, threshold_multiotsu, threshold_local, try_all_threshold
from scipy.ndimage import median_filter

##common routines
from common.common import *

##command line args
import argparse

gdal.UseExceptions()
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--global_region", type=str, help='global region, ex: 1 for USA')
parser.add_argument("-c", "--coastal_area", type=str, help='coastal area, ex: 4 for Alaska')
parser.add_argument("-rr", "--subregion", type=str, help='subregion')
parser.add_argument("-sss", "--shoreline_section", type=str, required=False, default = '', help='shoreline section')
parser.add_argument("-i", "--interval", type=str, required=True, default = '', help='composite interval, decadal or annual')
parser.add_argument("-d", "--directory_name", type=str, required=True, help='name of directory to save in')
parser.add_argument("-is", "--image_score", type=str, required=False, default='0.335', help='image suitability score')
parser.add_argument("-t", "--time", type=str, required=False, default = '', help='specific year or decade, e.g. 2015 for year 2015, 2010 for 2010s')


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def min_max_normalize(arr):
    """
    Normalizes array to minimum and maximum values

    inputs:
    arr (np.ndarray): input numpy array
    
    outputs:
    new_arr (np.ndarray): the output array
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    new_arr = (arr - min_val) / (max_val - min_val)
    return new_arr

def compress_and_remove_directory(source_directory):
    """
    Compresses a directory into a .tar.gz archive and then removes the original directory.

    Args:
        source_directory (str): The path to the directory to compress.
        output_filename (str): The desired name for the .tar.gz archive.
    """
    output_filename = source_directory+'.tar.gz'
    if os.path.isfile(output_filename)==False:
        try:
            # Create the .tar.gz archive
            with tarfile.open(output_filename, "w:gz") as tar:
                tar.add(source_directory, arcname=os.path.basename(source_directory))
            print(f"Successfully created {output_filename}")

            # Remove the original directory
            shutil.rmtree(source_directory)
            print(f"Successfully removed original directory: {source_directory}")

        except Exception as e:
            print(f"An error occurred: {e}")
            pass
    else:
        print('already archived')

def segment(array_to_seg, model_list):
    print('segmenting')
    ##segment image
    seg_lab, color_seg_lab = do_seg_array(array_to_seg,
                        model_list,
                        )

    ##simplify segmented image
    seg_lab[seg_lab==1] = 0
    seg_lab[seg_lab>0] = 1
    return seg_lab
                

def segment_vrt_sections_old_working(input_vrt, output_dir, tile_size=500, overlap_percent=10):
    """
    Reads a VRT file in 500x500 sections with overlap,
    applies segmentation, and saves GeoTIFFs with NaNs for invalid pixels.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model_str = "ak_segformer_RGB_4class_14037041"
    print('using AK model')

    settings = {
        "sample_direc": None,
        "use_GPU": "0",
        "implementation": "BEST",
        "model_type": model_str,
        "otsu": False,
        "tta": False,
        "use_local_model": False,
        "local_model_path": None,
        "img_type": "RGB"
    }

    zoo_model_instance = zoo_model.Zoo_Model()
    zoo_model_instance.set_settings(**settings)
    zoo_model_instance.prepare_model('BEST', model_str)
    model_list = zoo_model_instance.model_list

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    in_ds = gdal.Open(input_vrt, gdal.GA_ReadOnly)
    if not in_ds:
        raise Exception(f"Unable to open {input_vrt}")

    in_band_blue = in_ds.GetRasterBand(1)
    in_band_green = in_ds.GetRasterBand(2)
    in_band_red = in_ds.GetRasterBand(3)

    x_size = in_ds.RasterXSize
    y_size = in_ds.RasterYSize
    geotransform = in_ds.GetGeoTransform()
    projection = in_ds.GetProjection()

    overlap = int(tile_size * (overlap_percent / 100.0))
    step = tile_size - overlap
    driver = gdal.GetDriverByName("GTiff")
    output_files = []

    for y_offset in range(0, y_size, step):
        for x_offset in range(0, x_size, step):
            y_read_size = min(tile_size, y_size - y_offset)
            x_read_size = min(tile_size, x_size - x_offset)

            blue = in_band_blue.ReadAsArray(x_offset, y_offset, x_read_size, y_read_size)
            green = in_band_green.ReadAsArray(x_offset, y_offset, x_read_size, y_read_size)
            red = in_band_red.ReadAsArray(x_offset, y_offset, x_read_size, y_read_size)
            rgb = np.dstack((red, green, blue))

            print(f"Reading section at offset ({x_offset}, {y_offset}) with size ({x_read_size}, {y_read_size})")


            # Detect NaNs and black pixels
            no_data_blue = in_band_blue.GetNoDataValue()
            no_data_green = in_band_green.GetNoDataValue()
            no_data_red = in_band_red.GetNoDataValue()

            nan_mask = (
                (blue == no_data_blue) |
                (green == no_data_green) |
                (red == no_data_red)
            )
            black_mask = np.all(rgb == [0, 0, 0], axis=-1)
            invalid_mask = nan_mask | black_mask
            rgb[invalid_mask] = [0,0,0]
            rgb = scale_percentile(rgb, lower_percentile=5, upper_percentile=95)

            max_size = max(x_read_size, y_read_size)
            padded_rgb = np.zeros((max_size, max_size, 3), dtype=rgb.dtype)
            padded_rgb[:y_read_size, :x_read_size] = rgb
            rgb = padded_rgb

            # Segment
            seg_array = segment(rgb, model_list)
            seg_array = seg_array[:y_read_size, :x_read_size]
            # Apply invalid mask only to the valid region
            seg_array[invalid_mask] = 0


            # Georeference
            x_origin = geotransform[0] + x_offset * geotransform[1]
            y_origin = geotransform[3] + y_offset * geotransform[5]
            out_geotransform = (
                x_origin, geotransform[1], geotransform[2],
                y_origin, geotransform[4], geotransform[5]
            )

            output_filename = os.path.join(output_dir, f"section_{x_offset}_{y_offset}.tif")
            out_ds = driver.Create(output_filename, x_read_size, y_read_size, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform(out_geotransform)
            out_ds.SetProjection(projection)
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(seg_array)
            out_band.SetNoDataValue(np.nan)
            out_ds.FlushCache()
            out_ds = None
            output_files.append(output_filename)

    return output_files, output_dir

def segment_vrt_sections(input_vrt, output_dir, tile_size=500, overlap_percent=10, coverage_threshold=0.1):
    """
    Dynamically tiles a VRT file in square sections with overlap,
    applies segmentation only where valid data exists,
    and saves GeoTIFFs with NaNs for invalid pixels.
    """
    from segmentation.simple_segmentation import do_seg_array, binary_lab_to_color_lab
    from coastseg import coastseg_map
    from coastseg import zoo_model
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model_str = "ak_segformer_RGB_4class_14037041"
    settings = {
        "sample_direc": None,
        "use_GPU": "0",
        "implementation": "BEST",
        "model_type": model_str,
        "otsu": False,
        "tta": False,
        "use_local_model": False,
        "local_model_path": None,
        "img_type": "RGB"
    }

    zoo_model_instance = zoo_model.Zoo_Model()
    zoo_model_instance.set_settings(**settings)
    zoo_model_instance.prepare_model('BEST', model_str)
    model_list = zoo_model_instance.model_list

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    in_ds = gdal.Open(input_vrt, gdal.GA_ReadOnly)
    if not in_ds:
        raise Exception(f"Unable to open {input_vrt}")

    in_band_blue = in_ds.GetRasterBand(1)
    in_band_green = in_ds.GetRasterBand(2)
    in_band_red = in_ds.GetRasterBand(3)

    x_size = in_ds.RasterXSize
    y_size = in_ds.RasterYSize
    geotransform = in_ds.GetGeoTransform()
    projection = in_ds.GetProjection()

    # Adjust tile size to fit raster
    tile_size = min(tile_size, x_size, y_size)
    overlap = int(tile_size * (overlap_percent / 100.0))
    step = tile_size - overlap
    driver = gdal.GetDriverByName("GTiff")
    output_files = []

    print(f"Raster size: {x_size} x {y_size}")
    print(f"Adjusted tile size: {tile_size}, Step size: {step}")

    # Build a valid data mask
    red = in_band_red.ReadAsArray()
    green = in_band_green.ReadAsArray()
    blue = in_band_blue.ReadAsArray()

    no_data_red = in_band_red.GetNoDataValue()
    no_data_green = in_band_green.GetNoDataValue()
    no_data_blue = in_band_blue.GetNoDataValue()

    no_data_red = no_data_red if no_data_red is not None else -9999
    no_data_green = no_data_green if no_data_green is not None else -9999
    no_data_blue = no_data_blue if no_data_blue is not None else -9999

    nan_mask = (
        (red == no_data_red) |
        (green == no_data_green) |
        (blue == no_data_blue)
    )
    black_mask = np.all(np.dstack((red, green, blue)) == [0, 0, 0], axis=-1)
    valid_mask = ~(nan_mask | black_mask)

    print(f"Valid pixel count: {np.sum(valid_mask)}")

    for y_offset in range(0, y_size - tile_size + 1, step):
        for x_offset in range(0, x_size - tile_size + 1, step):
            print(f"Checking tile at ({x_offset}, {y_offset})")

            window_mask = valid_mask[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size]
            coverage = np.sum(window_mask) / (tile_size * tile_size)
            print(f"Coverage: {coverage:.2f}")

            if coverage < coverage_threshold:
                print("Coverage too low — skipping tile.")
                continue

            tile_red = red[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size]
            tile_green = green[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size]
            tile_blue = blue[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size]
            rgb = np.dstack((tile_red, tile_green, tile_blue))

            tile_nan_mask = (
                (tile_red == no_data_red) |
                (tile_green == no_data_green) |
                (tile_blue == no_data_blue)
            )
            tile_black_mask = np.all(rgb == [0, 0, 0], axis=-1)
            tile_invalid_mask = tile_nan_mask | tile_black_mask
            rgb[tile_invalid_mask] = [0, 0, 0]

            rgb = scale_percentile(rgb, lower_percentile=2, upper_percentile=98)
            seg_array = segment(rgb, model_list)
            seg_array = resize_image_numpy(seg_array, (tile_size, tile_size))
            seg_array[tile_invalid_mask] = 0

            x_origin = geotransform[0] + x_offset * geotransform[1]
            y_origin = geotransform[3] + y_offset * geotransform[5]
            out_geotransform = (
                x_origin, geotransform[1], geotransform[2],
                y_origin, geotransform[4], geotransform[5]
            )

            output_filename = os.path.join(output_dir, f"section_{x_offset}_{y_offset}.tif")
            out_ds = driver.Create(output_filename, tile_size, tile_size, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform(out_geotransform)
            out_ds.SetProjection(projection)
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(seg_array)
            out_band.SetNoDataValue(np.nan)
            out_ds.FlushCache()
            out_ds = None
            output_files.append(output_filename)

    print(f"Total tiles saved: {len(output_files)}")
    return output_files, output_dir


def create_final_vrt(input_files, output_vrt, output_tif, output_dir):
    """
    Creates a new VRT file that mosaics all the intermediate GeoTIFF files.
    """
    vrt_options = gdal.BuildVRTOptions(resampleAlg='mode', srcNodata=0)
    gdal.BuildVRT(output_vrt, input_files, options=vrt_options)
    ds = gdal.Open(output_vrt)
    translate_options = gdal.TranslateOptions(format='COG', creationOptions=['COMPRESS=LZW'])
    gdal.Translate(output_tif, ds, options=translate_options)

    shutil.rmtree(output_dir)
    print(f"Created final VRT: {output_vrt}")         

def mode_composite_to_cog(
    input_rasters,
    cog_output_path,
    compress='DEFLATE',
    blocksize=512,
    verbose=True):
    """
    Computes a pixel-wise mode across input rasters using strict multimode rules.
    Writes result as a Cloud Optimized GeoTIFF (COG).
    """

    # Open reference raster
    ref_ds = gdal.Open(input_rasters[0])
    if ref_ds is None:
        raise RuntimeError(f"Failed to open reference raster: {input_rasters[0]}")
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    xsize = ref_ds.RasterXSize
    ysize = ref_ds.RasterYSize
    xres = ref_gt[1]
    yres = abs(ref_gt[5])
    nodata = ref_ds.GetRasterBand(1).GetNoDataValue()
    if nodata is None:
        nodata = -9999.0

    # Warp all rasters to match reference
    arrays = []
    for path in input_rasters:
        ds = gdal.Open(path)
        warped = gdal.Warp(
            '', ds, format='MEM',
            outputBounds=(ref_gt[0], ref_gt[3] + ysize * ref_gt[5],
                          ref_gt[0] + xsize * xres, ref_gt[3]),
            xRes=xres, yRes=yres,
            dstSRS=ref_proj,
            resampleAlg='nearest',
            srcNodata=nodata,
            dstNodata=nodata
        )
        arr = warped.GetRasterBand(1).ReadAsArray().astype(np.float32)
        arr[arr == nodata] = np.nan
        arrays.append(arr)
        if verbose:
            print(f"Warped and loaded: {path} → shape {arr.shape}")

    # Stack and apply custom mode logic
    stack = np.stack(arrays, axis=0)

    def pixel_mode(values):
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        vals, counts = np.unique(valid, return_counts=True)
        max_count = np.max(counts)
        if max_count < 2:
            return np.nan  # no value appears more than once
        if np.sum(counts == max_count) > 1:
            return np.nan  # multiple values tied for mode
        return vals[np.argmax(counts)]


    mode_result = np.apply_along_axis(pixel_mode, 0, stack)

    # Replace NaNs with nodata
    mode_result = np.where(np.isnan(mode_result), nodata, mode_result)

    # Write to in-memory dataset
    mem_ds = gdal.GetDriverByName('MEM').Create('', xsize, ysize, 1, gdal.GDT_Float32)
    mem_ds.SetGeoTransform(ref_gt)
    mem_ds.SetProjection(ref_proj)
    mem_ds.GetRasterBand(1).WriteArray(mode_result)
    mem_ds.GetRasterBand(1).SetNoDataValue(nodata)

    # Translate to COG
    translate_options = gdal.TranslateOptions(
        format='COG',
        creationOptions=[
            f'COMPRESS={compress}',
            'TILED=YES',
            f'BLOCKSIZE={blocksize}',
            'OVERVIEWS=IGNORE_EXISTING',
            'NUM_THREADS=ALL_CPUS'
        ],
        noData=nodata
    )
    gdal.Translate(cog_output_path, mem_ds, options=translate_options)
    if verbose:
        print(f"Mode composite saved to: {cog_output_path}")


def average_mosaic_to_cog(
    tiff_list,
    cog_output_path,
    compress='DEFLATE',
    blocksize=512,
    clip_output=True,
    verbose=False):
    

    # Collect extents and resolutions
    extents = []
    x_res_list = []
    y_res_list = []
    for path in tiff_list:
        ds = gdal.Open(path)
        gt = ds.GetGeoTransform()
        x_res_list.append(gt[1])
        y_res_list.append(abs(gt[5]))
        x_min = gt[0]
        y_max = gt[3]
        x_max = x_min + ds.RasterXSize * gt[1]
        y_min = y_max + ds.RasterYSize * gt[5]
        extents.append((x_min, x_max, y_min, y_max))

    # Union extent
    x_min = min(e[0] for e in extents)
    x_max = max(e[1] for e in extents)
    y_min = min(e[2] for e in extents)
    y_max = max(e[3] for e in extents)

    # Use highest resolution
    x_res = min(x_res_list)
    y_res = min(y_res_list)

    # Reference projection
    ref_ds = gdal.Open(tiff_list[0])
    ref_proj = ref_ds.GetProjection()
    ref_dtype = gdal.GDT_Float32

    # Warp first raster to get output shape
    first_warp = gdal.Warp(
        '', ref_ds, format='MEM',
        dstSRS=ref_proj,
        outputBounds=(x_min, y_min, x_max, y_max),
        xRes=x_res, yRes=y_res,
        resampleAlg='nearest'
    )
    y_size = first_warp.RasterYSize
    x_size = first_warp.RasterXSize

    # Accumulators
    sum_array = np.zeros((5, y_size, x_size), dtype=np.float32)
    count_array = np.zeros((5, y_size, x_size), dtype=np.uint16)
    nodata_values = []

    # Warp and accumulate with per-image normalization
    for path in tqdm(tiff_list, desc="Building mosaic average"):
        ds = gdal.Open(path)
        warp_ds = gdal.Warp(
            '', ds, format='MEM',
            dstSRS=ref_proj,
            outputBounds=(x_min, y_min, x_max, y_max),
            xRes=x_res, yRes=y_res,
            resampleAlg='nearest'
        )
        for b in range(5):
            band = warp_ds.GetRasterBand(b + 1)
            arr = band.ReadAsArray().astype(np.float32)
            nodata = band.GetNoDataValue()
            if nodata is None:
                nodata = np.nan
            nodata_values.append(nodata)

            # Mask valid pixels
            mask = (~np.isnan(arr)) & (~np.isclose(arr, nodata))
            valid = arr[mask]

            # Normalize RGB bands using percentiles
            if b < 3 and np.any(valid):
                p2, p98 = np.percentile(valid, [2, 98])
                if verbose:
                    print(f"Band {b+1} RGB stretch: p2={p2}, p98={p98}")
                if not np.isclose(p98, p2):
                    arr[mask] = (valid - p2) / (p98 - p2)

            # Normalize NIR/SWIR bands using min-max
            elif b >= 3 and np.any(valid):
                min_val, max_val = np.min(valid), np.max(valid)
                if verbose:
                    print(f"Band {b+1} NIR/SWIR normalize: min={min_val}, max={max_val}")
                if not np.isclose(max_val, min_val):
                    arr[mask] = (valid - min_val) / (max_val - min_val)

            # Clip to [0, 1] before accumulation
            arr[mask] = np.clip(arr[mask], 0, 1)

            sum_array[b][mask] += arr[mask]
            count_array[b][mask] += 1

    # Determine most common nodata value
    nodata_counter = Counter(nodata_values)
    ref_nodata = nodata_counter.most_common(1)[0][0]
    if np.isnan(ref_nodata):
        ref_nodata = -9999.0

    # Final average
    avg_array = np.where(count_array > 0, sum_array / count_array, ref_nodata)
    avg_array[np.isnan(avg_array)] = ref_nodata

    # Optional final clipping
    if clip_output:
        for b in range(5):
            mask = avg_array[b] != ref_nodata
            avg_array[b][mask] = np.clip(avg_array[b][mask], 0, 1)

    # Write to in-memory dataset
    mem_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 5, ref_dtype)
    mem_ds.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    mem_ds.SetProjection(ref_proj)
    for b in range(5):
        band = mem_ds.GetRasterBand(b + 1)
        band.WriteArray(avg_array[b])
        band.SetNoDataValue(ref_nodata)

    mem_ds.SetMetadataItem('NODATA_VALUES', ','.join([str(ref_nodata)] * 5))

    # Translate to COG
    translate_options = gdal.TranslateOptions(
        format='COG',
        creationOptions=[
            f'COMPRESS={compress}',
            'TILED=YES',
            f'BLOCKSIZE={blocksize}',
            'OVERVIEWS=IGNORE_EXISTING',
            'NUM_THREADS=ALL_CPUS'
        ],
        noData=ref_nodata
    )
    gdal.Translate(cog_output_path, mem_ds, options=translate_options)
    print(f"Final COG saved to: {cog_output_path}")

def compute_local_threshold(input_path, metric='NIR'):

    # Load the VRT raster
    with rasterio.open(input_path) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        nir = src.read(4)
        swir = src.read(5)
        nodata = src.nodata
        profile = src.profile.copy()

    # Select band based on metric
    if metric == 'NIR':
        band = nir
    elif metric == 'SWIR':
        band = swir
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Clean band
    band = np.nan_to_num(band, nan=0, posinf=0, neginf=0)
    band = median_filter(band, size=5)
    # Clip outliers using percentiles
    lower = np.percentile(band, 2)
    upper = np.percentile(band, 98)
    if upper == lower:
        band = np.zeros_like(band)
    else:
        band = np.clip(band, lower, upper)
        band = (band - lower) / (upper - lower)

    # Mask nodata
    if nodata is not None:
        band = np.ma.masked_equal(band, nodata)

    thresh = threshold_multiotsu(band)
    print(thresh)
    binary_mask = band > max(thresh)

    # Convert to uint8
    mask_uint8 = binary_mask.astype('uint8')

    return mask_uint8, profile

def compute_otsu_threshold(input_path, metric='NIR'):

    # Load the VRT raster
    with rasterio.open(input_path) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        nir = src.read(4)
        swir = src.read(5)
        nodata = src.nodata
        profile = src.profile.copy()

    blue[blue==nodata] = np.nan
    red[red==nodata] = np.nan
    green[green==nodata] = np.nan
    nir[nir==nodata] = np.nan
    swir[swir==nodata] = np.nan

    # Select band based on metric
    if metric == 'NIR':
        band = nir
    elif metric == 'SWIR':
        band = swir
    elif metric == 'NDWI':
        band = (green-nir)/(green+nir)
        band = np.nan_to_num(band, nan=0, posinf=0, neginf=0)
        thresh = threshold_multiotsu(band)
        binary_mask = band < max(thresh)
    elif metric == 'MNDWI':
        band = (green-swir)/(green+swir)
        band = np.nan_to_num(band, nan=0, posinf=0, neginf=0)
        thresh = threshold_multiotsu(band)
        binary_mask = band < max(thresh)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Clean band
    band = np.nan_to_num(band, nan=0, posinf=0, neginf=0)

    # Clip outliers using percentiles
    lower = np.percentile(band, 2)
    upper = np.percentile(band, 98)
    if upper == lower:
        band = np.zeros_like(band)
    else:
        band = np.clip(band, lower, upper)
        band = (band - lower) / (upper - lower)

    # Mask nodata
    if metric == 'NIR' or metric == 'SWIR':
        if nodata is not None:
            band = np.ma.masked_equal(band, nodata)

            # Flatten for thresholding
            flattened = band.compressed() if np.ma.isMaskedArray(band) else band.flatten()

            otsu_thresh = threshold_otsu(flattened)
            otsu_threshes = threshold_multiotsu(band,4)
            binary_mask = band > otsu_thresh

    # Convert to uint8
    mask_uint8 = binary_mask.astype('uint8')

    return mask_uint8, profile

def composite_analysis_section(g, c, rr, sss, interval, alaska_vrts_folder, r_home, image_score, time=''):
    section='SSS'+sss
    section_str=g+c+rr+sss
    print(section_str)

    vrt_section_dir = os.path.join(alaska_vrts_folder, 'G'+g, 'C'+c,'RR'+rr,'SSS'+sss)
    if interval!='planet':
        csv_path = os.path.join(r_home, section, section_str + '_ms_lists', 'landsat_sentinel_ms_paths_scored_update.csv')
    else:
        csv_path = os.path.join(r_home, section, section_str + '_ms_lists', 'planet_ms_paths_scored_update.csv')
    reference_polygon = os.path.join(r_home, section, section_str+'_rois.geojson')
    reference_polygon = os.path.join(r_home, section, section_str+'_spatial_kde_otsu.geojson')
    final_df = pd.read_csv(csv_path)

    ###here we making a datetime column to use and also discard super noisy images 
    final_df['datetimes_utc'] = pd.to_datetime(final_df['datetimes_utc'])
    final_df = final_df[final_df['model_scores']>image_score].reset_index(drop=True)

    if interval=='decadal':
        final_df['interval_start'] = (final_df['datetimes_utc'].dt.year // 10) * 10
    elif interval=='annual':
        final_df['interval_start'] = final_df['datetimes_utc'].dt.year
    elif interval=='planet':
        final_df = final_df[final_df['satnames']=='PS'].reset_index(drop=True)
    
    try:
        os.mkdir(vrt_section_dir)
    except:
        pass

    ###if we already ran this site and want to re-run
    if os.path.isfile(vrt_section_dir+'.tar.gz')==True:
        unpack_tar_gz(vrt_section_dir+'.tar.gz', extract_dir=os.path.join(alaska_vrts_folder, 'G'+g, 'C'+c,'RR'+rr))
    
    ###filter data to specified time
    if time!='':
        time = int(time)
        final_df = final_df[final_df['interval_start']==time].reset_index(drop=True)

    ###here is the loop to go over each time interval
    shoreline_files = []
    land_masks = []
    if interval!='planet':
        for date in sorted(np.unique(final_df['interval_start'])):
            try:
                clip_tif = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_mosaic.tif')
                df_filter = final_df[final_df['interval_start']==date].reset_index(drop=True)
                df_filter = df_filter[df_filter['satnames']!='PS']
                rasters = list(df_filter['ms_tiff_path'])
                alaska_date_sat_vrt = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'.vrt')
                temp_otsu_path_nir = os.path.join(vrt_section_dir, 'temp_otsu_nir.tif')
                temp_otsu_path_swir = os.path.join(vrt_section_dir, 'temp_otsu_swir.tif')
                temp_otsu_path_ndwi = os.path.join(vrt_section_dir, 'temp_otsu_ndwi.tif')
                temp_otsu_path_mndwi = os.path.join(vrt_section_dir, 'temp_otsu_mndwi.tif')
                otsu_path_nir = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_otsu_nir.tif')
                otsu_path_swir = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_otsu_swir.tif')
                otsu_path_ndwi = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_otsu_ndwi.tif')
                otsu_path_mndwi = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_otsu_mndwi.tif')

                date = str(date)

                ####here we are making the mosaicked and composite b,g,r,nir,swir image
                output_tif = os.path.join(vrt_section_dir, 'temp.tif')
                clip_tif = os.path.join(vrt_section_dir, section_str+'_'+date+'_mosaic.tif')
                if os.path.isfile(clip_tif)==False:
                    try:
                        average_mosaic_to_cog(rasters, output_tif, compress='DEFLATE', blocksize=512)
                    except Exception as e:
                        traceback.print_exc()
                        print('error mosaicking')
                        continue

                    try:
                        # Open the input raster
                        with rasterio.open(output_tif) as src:
                            # Clip the raster
                            crs = src.crs
                            
                            gdf = gpd.read_file(reference_polygon)
                            gdf = gdf.to_crs(crs)
                            gdf['geometry'] = gdf.buffer(1200)
                            minx, miny, maxx, maxy = gdf.total_bounds
                            clipping_geometry = [box(minx, miny, maxx, maxy)]
                            out_image, out_transform = mask(src, clipping_geometry, crop=True)
                            # Update metadata for the output raster
                            out_meta = src.meta.copy()
                            out_meta.update({
                                "driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform
                            })

                            # Write the clipped raster to a new file
                            with rasterio.open(clip_tif, "w", **out_meta) as dest:
                                dest.write(out_image)
                    except Exception as e:
                        traceback.print_exc()
                        print('error clipping')
                        continue
                
                ###remove the otsu thresholds if they already exist
                if os.path.isfile(otsu_path_nir)==True and os.path.isfile(otsu_path_swir==True):
                    os.remove(otsu_path_nir)
                    os.remove(otsu_path_swir)
                if os.path.isfile(otsu_path_mndwi)==True and os.path.isfile(otsu_path_ndwi==True):
                    os.remove(otsu_path_mndwi)
                    os.remove(otsu_path_ndwi)


                ###here we are computing the otsu thresholds
                nir_thresh,profile = compute_otsu_threshold(clip_tif, metric='NIR')
                swir_thresh,profile = compute_otsu_threshold(clip_tif, metric='SWIR')

                ####here we are making the profile for the otsu thresholds
                profile.update({
                    'driver':'GTiff',
                    'dtype': 'uint8',
                    'count': 1,
                    'nodata': 0,
                    'compress': 'lzw'  # Optional: compress the output
                })

                ###here we are writing the otsu thresholds
                with rasterio.open(temp_otsu_path_nir, 'w', **profile) as dst:
                    dst.write(nir_thresh, 1)
                with rasterio.open(temp_otsu_path_swir, 'w', **profile) as dst:
                    dst.write(swir_thresh, 1)


                ###########this runs deep learning model, there are issues with segment_vrt_sections that lead to unreliable model outputs
                # output_dir = os.path.join(vrt_section_dir, 'seg')
                # output_files, output_dir = segment_vrt_sections(clip_tif, output_dir, tile_size=1000, overlap_percent=50)
                # output_tif = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_seg.tif')
                # output_vrt = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_seg.vrt')
                # create_final_vrt(output_files, output_vrt, output_tif, output_dir)


                ###here we are clipping the otsu thresholds to the buffered shoreline change envelope
                try:
                    with rasterio.open(temp_otsu_path_nir) as src:
                        # Clip the raster
                        crs = src.crs
                        gdf = gpd.read_file(reference_polygon)
                        gdf = gdf.to_crs(crs)
                        gdf['geometry'] = gdf.buffer(1000)
                        minx, miny, maxx, maxy = gdf.total_bounds
                        clipping_geometry = [box(minx, miny, maxx, maxy)]
                        out_image, out_transform = mask(src, clipping_geometry, crop=True)
                        # Update metadata for the output raster
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })

                        # Write the clipped raster to a new file
                        with rasterio.open(otsu_path_nir, "w", **out_meta) as dest:
                            dest.write(out_image)
                except Exception as e:
                    traceback.print_exc()
                    continue
                    print('error clipping otsu')
                try:
                    with rasterio.open(temp_otsu_path_swir) as src:
                        # Clip the raster
                        crs = src.crs
                        gdf = gpd.read_file(reference_polygon)
                        gdf = gdf.to_crs(crs)
                        gdf['geometry'] = gdf.buffer(1000)
                        minx, miny, maxx, maxy = gdf.total_bounds
                        clipping_geometry = [box(minx, miny, maxx, maxy)]
                        out_image, out_transform = mask(src, clipping_geometry, crop=True)
                        # Update metadata for the output raster
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })

                        # Write the clipped raster to a new file
                        with rasterio.open(otsu_path_swir, "w", **out_meta) as dest:
                            dest.write(out_image)
                except Exception as e:
                    traceback.print_exc()
                    continue
                    print('error clipping otsu')
                # try:
                #     with rasterio.open(temp_otsu_path_ndwi) as src:
                #         # Clip the raster
                #         crs = src.crs
                #         gdf = gpd.read_file(reference_polygon)
                #         gdf = gdf.to_crs(crs)
                #         gdf['geometry'] = gdf.buffer(1000)
                #         minx, miny, maxx, maxy = gdf.total_bounds
                #         clipping_geometry = [box(minx, miny, maxx, maxy)]
                #         out_image, out_transform = mask(src, clipping_geometry, crop=True)
                #         # Update metadata for the output raster
                #         out_meta = src.meta.copy()
                #         out_meta.update({
                #             "driver": "GTiff",
                #             "height": out_image.shape[1],
                #             "width": out_image.shape[2],
                #             "transform": out_transform
                #         })

                #         # Write the clipped raster to a new file
                #         with rasterio.open(otsu_path_ndwi, "w", **out_meta) as dest:
                #             dest.write(out_image)
                # except Exception as e:
                #     traceback.print_exc()
                #     continue
                #     print('error clipping otsu')
                # try:
                #     with rasterio.open(temp_otsu_path_mndwi) as src:
                #         # Clip the raster
                #         crs = src.crs
                #         gdf = gpd.read_file(reference_polygon)
                #         gdf = gdf.to_crs(crs)
                #         gdf['geometry'] = gdf.buffer(1000)
                #         minx, miny, maxx, maxy = gdf.total_bounds
                #         clipping_geometry = [box(minx, miny, maxx, maxy)]
                #         out_image, out_transform = mask(src, clipping_geometry, crop=True)
                #         # Update metadata for the output raster
                #         out_meta = src.meta.copy()
                #         out_meta.update({
                #             "driver": "GTiff",
                #             "height": out_image.shape[1],
                #             "width": out_image.shape[2],
                #             "transform": out_transform
                #         })

                #         # Write the clipped raster to a new file
                #         with rasterio.open(otsu_path_mndwi, "w", **out_meta) as dest:
                #             dest.write(out_image)
                # except Exception as e:
                #     traceback.print_exc()
                #     continue
                #     print('error clipping otsu')

                ##this computes average land mask, if deep learning output is to be included then add in output_tif to the rasters list
                rasters = [otsu_path_nir, otsu_path_swir]#, otsu_path_ndwi, otsu_path_mndwi]#, output_tif]
                output_tif_path = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_avg_land_mask.tif')
                reference_polygon_path = os.path.join(r_home, section, section_str+'_reference_polygon.geojson')
                polygon_path = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_avg_land_mask.geojson')
                if os.path.isfile(output_tif_path)==True:
                    os.remove(output_tif_path)   
                if os.path.isfile(polygon_path)==True:
                    os.remove(polygon_path) 
                try:
                    mode_composite_to_cog(rasters,output_tif_path,compress='DEFLATE',blocksize=512,verbose=True)
                except Exception as e:
                    traceback.print_exc()
                    continue
                    pass
                try:
                    os.remove(os.path.join(vrt_section_dir, 'temp_otsu_nir.tif'))
                except:
                    pass
                try:
                    os.remove(os.path.join(vrt_section_dir, 'temp_otsu_swir.tif'))
                except:
                    pass
                try:
                    os.remove(os.path.join(vrt_section_dir, 'temp.tif'))
                except:
                    pass
                with rasterio.open(clip_tif) as src:
                    x_res, y_res = src.res
                    RES = max(x_res,y_res)


                ###polygonzing the land mask
                cmd = 'python gdal_polygonize.py ' + output_tif_path +' -b 1 -f GeoJSON ' + polygon_path + ' land_mask'
                land_masks.append(polygon_path)
                os.system(cmd)


                ###here we are adding decade as a field to the land mask
                reference_polygon_gdf = gpd.read_file(reference_polygon_path)
                reference_polygon_gdf = wgs84_to_utm_df(reference_polygon_gdf)
                reference_polygon_gdf = reference_polygon_gdf.buffer(100)
                reference_polygon_gdf = utm_to_wgs84_df(reference_polygon_gdf)
                land_mask_gdf = gpd.read_file(polygon_path)
                land_mask_gdf = land_mask_gdf.to_crs(reference_polygon_gdf.crs)
                if interval=='decadal':
                    land_mask_gdf['decade'] = [int(date)]*len(land_mask_gdf)
                elif interval=='annual':
                    land_mask_gdf['year'] = [int(date)]*len(land_mask_gdf)
                land_mask_gdf.to_file(polygon_path)

                ##here we we are making some shorelines from the land mask, smoothing, and adding decade as a field
                land_mask_lines_gdf = land_mask_gdf.boundary
                land_mask_lines_gdf_clip = gpd.clip(land_mask_lines_gdf, reference_polygon_gdf)
                multilines = [geom for geom in land_mask_lines_gdf_clip if geom.geom_type == 'MultiLineString']
                singlelines = [geom for geom in land_mask_lines_gdf_clip if geom.geom_type == 'LineString']
                lines =convert_multilinestring_list_to_linestring_list(multilines)
                lines = lines+singlelines
                if interval=='decadal':
                    lines_gdf  = {'decade':[int(date)]*len(lines),
                                    'geometry': lines
                                    }
                elif interval=='annual':
                    lines_gdf  = {'year':[int(date)]*len(lines),
                                'geometry': lines
                                    }      
                lines_gdf = gpd.GeoDataFrame(lines_gdf, crs="EPSG:4326")
                shoreline_file = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_avg_land_mask_shoreline.geojson')
                if os.path.isfile(shoreline_file)==True:
                    os.remove(shoreline_file)
                if len(lines_gdf)>0:
                    lines_gdf = smooth_lines_constant(lines_gdf,RES,refinements=2)
                    lines_gdf.to_file(shoreline_file)
                    shoreline_files.append(shoreline_file)
            except:
                continue
    else:
        shoreline_files=[]
        land_masks=[]
        date = 'planet'
        clip_tif = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_mosaic.tif')
        df_filter = final_df
        rasters = list(df_filter['ms_tiff_path'])
        alaska_date_sat_vrt = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'.vrt')
        temp_otsu_path_nir = os.path.join(vrt_section_dir, 'temp_otsu_nir.tif')
        temp_otsu_path_swir = os.path.join(vrt_section_dir, 'temp_otsu_swir.tif')
        otsu_path_nir = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_otsu_nir.tif')
        otsu_path_swir = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_otsu_swir.tif')

        

        ####here we are making the mosaicked and composite b,g,r,nir,swir image
        output_tif = os.path.join(vrt_section_dir, 'temp.tif')
        clip_tif = os.path.join(vrt_section_dir, section_str+'_'+date+'_mosaic.tif')
        if os.path.isfile(clip_tif)==False:
            try:
                average_mosaic_to_cog(rasters, output_tif, compress='DEFLATE', blocksize=512)
            except Exception as e:
                traceback.print_exc()
                print('error mosaicking')
                pass

            try:
                # Open the input raster
                with rasterio.open(output_tif) as src:
                    # Clip the raster
                    crs = src.crs
                    
                    gdf = gpd.read_file(reference_polygon)
                    gdf = gdf.to_crs(crs)
                    gdf['geometry'] = gdf.buffer(1200)
                    minx, miny, maxx, maxy = gdf.total_bounds
                    clipping_geometry = [box(minx, miny, maxx, maxy)]
                    out_image, out_transform = mask(src, clipping_geometry, crop=True)
                    # Update metadata for the output raster
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })

                    # Write the clipped raster to a new file
                    with rasterio.open(clip_tif, "w", **out_meta) as dest:
                        dest.write(out_image)
            except:
                print('error clipping')
        
        ###remove the otsu thresholds if they already exist
        if os.path.isfile(otsu_path_nir)==True and os.path.isfile(otsu_path_swir==True):
            os.remove(otsu_path_nir)
            os.remove(otsu_path_swir)
            os.remove(otsu_path_ndwi)
            os.remove(otsu_path_mndwi)


        ###here we are computing the otsu thresholds
        nir_thresh,profile = compute_otsu_threshold(clip_tif, metric='NIR')
        swir_thresh,profile = compute_otsu_threshold(clip_tif, metric='SWIR')
        ndwi_thresh,profile = compute_otsu_threshold(clip_tif, metric='NDWI')
        mndwi_thresh,profile = compute_otsu_threshold(clip_tif, metric='MNDWI')
        

        ####here we are making the profile for the otsu thresholds
        profile.update({
            'driver':'GTiff',
            'dtype': 'uint8',
            'count': 1,
            'nodata': 0,
            'compress': 'lzw'  # Optional: compress the output
        })

        ###here we are writing the otsu thresholds
        with rasterio.open(temp_otsu_path_nir, 'w', **profile) as dst:
            dst.write(nir_thresh, 1)
        with rasterio.open(temp_otsu_path_swir, 'w', **profile) as dst:
            dst.write(swir_thresh, 1)
        with rasterio.open(temp_otsu_path_ndwi, 'w', **profile) as dst:
            dst.write(ndwi_thresh, 1)
        with rasterio.open(temp_otsu_path_mndwi, 'w', **profile) as dst:
            dst.write(mndwi_thresh, 1)            

        ###########this runs deep learning model, there are issues with segment_vrt_sections that lead to unreliable model outputs
        # output_dir = os.path.join(vrt_section_dir, 'seg')
        # output_files, output_dir = segment_vrt_sections(clip_tif, output_dir, tile_size=1000, overlap_percent=50)
        # output_tif = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_seg.tif')
        # output_vrt = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_seg.vrt')
        # create_final_vrt(output_files, output_vrt, output_tif, output_dir)


        ###here we are clipping the otsu thresholds to the buffered shoreline change envelope
        try:
            with rasterio.open(temp_otsu_path_nir) as src:
                # Clip the raster
                crs = src.crs
                gdf = gpd.read_file(reference_polygon)
                gdf = gdf.to_crs(crs)
                gdf['geometry'] = gdf.buffer(1000)
                minx, miny, maxx, maxy = gdf.total_bounds
                clipping_geometry = [box(minx, miny, maxx, maxy)]
                out_image, out_transform = mask(src, clipping_geometry, crop=True)
                # Update metadata for the output raster
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Write the clipped raster to a new file
                with rasterio.open(otsu_path_nir, "w", **out_meta) as dest:
                    dest.write(out_image)
        except:
            print('error clipping otsu')
        try:
            with rasterio.open(temp_otsu_path_swir) as src:
                # Clip the raster
                crs = src.crs
                gdf = gpd.read_file(reference_polygon)
                gdf = gdf.to_crs(crs)
                gdf['geometry'] = gdf.buffer(1000)
                minx, miny, maxx, maxy = gdf.total_bounds
                clipping_geometry = [box(minx, miny, maxx, maxy)]
                out_image, out_transform = mask(src, clipping_geometry, crop=True)
                # Update metadata for the output raster
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Write the clipped raster to a new file
                with rasterio.open(otsu_path_swir, "w", **out_meta) as dest:
                    dest.write(out_image)
        except:
            print('error clipping otsu')
        try:
            with rasterio.open(temp_otsu_path_ndwi) as src:
                # Clip the raster
                crs = src.crs
                gdf = gpd.read_file(reference_polygon)
                gdf = gdf.to_crs(crs)
                gdf['geometry'] = gdf.buffer(1000)
                minx, miny, maxx, maxy = gdf.total_bounds
                clipping_geometry = [box(minx, miny, maxx, maxy)]
                out_image, out_transform = mask(src, clipping_geometry, crop=True)
                # Update metadata for the output raster
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Write the clipped raster to a new file
                with rasterio.open(otsu_path_ndwi, "w", **out_meta) as dest:
                    dest.write(out_image)
        except:
            print('error clipping otsu')
        try:
            with rasterio.open(temp_otsu_path_mndwi) as src:
                # Clip the raster
                crs = src.crs
                gdf = gpd.read_file(reference_polygon)
                gdf = gdf.to_crs(crs)
                gdf['geometry'] = gdf.buffer(1000)
                minx, miny, maxx, maxy = gdf.total_bounds
                clipping_geometry = [box(minx, miny, maxx, maxy)]
                out_image, out_transform = mask(src, clipping_geometry, crop=True)
                # Update metadata for the output raster
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Write the clipped raster to a new file
                with rasterio.open(otsu_path_mndwi, "w", **out_meta) as dest:
                    dest.write(out_image)
        except:
            print('error clipping otsu')

        ##this computes average land mask, if deep learning output is to be included then add in output_tif to the rasters list
        rasters = [otsu_path_nir, otsu_path_swir, otsu_path_ndwi, otsu_path_mndwi]#, output_tif]
        output_tif_path = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_avg_land_mask.tif')
        reference_polygon_path = os.path.join(r_home, section, section_str+'_reference_polygon.geojson')
        polygon_path = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_avg_land_mask.geojson')
        if os.path.isfile(output_tif_path)==True:
            os.remove(output_tif_path)   
        if os.path.isfile(polygon_path)==True:
            os.remove(polygon_path) 
        try:
            mode_composite_to_cog(rasters,output_tif_path,compress='DEFLATE',blocksize=512,verbose=True)
        except Exception as e:
            traceback.print_exc()
            pass
        try:
            os.remove(os.path.join(vrt_section_dir, 'temp_otsu_nir.tif'))
        except:
            pass
        try:
            os.remove(os.path.join(vrt_section_dir, 'temp_otsu_swir.tif'))
        except:
            pass
        try:
            os.remove(os.path.join(vrt_section_dir, 'temp.tif'))
        except:
            pass
        with rasterio.open(clip_tif) as src:
            x_res, y_res = src.res
            RES = max(x_res,y_res)


        ###polygonzing the land mask
        cmd = 'python gdal_polygonize.py ' + output_tif_path +' -b 1 -f GeoJSON ' + polygon_path + ' land_mask'
        land_masks.append(polygon_path)
        os.system(cmd)


        ###here we are adding decade as a field to the land mask
        reference_polygon_gdf = gpd.read_file(reference_polygon_path)
        reference_polygon_gdf = wgs84_to_utm_df(reference_polygon_gdf)
        reference_polygon_gdf = reference_polygon_gdf.buffer(100)
        reference_polygon_gdf = utm_to_wgs84_df(reference_polygon_gdf)
        land_mask_gdf = gpd.read_file(polygon_path)
        land_mask_gdf = land_mask_gdf.to_crs(reference_polygon_gdf.crs)
        land_mask_gdf.to_file(polygon_path)

        ##here we we are making some shorelines from the land mask, smoothing, and adding decade as a field
        land_mask_lines_gdf = land_mask_gdf.boundary
        land_mask_lines_gdf_clip = gpd.clip(land_mask_lines_gdf, reference_polygon_gdf)
        multilines = [geom for geom in land_mask_lines_gdf_clip if geom.geom_type == 'MultiLineString']
        singlelines = [geom for geom in land_mask_lines_gdf_clip if geom.geom_type == 'LineString']
        lines =convert_multilinestring_list_to_linestring_list(multilines)
        lines = lines+singlelines
        if interval=='decadal':
            lines_gdf  = {'decade':[int(date)]*len(lines),
                            'geometry': lines
                            }
        elif interval=='annual':
            lines_gdf  = {'year':[int(date)]*len(lines),
                        'geometry': lines
                            }      
        lines_gdf = gpd.GeoDataFrame(lines_gdf, crs="EPSG:4326")
        shoreline_file = os.path.join(vrt_section_dir, section_str+'_'+str(date)+'_avg_land_mask_shoreline.geojson')
        if os.path.isfile(shoreline_file)==True:
            os.remove(shoreline_file)
        if len(lines_gdf)>0:
            lines_gdf = smooth_lines_constant(lines_gdf,RES,refinements=2)
            lines_gdf.to_file(shoreline_file)
            shoreline_files.append(shoreline_file)  

    ##archive the section
    compress_and_remove_directory(vrt_section_dir)   

    return shoreline_files, land_masks

def composite_analysis_region(g, c, rr, interval, r_home, alaska_vrts_folder, image_score, time=''):
    
    try:
        os.mkdir(os.path.join(alaska_vrts_folder, 'G'+g, 'C'+c, 'RR'+rr))
    except:
        pass
    sections = sorted(get_immediate_subdirectories(r_home))
    for section in sections:
        sss=section[3:]
        section_str=g+c+rr+sss
        vrt_section_dir = os.path.join(alaska_vrts_folder, 'G'+g, 'C'+c,'RR'+rr,'SSS'+sss)
        shoreline_files, land_masks = composite_analysis_section(g,c,rr,sss,interval,alaska_vrts_folder,r_home,image_score, time=time)
        try:
            shoreline_files_gdf = pd.concat([gpd.read_file(f) for f in shoreline_files])
            if inteval=='decadal':
                shoreline_files_gdf.to_file(os.path.join(vrt_section_dir, section_str+'_decadal_shorelines.geojson'))
            elif interval=='annual':
                shoreline_files_gdf.to_file(os.path.join(vrt_section_dir, section_str+'_annual_shorelines.geojson'))
        except Exception as e:
            traceback.print_exc()
            pass
        for f in shoreline_files:
            try:
                os.remove(f)
            except Exception as e:
                traceback.print_exc()
                pass
        try:
            land_mask_files_gdf = pd.concat([gpd.read_file(f) for f in land_masks])
            if interval=='decadal':
                land_mask_files_gdf.to_file(os.path.join(vrt_section_dir, section_str+'_decadal_land_masks.geojson'))
            elif interval=='annual':
                land_mask_files_gdf.to_file(os.path.join(vrt_section_dir, section_str+'_annual_land_masks.geojson'))
        except Exception as e:
            traceback.print_exc()
            pass
        for f in land_masks:
            try:
                os.remove(f)
            except Exception as e:
                traceback.print_exc()
                pass


args = parser.parse_args()
g = args.global_region
c = args.coastal_area
rr = args.subregion
sss = args.shoreline_section
interval = args.interval
directory_name = args.directory_name
image_score = args.image_score
image_score = float(image_score)
time = args.time
home = os.path.join('/', 'mnt', 'hdd_6tb','Alaska_Analysis_Images', 'G'+g, 'C'+c)
alaska_vrts_folder = os.path.join('/', 'mnt', 'f', 'Merbok')
alaska_vrts_folder = os.path.join(alaska_vrts_folder, directory_name)
try:
    os.mkdir(alaska_vrts_folder)
except:
    pass
try:
    os.mkdir(os.path.join(alaska_vrts_folder, 'G'+g))
except:
    pass
try:
    os.mkdir(os.path.join(alaska_vrts_folder, 'G'+g, 'C'+c))
except:
    pass
try:
    os.mkdir(os.path.join(alaska_vrts_folder, 'G'+g, 'C'+c, 'RR'+rr))
except:
    pass
r_home = os.path.join(home, 'RR'+rr)
if sss!='':
    sss = args.shoreline_section
    composite_analysis_section(g, c, rr, sss, interval, alaska_vrts_folder, r_home, image_score, time=time)
else:
    composite_analysis_region(g, c, rr, interval, r_home, alaska_vrts_folder, image_score, time=time)


