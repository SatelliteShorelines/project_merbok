import pandas as pd
import numpy as np
import os
import rasterio
from tqdm import tqdm
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from rasterio.warp import reproject, calculate_default_transform
import helpers

def apply_shifts_to_tiffs(df,coregistered_dir,session_dir,satellites:list=None,apply_shifts_filter_passed=True):
    """
    Applies shifts to TIFF files based on the provided DataFrame and copies unregistered files to the coregistered directory.
    Parameters:
    df (pandas.DataFrame): DataFrame containing information about the files, including whether they passed filtering.
    coregistered_dir (str): Directory where the coregistered files will be stored.
    session_dir (str): Directory of the current session containing the original files.
    satellites (list): List of satellite names to process.
    apply_shifts_filter_passed (bool): If True, apply the shifts to only the files that passed the filtering. If False, apply the shifts to all files.
    Returns:
    None
    """    
    if apply_shifts_filter_passed:
        # Apply the shifts to the other files if they passed the filtering
        filenames = df[df['filter_passed']==True]['filename']
    else: # get all the filenames whether they passed the filtering or not
        filenames = df['filename']

    if satellites:
        apply_shifts_for_satellites(df,filenames,coregistered_dir,session_dir,satellites)
    else:
        apply_shifts_to_files_planet(df,filenames,coregistered_dir,session_dir)

def apply_shift_to_tiff(target_path:str, output_path:str, shift:np.ndarray,verbose=False):
    """
    Applies a spatial shift to a GeoTIFF file and writes the result to a new file.
    Parameters:
    target_path (str): The file path to the input GeoTIFF file.
    output_path (str): The file path to save the output GeoTIFF file with the applied shift.
    shift (np.ndarray): A numpy array containing the shift values [y_shift, x_shift].
    verbose (bool, optional): If True, prints detailed information about the process. Default is False.
    Returns:
    None
    """
    if verbose:
        print(f"Applying shift {shift}")
    with rasterio.open(target_path) as src:
        meta = src.meta.copy()

        transform_shift = Affine.translation(shift[1], shift[0])  # x=shift[1], y=shift[0]
        meta['transform'] = src.transform * transform_shift

        # Ensure no changes to image data
        meta.update({
            'compress': src.compression if src.compression else 'lzw',  # Preserve compression
            'dtype': src.dtypes[0],  # Preserve data type
        })

        if src.nodata is not None:
            meta.update({'nodata': src.nodata})

        if verbose:
            print(f"Original transform:\n{src.transform}")
            print(f"Updated transform:\n{meta['transform']}")

        # Write a new file with updated metadata
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(src.read())  # Writes all bands directly

def change_to_crs(new_crs, target_path, output_dir,keep_resolution=True):
    """
    Matches the Coordinate Reference System (CRS) of the target image to that of the reference image and saves the resampled image.
    NOTE: IF KEEP RESOLUTION IS FALSE THIS CHANGES THE RESOLUTION OF THE TARGET IMAGE

    NOTE: THIS WILL CHANGE THE WIDTH AND HEIGHT OF THE TARGET IMAGE

    # IF KEEP RESOLUTION IS FALSE THE RESOLUTION,WIDTH AND HEIGHT OF THE TARGET IMAGE WILL BE CHANGED
    
    Parameters:
    new_crs (str): The CRS to which the target image will be resampled.
    target_path (str): Path to the target image file that needs to be resampled.
    output_dir (str): Directory where the resampled image will be saved.
    keep_resolution (bool): If True, the resolution of the target image is maintained. Default is True.
    Returns:
    str: Path to the resampled image with the matched CRS.
    """
    # print(f"new_crs: {new_crs}")

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(target_path))

    # Open the target image and update metadata to match reference CRS
    with rasterio.open(target_path) as target:
        src_crs = target.crs
        target_meta = target.meta.copy()
        # Calculate the transformation parameters
        if keep_resolution:
            transform, width, height = calculate_default_transform(
                src_crs, new_crs, target.width, target.height, *target.bounds,resolution=target.res
            )
        else:
            transform, width, height = calculate_default_transform(
                src_crs, new_crs, target.width, target.height, *target.bounds)
            
        target_meta.update({
            'driver': 'GTiff',
            'transform': transform,
            'crs': new_crs,
            'width': width,
            'height': height
        })

        # Create output file and perform resampling
        with rasterio.open(output_path, 'w', **target_meta, 
        ) as dst:
            for i in range(1, target.count + 1):  # Iterate over each band
                target_band = target.read(i)
                reproject(
                    source=target_band,
                    destination=rasterio.band(dst, i),
                    src_transform=target.transform,
                    src_crs=target.crs,
                    dst_transform=transform,
                    dst_crs=new_crs,
                    resampling=Resampling.bilinear,
                )
    return output_path

def apply_shifts_to_satellite_files(df: pd.DataFrame, valid_files: list, src_dir: str, dst_dir: str, satname: str,folder_name:str,verbose=False):
    """
    Apply shifts to the specified files based on the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'filename', 'shift_x', and 'shift_y' columns.
        valid_files (list): List of filenames to which the shifts should be applied. (not the full path)
        src_dir (str): Directory path where the original files are located.
        dst_dir (str): Directory path where the coregistered files should be saved.
        satname (str): Satellite name to be used in the directory path.
        folder_name (str): Folder name to be used in the directory path.
            Example: If you want to apply shifts to the 'mask' directory this should be 'mask'.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    # Get the pan/swir & mask path for each ms file
    for file in valid_files:
        # Apply the shifts to the files
        # 1. Get the shift from the DataFrame
        shift_x = df[df['filename'] == file]['shift_x'].values[0]
        shift_y = df[df['filename'] == file]['shift_y'].values[0]

        # Check if the CRS was changed
        new_crs = df[df['filename'] == file]['CRS'].values[0]
        CRS_converted = df[df['filename'] == file]['CRS_converted'].values[0]

        # 2. Apply the shift to the file
        src_path = os.path.join(src_dir, file.replace('ms', folder_name))
        # dst_folder = os.path.join(dst_dir, satname, folder_name)

        dst_path = os.path.join(dst_dir, satname, folder_name, file.replace('ms', folder_name))
        if os.path.exists(src_path):
            # convert the shift 
            if new_crs:
                src_path = helpers.convert_to_new_crs(src_path,new_crs,keep_resolution=True)

            apply_shift_to_tiff(src_path, dst_path, (shift_y, shift_x), verbose=verbose)
            # print(f"Applied shift to {os.path.basename(dst_path)}")

def apply_shifts_to_files_planet(df: pd.DataFrame, valid_files: list, src_dir: str, dst_dir: str,verbose=False):
    """
    Apply shifts to the specified files based on the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'filename', 'shift_x', and 'shift_y' columns.
        valid_files (list): List of filenames to which the shifts should be applied. (not the full path)
        src_dir (str): Directory path where the original files are located.
        dst_dir (str): Directory path where the coregistered files should be saved.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    # Get the pan/swir & mask path for each ms file
    for file in valid_files:
        # Apply the shifts to the files
        # 1. Get the shift from the DataFrame
        shift_x = df[df['filename'] == file]['shift_x'].values[0]
        shift_y = df[df['filename'] == file]['shift_y'].values[0]

        # udm file : 20200603_203636_82_1068_3B_udm2_clip.tif
        # ms file  : 20200603_203636_82_1068_3B_AnalyticMS_toar_clip.tif

        # 2. Apply the shift to the file
        src_path = os.path.join(src_dir, file.replace('AnalyticMS_toar_clip', 'udm2_clip'))
        dst_path = os.path.join(dst_dir, file.replace('AnalyticMS_toar_clip', 'udm2_clip'))
        if os.path.exists(src_path):
            apply_shift_to_tiff(src_path, dst_path, (shift_y, shift_x), verbose=verbose)


def apply_shifts_for_satellites(df,passed_coregs,coreg_dir,unregistered_dir,satellites:list[str],verbose:bool=False):
    # make a subdirectory for each satellite
    for satname in tqdm(satellites,desc = "Applying shifts 'mask','pan', and 'swir' folders for all the satellites"):
        mask_dir = os.path.join(unregistered_dir, satname, 'mask')
        apply_shifts_to_satellite_files(df, passed_coregs, mask_dir, coreg_dir, satname, 'mask',verbose=verbose)
        if satname == 'S2':
            swir_dir = os.path.join(unregistered_dir, satname, 'swir')
            apply_shifts_to_satellite_files(df, passed_coregs, swir_dir, coreg_dir, satname, 'swir',verbose=verbose)
        elif satname in ['L7','L8','L9']:
            pan_dir = os.path.join(unregistered_dir, satname, 'pan')
            apply_shifts_to_satellite_files(df, passed_coregs, pan_dir, coreg_dir, satname, 'pan',verbose=verbose)