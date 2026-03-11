import os
import json
import arosics
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.warp import reproject, calculate_default_transform
import shutil
import glob
import re
import tqdm
from collections import OrderedDict
from enum import Enum
from tempfile import NamedTemporaryFile
from shutil import move

def find_satellite_in_filename(filename: str) -> str:
    """Use regex to find the satellite name in the filename.
    Satellite name is case-insensitive and can be separated by underscore (_) or period (.)"""
    class Satellite(Enum):
        L5 = 'L5'
        L7 = 'L7'
        L8 = 'L8'
        L9 = 'L9'
        S2 = 'S2'
    for satellite in Satellite:
        # Adjusting the regex pattern to consider period (.) as a valid position after the satellite name
        if re.search(fr'(?<=[\b_]){satellite.value}(?=[\b_.]|$)', filename, re.IGNORECASE):
            return satellite.value
    return ""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

# turn the list into a dictionary
def merge_list_of_dicts(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        merged_dict.update(d)
    return merged_dict

def save_coregistered_results(results, satellite,  result_json_path, settings):
    """
    Process and save coregistration results ensuring 'settings' is the last item in the dictionary.

    Args:
        results (dict): The coregistration results dictionary.
        satellite (str): The satellite name.
        result_json_path (str): Path to save the resulting JSON file.
        settings (dict): Additional settings to include in the results.

    Returns:
        OrderedDict: The processed results dictionary with 'settings' as the last item.
    """

    # Merge results for the current satellite
    results[satellite] = merge_list_of_dicts(results[satellite])

    results['settings'] = settings

    # Ensure 'settings' is the last key
    results_ordered = OrderedDict(results)
    results_ordered.move_to_end('settings')

    # Save to JSON file
    with open(result_json_path, 'w') as json_file:
        json.dump(results_ordered, json_file, indent=4,cls=NumpyEncoder)

    print(f"Saved results to: {result_json_path}")

    return results_ordered

def get_crs(image_path):
    """
    Get the Coordinate Reference System (CRS) of an image file.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    str: The CRS of the image.
    """
    with rasterio.open(image_path) as img:
        return img.crs

def coregister_file(im_reference, im_target,output_folder,modified_target_folder,coregister_settings):
    """
    Coregisters a target image to a reference image and handles CRS conversion and no data value updates needed to make the
    files fully compatible for coregistration. The coregistered image is saved to the output folder.

    Parameters:
    im_reference (str): Path to the reference image.
    im_target (str): Path to the target image to be coregistered.
    output_folder (str): Path to the folder where the coregistered image will be saved.
    modified_target_folder (str): Path to the folder where modified target images will be saved.
    coregister_settings (dict): Dictionary containing settings for the coregistration process.
    Returns:
    dict: A dictionary containing coregistration results, including CRS information and whether CRS was converted.
        Format:
        {
            "target_filename.tiff": { 'original_ssim': 0.0,
                                    'coregistered_ssim': 0.0,
                                    "change_ssim": 0.0,
                                    'shift_x': 0,
                                    'shift_y': 0,
                                    'shift_x_meters': 0.0,
                                    'shift_y_meters': 0.0,
                                    'shift_reliability': False,
                                    'window_size': [0, 0],
                                    'success': False,
                                    'CRS': CRS,
                                    'CRS_converted':CRS_converted
                                    }

    """
    new_crs = None
    CRS_converted= False
    coreg_result = {os.path.basename(im_target): make_coreg_info()}

    modified_target_path = os.path.join(modified_target_folder, os.path.basename(im_target))
    os.makedirs(modified_target_folder, exist_ok=True)

    # Step 1. If the crs is not the same for the reference and the target, skip the coregistration
    if not check_crs(im_reference, im_target,raise_error=False):
        # create a subfolder in the output folder called "new_crs" and save the target image with the new crs
        new_crs_folder = os.path.join(modified_target_folder, 'new_crs')
        os.makedirs(new_crs_folder, exist_ok=True)
        new_crs = get_crs(im_reference)
        new_crs = str(new_crs)
        im_target = convert_to_new_crs(im_target, new_crs,output_path=modified_target_path, keep_resolution=True)
        CRS_converted = True
        
    # Step 2. Read the current no data value of the target and convert it to a valid no data value
    #    - Invalid No Data Values : -inf, inf

    if os.path.exists(modified_target_path):
        im_target = update_nodata_value( target_path = modified_target_path, new_nodata=0)
    else:
        im_target = update_nodata_value( target_path = im_target,output_path=modified_target_path, new_nodata=0)

    # Step 3. Coregister the target to the reference
    result = coregister_image(im_reference, im_target,output_folder,coregister_settings,verbose=coregister_settings.get("v",False))

    # update coreg_result with the result
    coreg_result.update(result)
    coreg_result[os.path.basename(im_target)].update({'CRS': new_crs, 'CRS_converted':CRS_converted})

    return coreg_result
    

def coregister_files(tif_files, template_path, coregistered_dir,modified_target_folder, coregister_settings,desc='Detecting shifts for files:'):
    """
    Coregisters a list of TIFF files with a given template file and saves the results.
    Parameters:
    tif_files (list of str): List of paths to the target TIFF files to be coregistered.
    template_path (str): Path to the template TIFF file.
    coregistered_dir (str): Directory where the coregistered files will be saved.
    modified_target_folder (str): Directory where the modified target files will be saved.
    coregister_settings (dict): Dictionary containing settings for the coregistration process.
    desc (str, optional): Description for the progress bar. Default is 'Detecting shifts for files:'.
    Returns:
    list: A list of results from the coregistration process for each file.
    """
    results = []
    
    if tif_files == []:
        return results
    
    for target_path in tqdm.tqdm(tif_files,desc=desc):
        result = coregister_file(template_path, target_path,coregistered_dir,modified_target_folder,coregister_settings)
        results.append(result)

    return results

def extract_date_from_filename(filename: str) -> str:
    """Extracts the first instance date string "YYYY-MM-DD-HH-MM-SS" from a filename.
     - The date string is expected to be in the format "YYYY-MM-DD-HH-MM-SS".
     - Example 2024-05-28-22-18-07 would be extracted from "2024-05-28-22-18-07_S2_ID_1_datetime11-04-24__04_30_52_ms.tif"
    """
    pattern = r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    match = re.match(pattern, filename)
    if match:
        return match.group(0)
    else:
        return ""
  
def get_filtered_dates_dict(directory: str, file_type: str, ) -> dict:
    """
    Scans the directory for files with the given file_type and extracts the date from the filename and returns a dictionary with the satellite name as the key and a set of dates as the value.


    Parameters:
    -----------
    directory : str
        The directory where the files are located.

    file_type : str
        The filetype of the files to be included.
        Ex. 'jpg'


    Returns:
    --------
    dict
        a dictionary where each key is a satellite name and each value is a set of the dates in the format "YYYY-MM-DD-HH-MM-SS" that represents the time the scene was captured.
    
    Example:
        {
        "L5":{'2014-12-19-18-22-40',},
        "L7":{},
        "L8":{'2014-12-19-18-22-40',},
        "L9":{},
        "S2":{},
    }
    
    """
    filepaths = glob.iglob(os.path.join(directory, f"*.{file_type}"))

    satellites = {"L5": set(), "L7": set(), "L8": set(), "L9": set(), "S2": set()}
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        date = extract_date_from_filename(filename)
        if not date:
            continue

        satname = find_satellite_in_filename(filename)
        if not satname:
            continue
        
        if satname in satellites:
            satellites[satname].add(date)

    return satellites
        
def move_files_to_folder(filenames,source_folder,destination_folder,copy_only=False,move_only=False,verbose=False):
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder {source_folder} does not exist")
    
    os.makedirs(destination_folder, exist_ok=True)
    # Copy or move the failed coregistration files
    for filename in filenames:
        src = os.path.join(source_folder, filename)
        dst = os.path.join(destination_folder, filename)
        if os.path.exists(src):
            if copy_only:
                shutil.copy(src, dst)
                if verbose:
                    print(f"Copied {filename} to {dst}")
            elif move_only:
                shutil.move(src, dst)
                if verbose:
                    print(f"Moved {filename} to {dst}")

        
def scale_raster(input_path, output_path, scale_factor):
    """
    Scale all band values of a raster by a specified factor.

    Parameters:
    input_path (str): The path to the input raster file.
    output_path (str): The path where the scaled output raster will be saved.
    scale_factor (float): The factor by which to scale the band values.
    """
    # Open the input raster file
    with rasterio.open(input_path) as src:
        # Read metadata from the source
        meta = src.meta
        
        # Create a new raster file with the same metadata
        with rasterio.open(output_path, 'w', **meta) as dst:
            # Loop through each band
            for i in range(1, src.count + 1):
                # Read the band data
                band = src.read(i)
                # Scale the band data
                scaled_band = band * scale_factor
                # Write the scaled band data to the output raster
                dst.write(scaled_band, i)

    return output_path

def make_coreg_info(CR=None,CRS=None,CRS_converted=False):
    """
    Creates a dictionary containing coregistration results with default values. 
    If a coregistration object is provided, it extracts values from it; 
    otherwise, it uses default values.

    Args:
        CR: Optional. A coregistration object containing various attributes with coregistration results.

    Returns:
        dict: A dictionary with coregistration results and metrics, filled with defaults where data is missing or if no object is provided.
    """
    defaults = {
        'original_ssim': 0.0,
        'coregistered_ssim': 0.0,
        "change_ssim": 0.0,
        'shift_x': 0,
        'shift_y': 0,
        'shift_x_meters': 0.0,
        'shift_y_meters': 0.0,
        'shift_reliability': False,
        'window_size': [0, 0],
        'success': False,
        'CRS': CRS,
        'CRS_converted':CRS_converted
    }

    if CR is None:
        return defaults

    original_ssim = getattr(CR, 'ssim_orig', defaults['original_ssim'])
    coregistered_ssim = getattr(CR, 'ssim_deshifted', defaults['coregistered_ssim'])
    # if either of these is None then the change in ssim is also None
    change_ssim = coregistered_ssim - original_ssim if original_ssim is not None and coregistered_ssim is not None else defaults['change_ssim']

    shift_x = CR.coreg_info.get('corrected_shifts_px', {}).get('x', defaults['shift_x']) if hasattr(CR, 'coreg_info') else defaults['shift_x']
    shift_y = CR.coreg_info.get('corrected_shifts_px', {}).get('y', defaults['shift_y']) if hasattr(CR, 'coreg_info') else defaults['shift_y']
    shift_x_meters = CR.coreg_info.get('corrected_shifts_map', {}).get('x', defaults['shift_x_meters']) if hasattr(CR, 'coreg_info') else defaults['shift_x_meters']
    shift_y_meters = CR.coreg_info.get('corrected_shifts_map', {}).get('y', defaults['shift_y_meters']) if hasattr(CR, 'coreg_info') else defaults['shift_y_meters']
    shift_reliability = getattr(CR, 'shift_reliability', defaults['shift_reliability'])

    return {
        'original_ssim': 0  if original_ssim=='null' else original_ssim,
        'coregistered_ssim': 0 if coregistered_ssim=='null' else coregistered_ssim,
        "change_ssim": change_ssim,
        'shift_x': 0 if shift_x == 'null' else shift_x,
        'shift_y': 0 if shift_y == 'null' else shift_y,
        'shift_x_meters': 0 if shift_x_meters == 'null' else shift_x_meters,
        'shift_y_meters': 0 if shift_y_meters == 'null' else shift_y_meters,
        'shift_reliability': 0 if shift_reliability == 'null' else shift_reliability,
        'window_size': getattr(CR, 'fft_win_size_YX', defaults['window_size'])[::-1],
        'success': getattr(CR, 'success', defaults['success']),
        'CRS':CRS,
        'CRS_converted':CRS_converted,
    }


def check_crs(im_reference, im_target,raise_error=False):
    """
    Check if the Coordinate Reference Systems (CRS) of two images are the same.

    Parameters:
    im_reference: The reference image whose CRS is to be compared.
    im_target: The target image whose CRS is to be compared with the reference image.
    raise_error (bool): If True, raises a ValueError when the CRS of the reference and target images are different. 
                        If False, returns False when the CRS are different. Default is False.

    Returns:
    bool: True if the CRS of both images are the same, otherwise False if raise_error is False.

    Raises:
    ValueError: If raise_error is True and the CRS of the reference and target images are different.
    """
    if read_crs(im_reference) != read_crs(im_target):
        # print("Cannot coregister images with different CRS")
        if raise_error:
            raise ValueError("The CRS of the reference and target images are different")
        else:
            return False
    return True

def coregister_image(im_reference, im_target,out_folder,coregister_settings,verbose=False):
    """
    Coregisters the target image to the reference image and saves the coregistered image to the specified output folder.
    Parameters:
    im_reference (str): Path to the reference image.
    im_target (str): Path to the target image to be coregistered.
    out_folder (str): Directory where the coregistered image will be saved.
    coregister_settings (dict): Dictionary containing settings for the coregistration process.
    verbose (bool, optional): If True, prints the path of the saved coregistered image. Default is True.
    Returns:
    dict: A dictionary containing the coregistration results, including:
        - 'original_ssim' (float): Structural similarity index of the original images.
        - 'coregistered_ssim' (float): Structural similarity index of the coregistered images.
        - 'shift_x' (float): Shift in the x-direction in pixels.
        - 'shift_y' (float): Shift in the y-direction in pixels.
        - 'shift_x_meters' (float): Shift in the x-direction in meters.
        - 'shift_y_meters' (float): Shift in the y-direction in meters.
        - 'shift_reliability' (float): Reliability of the calculated shifts.
        - 'window_size' (list): Size of the FFT window used for coregistration [width, height].
        - 'success' (bool): Indicates whether the coregistration was successful.
    """
    path_out = os.path.join(out_folder, os.path.basename(im_target))
    CR = arosics.COREG(im_reference, im_target,path_out = path_out, **coregister_settings)
    CR.calculate_spatial_shifts()
    CR.correct_shifts()

    if verbose:
        print(f"Coregistered image saved to {path_out}")
    # Step 5. Get the coregistration information
    coreg_result=make_coreg_info(CR)
    return {os.path.basename(path_out):coreg_result}

def save_to_json(data, output_path,verbose=False):
    # Save all coregistration results to a JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f,cls=NumpyEncoder) 
    if verbose:
        print(f"Saved coregistration results to {output_path}")  


def update_nodata_value(target_path, output_path=None,new_nodata=0, ):
    """
    Change the nodata value of a raster image and either overwrite the original or write to a specified path.

    Parameters:
    target_path (str): Path to the target image file whose nodata value needs to be updated.
    new_nodata (int or float): The new nodata value to set in the raster.
    output_path (str, optional): Path where the updated image should be saved. If not provided, the original file is overwritten.

    Returns:
    str: Path to the raster image with the updated nodata value.
    """

    # Determine the output file path; use a temporary file if no specific output path is provided
    if output_path is None:
        temp_file = NamedTemporaryFile(delete=False, suffix='.tif')
        output_file_path = temp_file.name
        temp_file.close()  # Close the file so rasterio can open it
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_file_path = output_path

    try:
        with rasterio.open(target_path) as src:
            meta = src.meta.copy()
            old_nodata = src.nodata

            # Update the nodata value in the metadata
            meta['nodata'] = new_nodata

            with rasterio.open(output_file_path, 'w', **meta) as dst:
                for i in range(1, src.count + 1):
                    data = src.read(i)

                    # Replace the old nodata value with the new nodata value if there was one originally
                    if old_nodata is not None:
                        data[data == old_nodata] = new_nodata

                    dst.write(data, i)

        # If no specific output path was given, replace the original file
        if output_path is None:
            os.remove(target_path)
            move(output_file_path, target_path)
            output_file_path = target_path

        return output_file_path

    except Exception as e:
        if output_path is None:
            os.remove(output_file_path)  # Clean up temporary file on failure
        raise RuntimeError(f"Failed to update nodata value: {e}")


# def update_nodata_value(target_path, output_path, new_nodata=0):
#     """
#     Change the nodata value of a raster image.
#     """

#     with rasterio.open(target_path) as src:
#         # Copy metadata from the source file
#         meta = src.meta.copy()

#         # # if the new nodata value is the same as the old nodata value, return the target path
#         # if new_nodata == src.nodata:
#         #     return target_path
        
#         # Update the nodata value in metadata
#         meta.update({'nodata': new_nodata})
        
#         # Create output file with updated nodata value
#         with rasterio.open(output_path, 'w', **meta) as dst:
#             # Iterate over each band in the raster file
#             for i in range(1, src.count + 1):
#                 # Read data from the current band
#                 data = src.read(i)
                
#                 # Replace the old nodata value with the new nodata value
#                 old_nodata = src.nodata
#                 if old_nodata is not None:  # Ensure there's a nodata value to replace
#                     data[data == old_nodata] = new_nodata
                
#                 # Write updated data to the output file
#                 dst.write(data, i)

#     return output_path

def read_crs(path):
    """
    Reads the Coordinate Reference System (CRS) of an image file.
    Parameters:
    path (str): Path to the image file.
    Returns:
    str: The CRS of the image.
    """
    with rasterio.open(path) as img:
        return img.crs

def convert_to_new_crs(target_path, ref_crs, output_path=None, keep_resolution=True):
    """
    Matches the Coordinate Reference System (CRS) of the target image to a specified CRS,
    optionally overwriting the original image or writing to a specified output path.

    If keep_resolution is True, the resolution of the target image is maintained.
    If keep_resolution is False, the resolution of the target image is changed to match the reference CRS.

    NOTE: IF KEEP RESOLUTION IS FALSE THIS CHANGES THE RESOLUTION OF THE TARGET IMAGE
    NOTE: THIS WILL CHANGE THE WIDTH AND HEIGHT OF THE TARGET IMAGE
    # IF KEEP RESOLUTION IS FALSE THE RESOLUTION,WIDTH AND HEIGHT OF THE TARGET IMAGE WILL BE CHANGED
    
    Parameters:
    target_path (str): Path to the target image file that needs to be resampled.
    ref_crs (str): Reference CRS to which the target image should be matched.
    output_path (str, optional): If provided, the directory where the resampled image will be saved. 
                                 If not provided, the original image will be overwritten.
    keep_resolution (bool): If True, maintains the resolution of the target image. Default is True.

    Returns:
    str: Path to the resampled image with the matched CRS.
    """
    # Determine output file path
    if output_path is None:
        temp_file = NamedTemporaryFile(delete=False, suffix='.tif')
        output_file_path = temp_file.name
        temp_file.close()  # Close to allow opening by rasterio
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_file_path = output_path

    try:
        # Open the target image to update metadata and perform resampling
        with rasterio.open(target_path) as target:
            src_crs = target.crs
            target_meta = target.meta.copy()
            no_data = target.nodata  # Capture no-data value from the target

            # Calculate transformation parameters
            if keep_resolution:
                transform, width, height = calculate_default_transform(
                    src_crs, ref_crs, target.width, target.height, *target.bounds, resolution=target.res)
            else:
                transform, width, height = calculate_default_transform(
                    src_crs, ref_crs, target.width, target.height, *target.bounds)

            # Update metadata with new CRS, dimensions, and no-data value
            target_meta.update({
                'driver': 'GTiff',
                'crs': ref_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': no_data
            })

            # Write resampled data to output file
            with rasterio.open(output_file_path, 'w', **target_meta) as dst:
                for i in range(1, target.count + 1):
                    target_band = target.read(i)
                    reproject(
                        source=target_band,
                        destination=rasterio.band(dst, i),
                        src_transform=target.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=no_data,
                        dst_nodata=no_data
                    )

        # If no specific output path was given, replace the original file
        if output_path is None:
            os.remove(target_path)
            move(output_file_path, target_path)
            output_file_path = target_path

        return output_file_path

    except Exception as e:
        if output_path is None:
            os.remove(output_file_path)  # Clean up temporary file on failure
        raise RuntimeError(f"Failed to resample the file: {e}")


# def convert_to_new_crs(reference_path, target_path, output_dir,keep_resolution=True):
#     """
#     Matches the Coordinate Reference System (CRS) of the target image to that of the reference image and saves the resampled image.
#     NOTE: IF KEEP RESOLUTION IS FALSE THIS CHANGES THE RESOLUTION OF THE TARGET IMAGE

#     NOTE: THIS WILL CHANGE THE WIDTH AND HEIGHT OF THE TARGET IMAGE

#     # IF KEEP RESOLUTION IS FALSE THE RESOLUTION,WIDTH AND HEIGHT OF THE TARGET IMAGE WILL BE CHANGED
    
#     Parameters:
#     reference_path (str): Path to the reference image file.
#     target_path (str): Path to the target image file that needs to be resampled.
#     output_dir (str): Directory where the resampled image will be saved.
#     keep_resolution (bool): If True, the resolution of the target image is maintained. Default is True.
#     Returns:
#     str: Path to the resampled image with the matched CRS.
#     """
#     # Create the output directory if it does not exist
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, os.path.basename(target_path))
#     # Open the reference image to get CRS and nodata values
#     with rasterio.open(reference_path) as ref:
#         ref_crs = ref.crs
        
#     # Open the target image and update metadata to match reference CRS
#     with rasterio.open(target_path) as target:
#         src_crs = target.crs
#         target_meta = target.meta.copy()
#         no_data = target.nodata
#         # Calculate the transformation parameters
#         if keep_resolution:
#             transform, width, height = calculate_default_transform(
#                 src_crs, ref_crs, target.width, target.height, *target.bounds,resolution=target.res
#             )
#         else:
#             transform, width, height = calculate_default_transform(
#                 src_crs, ref_crs, target.width, target.height, *target.bounds)
            
#         target_meta.update({
#             'driver': 'GTiff',
#             'transform': transform,
#             'crs': ref_crs,
#             'width': width,
#             'height': height,
#             'nodata': no_data
#         })

#         # Create output file and perform resampling
#         with rasterio.open(output_path, 'w', **target_meta, 
#         ) as dst:
#             for i in range(1, target.count + 1):  # Iterate over each band
#                 target_band = target.read(i)
#                 reproject(
#                     source=target_band,
#                     destination=rasterio.band(dst, i),
#                     src_transform=target.transform,
#                     src_crs=target.crs,
#                     dst_transform=transform,
#                     dst_crs=ref_crs,
#                     resampling=Resampling.bilinear,
#                 src_nodata=no_data,  # Specify source no-data
#                 dst_nodata=no_data,  # Specify destination no-data
#                 )
#     return output_path,ref_crs



def resample_img(template_path, target_path, output_path):
    """
    Reprojects a target image to match the spatial resolution, transform, CRS, and nodata values of a template image.
    Parameters:
    template_path (str): Path to the template image file which provides the reference resolution, transform, CRS, and nodata values.
    target_path (str): Path to the target image file that needs to be reprojected.
    output_path (str): Path where the reprojected image will be saved.
    Returns:
    str: The path to the reprojected image file.
    """
    
    # Open the reference image to get resolution, transform, CRS, and nodata values
    with rasterio.open(template_path) as ref:
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_crs = ref.crs
        ref_nodata = ref.nodata
        ref_dtype = ref.dtypes[0]  # Get the data type of the reference image

    # Open the target image and update metadata to match reference
    with rasterio.open(target_path) as target:
        target_meta = target.meta.copy()
        target_meta.update({
            'driver': 'GTiff',
            'height': ref_height,
            'width': ref_width,
            'transform': ref_transform,
            'crs': ref_crs,
            'nodata': ref_nodata,
            'dtype': ref_dtype  # Set the data type to match the reference image
        })

        # Check and replace invalid nodata value
        if ref_nodata is None or ref_nodata == float('-inf'):
            ref_nodata = 0  # or another valid nodata value for uint16

        target_meta['nodata'] = ref_nodata

        # Create output file and perform resampling
        with rasterio.open(output_path, 'w', **target_meta) as dst:
            for i in range(1, target.count + 1):  # Iterate over each band
                target_band = target.read(i)
                # replace the nodata value in the target image
                target_band[target_band == target.nodata] = ref_nodata
                reproject(
                    source=target_band,
                    destination=rasterio.band(dst, i),
                    src_transform=target.transform,
                    src_crs=target.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.cubic,
                    src_nodata=ref_nodata,
                    dst_nodata=ref_nodata
                )

    return output_path

def modify_and_reproject_tif(target_path, output_path, new_nodata=0, new_crs='EPSG:4326'):
    """
    Change the nodata value and reproject an image to a new CRS.
    """

    with rasterio.open(target_path) as target:
        target_meta = target.meta.copy()
        # print the crs
        print(f"Current CRS: {target.crs}")

        # get the resolution
        resolution = target.res
        print(f"Resolution: {resolution}")
        
        # Calculate the new transform and dimensions based on the new CRS
        new_transform, width, height = calculate_default_transform(
            target.crs, new_crs, target.width, target.height,resolution=target.res, *target.bounds,dst_width=target.width, dst_height=target.height)
        
        # Update metadata with new settings
        target_meta.update({
            'driver': 'GTiff',
            'nodata': new_nodata,
            'crs': new_crs,
            'transform': new_transform,
            'width': width,
            'height': height
        })
        
        # Create output file and perform resampling and reprojection
        with rasterio.open(output_path, 'w', **target_meta) as dst:
            for i in range(1, target.count + 1):  # Iterate over each band
                target_band = target.read(i)
                # Replace the nodata value in the target image
                target_band[target_band == target.nodata] = new_nodata
                
                # Allocate a destination array for the reprojected band
                destination = rasterio.band(dst, i)
                
                # Perform reprojection
                reproject(
                    source=target_band,
                    destination=destination,
                    src_transform=target.transform,
                    src_crs=target.crs,
                    dst_transform=new_transform,
                    dst_crs=new_crs,
                    resampling=Resampling.cubic,
                    src_nodata=target.nodata,
                    dst_nodata=new_nodata
                )
        #print the resolution of the image
        print(f"Resolution: {dst.res}")

    return output_path

def modify_tif(target_path, output_path,new_nodata=0):
    """
    Changes the nodata value of an image to a new value.
    """
    # Open the target image and update metadata to match reference
    with rasterio.open(target_path) as target:
        target_meta = target.meta.copy()
        target_meta.update({
            'driver': 'GTiff',
            'nodata': new_nodata,
            'crs': new_crs
        })

        # Create output file and perform resampling
        with rasterio.open(output_path, 'w', **target_meta) as dst:
            for i in range(1, target.count + 1):  # Iterate over each band
                target_band = target.read(i)
                # replace the nodata value in the target image
                target_band[target_band == target.nodata] = new_nodata
                reproject(
                    source=target_band,
                    destination=rasterio.band(dst, i),
                    src_transform=target.transform,
                    src_crs=target.crs,
                    dst_transform=target.transform,
                    dst_crs=new_crs,
                    resampling=Resampling.cubic,
                    src_nodata=new_nodata,
                    dst_nodata=new_nodata
                )

    return output_path