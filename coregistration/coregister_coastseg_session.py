# This script was written on 1/21/25
# By Sharon Batiste

# This script will coregister a directory of tiffs to a single template/reference  image.
# The coregistered images will be saved to a folder called "coregistered" within the session directory.
# The coregistered images will be filtered and the results will be saved to a csv file called "filtered_files.csv" within the "coregistered" folder.
# Any tiffs that failed the coregistration will be saved to a folder called "failed_coregistration" within the "coregistered" folder.
# The coregistration results will be saved to a json file called "coreg_results.json" within the "coregistered" folder.

# -------------
# Excerpt from the arosics documentation: https://danschef.git-pages.gfz-potsdam.de/arosics/doc/usage/input_data_requirements.html
# The no-data value of each image is automatically derived from the image corners. However, this may fail if the actual no-data value is not present within a 3x3 matrix at the image corners.
#  User provided no-data values will speed up the computation and avoid wrongly derived values.
# -------------

from helpers import *
from arosics_filter import *
import os
import glob
from tqdm import tqdm
import file_utilites as file_utils
import geo_utils

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

# Step 1. Make a set of settings to coregitser the target to the reference
coregister_settings = {
    "ws": (256, 256), # window size
    "nodata":(0,0),
    "max_shift":100,
    "binary_ws":False, # This forces the window size to be a power of 2
    "progress":True,  # This shows the progress of the coregistration
    "v":False,         # This shows the verbose output
    "ignore_errors":True, # Useful for batch processing. In case of error COREG.success == False and COREG.x_shift_px/COREG.y_shift_px is None        
    "fmt_out": "GTiff",
}
# Step 1b. Make a set of settings to filter the coregistration results
filtering_settings = {
    'shift_reliability': 40,  # Default minimum threshold for shift reliability percentage. 
    'window_size': 50,  # Minimum size of the window used to calculate coregistration shifts in pixels (smaller is worse)
    'max_shift_meters': 250,  # Default maximum allowable shift in meters
    'filter_z_score': True,  # Flag to determine if z-score filtering should be applied
    'filter_z_score_filter_passed_only': False,  # Flag to apply z-score filtering only to data that has passed previous filters
    'z_score_threshold': 2  # Threshold for z-score beyond which data points are considered outliers
}
# Step 1c. Choose whether to pansharpen all bands or use the CoastSat pansharpening method to create jgps
pansharpen_all_bands = True # If true, pansharpens all bands before creating jpgs

# This is the directory to save any of the targets that have been modified
modified_target_folder = "modified_targets"
modified_template_folder = "modified_templates"
  
os.makedirs(modified_target_folder, exist_ok=True)
os.makedirs(modified_template_folder, exist_ok=True)

create_jpgs = True # Create jpgs for the files that passed the filtering

# Session and Template Paths
ROI_ID = "" # set this if you have a session with multiple ROIs or leave it as None if you want to use only the first ROI

# Step 2. Define the session directory and the template path
session_dir = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original_mess_with'
template_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original\L9\ms\2023-06-30-22-01-55_L9_ID_1_datetime11-04-24__04_30_52_ms.tif"
sorted_jpg_path = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original_mess_with\jpg_files\preprocessed\RGB'



# preprocess the reference image
modified_reference_path = os.path.join(modified_template_folder, os.path.basename(template_path))
template_path = update_nodata_value(template_path, modified_reference_path, new_nodata=0)

# CoastSeg Session
# 1. read config.json file to get the satellites
config_path = os.path.join(session_dir, 'config.json')
config = file_utils.read_json_file(config_path)
# Allow the user to select a specific ROI or just use the first one
# Enter a specific ROI ID or or just use the first one by entering roi_id = None
satellites = file_utils.get_satellites(config_path,roi_id=None)
# get the first ROI ID or replace this one
roi_id = config['roi_ids'][0] if not ROI_ID else ROI_ID

# remove L7 since we can't coregister it
if 'L7' in satellites:
    satellites.remove('L7')
print(f"Satellites: {satellites}")

# make a new coregistered directory to save everything in ( not a problem is it already exists)
coregistered_dir= file_utils.create_coregistered_directory(session_dir,satellites)
result_json_path = os.path.join(coregistered_dir, 'transformation_results.json')
# path to save the csv with the results
csv_path = os.path.join(coregistered_dir, "filtered_files.csv")

# copy the config_gdf.geojson files to the coregistered directory
shutil.copy(os.path.join(session_dir, 'config_gdf.geojson'), os.path.join(coregistered_dir, 'config_gdf.geojson'))
print(f"Coregisted directory: {coregistered_dir}")


# get the filtered jpgs
# returns dictionary of satname : [date, date, date]
filtered_dates_by_sat = get_filtered_dates_dict(sorted_jpg_path, 'jpg')
# drop L7 since we can't coregister it
if 'L7' in filtered_dates_by_sat:
    filtered_dates_by_sat.pop('L7')

# # @ debug ONLY only keep S2 for now
# satellites = ['S2','L8']
# filtered_dates_by_sat = {satellite: filtered_dates_by_sat[satellite] for satellite in satellites}


# list the directories in the session directory
# 2. loop through the directories
results = {}
for satellite in tqdm(filtered_dates_by_sat.keys(),desc='Coregistering Satellites'):
    print(f"Processing {satellite}")
    # 1. Get the satellite directory and its multispectral directory (ms)
    satellite_dir = os.path.join(session_dir,satellite)
    ms_dir = os.path.join(satellite_dir,'ms')
    # only read the ms files that match the dates in 
    tif_files = glob.glob(os.path.join(ms_dir, '*.tif')) 
    # convert each tif to a date and if its not in the filtered dates, remove it
    # example ms file : 2021-05-15-22-02-03_L8_ID_1_datetime11-04-24__04_30_52_ms.tif
    tif_filenames = [tif for tif in tif_files if extract_date_from_filename(os.path.basename(tif)) in filtered_dates_by_sat[satellite]]
    # create full paths again for the tif files
    tif_files = [os.path.join(ms_dir, tif) for tif in tif_filenames]
    
    output_dir = os.path.join(coregistered_dir,satellite,'ms') # directory to save coregistered files to
    # coregister all the tif files for this satellite
    os.makedirs(modified_target_folder, exist_ok=True)
    results[satellite] = coregister_files(tif_files, template_path, output_dir,modified_target_folder, coregister_settings,desc=f'Detecting shifts for {satellite} files:')
    
    # remove the modified target files for that satellite
    file_utils.delete_folder(modified_target_folder)


    # after each set of tif files are coregistered, save the results

    save_coregistered_results(results, satellite,  result_json_path, coregister_settings.copy().update({ 'template_path': template_path}))


new_config_path = file_utils.save_coregistered_config(config_path,coregistered_dir,coregister_settings)

#Step 6. Filter the coregistration results
df = filter_coregistration(result_json_path,coregistered_dir,csv_path,filtering_settings)

# Move the files that failed the filtering to a new directory in coregistered directory
failed_coregs = df[~df['filter_passed']].groupby('satellite')['filename'].apply(list).to_dict()
file_utils.process_failed_coregistrations(failed_coregs, coregistered_dir, session_dir,replace=False, copy_only=False, move_only=True, subfolder_name='ms')

# Copy remaining files (swir,pan,mask,meta) to the coregistered directory. If replace replace_failed_files = true copy the unregistered versions of these files
file_utils.copy_remaining_tiffs(df,coregistered_dir,session_dir,satellites,replace_failed_files=False)

# Copy the files (meta, swir, pan ) that passed coregistration and apply the shifts to them
geo_utils.apply_shifts_to_tiffs(df,coregistered_dir,session_dir,satellites,apply_shifts_filter_passed=True)

# Create jpgs for the files which passed the filtering and copy the jpgs from the files that failed the filtering
# make sure to allow users to turn off the copying of the origianl files just in case they don't want them

# read these from the settings section of config.json
inputs  = file_utils.get_config(new_config_path,roi_id) # this gets the setting for this ROI ID
config = file_utils.get_config(new_config_path)

# rename sat_list to satname to make it work with create_coregistered_jpgs
inputs['satname'] = inputs.pop('sat_list')
file_utils.create_coregistered_jpgs(inputs, settings = config['settings'], pansharpen_all_bands = pansharpen_all_bands)


print(f"\n\nCoregistration complete! You can find all coregistered outputs in the following directory: {coregistered_dir}")
print(f"\nDetails of the transformation results are available in this file: 'transformation_results.json' at: {result_json_path}")
print(f"\nA CSV file listing all applied filters to the files, 'filtered_files.csv', is located at: {csv_path}")