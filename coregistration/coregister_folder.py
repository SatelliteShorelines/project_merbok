# This script was written on 1/21/25
# By Sharon Batiste

# This script will coregister a directory of tiffs to a single template/reference  image.
# The coregistered images will be saved to a folder called "coregistered_planet" 
# The coregistered images will be filtered and the results will be saved to a csv file called "filtered_files.csv".
# Any tiffs that failed the coregistration will be saved to a folder called "failed_coregistration" within the "coregistered_planet" folder.
# The coregistration results will be saved to a json file called "coreg_results.json".

# -------------
# Note: In order to use arosics the nodata values cannot be set to -inf and inf & both tiffs must be in the same CRS. 
# This script takes care of that. If you want to learn more read the documentation below:
# Excerpt from the arosics documentation:
#   "The no-data value of each image is automatically derived from the image corners. However, this may fail if the actual no-data value is not present within a 3x3 matrix at the image corners.
#    User provided no-data values will speed up the computation and avoid wrongly derived values."
#    From : https://danschef.git-pages.gfz-potsdam.de/arosics/doc/usage/input_data_requirements.html
# -------------

import os
import glob
from coregistration.helpers import *
from coregistration.arosics_filter import *
import coregistration.file_utilites as file_utils


def co_register_folder(reference_image,
                       target_dir):
    """
    Co registers folder of tiffs

    inputs:
    reference_image (str): path to the reference tif
    target_dir (str): path to the tifs to co-register

    outputs:
    None
    """
    # 1. Create a folder to save the coregistered images
    coregistered_dir = os.path.join(target_dir, "coregistered")
    # This is the directory to save any of the targets that have been modified
    modified_target_folder = "modified_targets"
    modified_template_folder = "modified_templates"
    
    os.makedirs(coregistered_dir, exist_ok=True)
    os.makedirs(modified_target_folder, exist_ok=True)
    os.makedirs(modified_template_folder, exist_ok=True)

    # Step 2a. Make a set of settings to coregitser the target to the reference
    # If you want to use other arosics coregister settings see here: https://danschef.git-pages.gfz-potsdam.de/arosics/doc/arosics.html#module-arosics.CoReg
    coregister_settings = {
        "ws": (256, 256), # window size
        "nodata":(0,0),
        "max_shift":100,
        "binary_ws":False, # This forces the window size to be a power of 2
        "progress":False,  # This shows the progress of the coregistration
        "v":False,         # This shows the verbose output
        "ignore_errors":True, # Useful for batch processing. In case of error COREG.success == False and COREG.x_shift_px/COREG.y_shift_px is None        
        "fmt_out": "GTiff",
    }
    # Step 2b. Make a set of settings to filter the coregistration results
    filtering_settings = {
        'shift_reliability': 40,  # Default minimum threshold for shift reliability percentage. 
        'window_size': 50,  # Minimum size of the window used to calculate coregistration shifts in pixels (smaller is worse)
        'max_shift_meters': 250,  # Default maximum allowable shift in meters
        'filter_z_score': True,  # Flag to determine if z-score filtering should be applied
        'filter_z_score_filter_passed_only': False,  # Flag to apply z-score filtering only to data that has passed previous filters
        'z_score_threshold': 2  # Threshold for z-score beyond which data points are considered outliers
    }

    # Step 3. Define the reference and target images
    im_reference = reference_image
    target_folder = target_dir

    # preprocess the reference image
    modified_reference_path = os.path.join(modified_template_folder, os.path.basename(im_reference))
    im_reference = update_nodata_value(im_reference, modified_reference_path, new_nodata=0)


    # Step 4. Coregister the images in the target folder
    # a. match files that match the pattern
    pattern = '*.tif'
    # get files that match the pattern with glob
    tif_files = glob.glob(os.path.join(target_folder, pattern))
    # b. coregister the files
    coreg_results = coregister_files(tif_files,im_reference,coregistered_dir,modified_target_folder,coregister_settings,desc="Coregistering Images")
    # c. add the settings to the coreg_results under the key "coregister_settings"
    coreg_results.append({"settings":coregister_settings})
    #d. merge the list of dictionaries into a single dictionary
    coreg_results = merge_list_of_dicts(coreg_results)


    # Step 5. Save the coregistered Result to a json file
    result_json_path = os.path.join(coregistered_dir, "coreg_results.json")
    print(f"Saving the coregistered results to {result_json_path}")
    save_to_json(coreg_results, result_json_path)

    #Step 6. Filter the coregistration results
    csv_path = os.path.join(coregistered_dir, "filtered_files.csv")
    df = filter_coregistration(result_json_path,coregistered_dir,csv_path,filtering_settings)
    failed_coregs = list(df[~df['filter_passed']]['filename'].values)

    bad_folder = os.path.join(coregistered_dir, "failed_coregistration")
    os.makedirs(bad_folder, exist_ok=True)

    failed_coregs = list(df[~df['filter_passed']]['filename'].values)
    print(f"Number of failed coregistrations: {len(failed_coregs)}")
    print(f"Total number of coregistrations: {len(df)}")

    move_files_to_folder(failed_coregs,coregistered_dir,bad_folder,move_only=True)

    print(f"\n\nCoregistration complete! You can find all coregistered outputs in the following directory: {coregistered_dir}")
    print(f"\nDetails of the transformation results are available in this file: 'transformation_results.json' at: {result_json_path}")
    print(f"\nA CSV file listing all applied filters to the files, 'filtered_files.csv', is located at: {csv_path}")

    # deletes the modified target folder and the modified template folder
    file_utils.delete_folder(modified_target_folder)
    file_utils.delete_folder(modified_template_folder)