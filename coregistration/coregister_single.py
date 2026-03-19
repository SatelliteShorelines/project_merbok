# This script was written on 1/27/25
# By Sharon Batiste

# This script will coregister a tiff to a single template/reference image.
# The coregistered images will be saved to a folder called "coregistered_planet" 
# The coregistration results will be saved to a json file called "coreg_results.json".

# -------------
# Note: In order to use arosics the nodata values cannot be set to -inf and inf & both tiffs must be in the same CRS. 
# This script takes care of that. If you want to learn more read the documentation below:
# Excerpt from the arosics documentation:
#   "The no-data value of each image is automatically derived from the image corners. However, this may fail if the actual no-data value is not present within a 3x3 matrix at the image corners.
#    User provided no-data values will speed up the computation and avoid wrongly derived values."
#    From : https://danschef.git-pages.gfz-potsdam.de/arosics/doc/usage/input_data_requirements.html
# -------------

from coregistration.helpers import *
from coregistration.arosics_filter import *
import coregistration.file_utilites as file_utils
import os

def co_register_single(im_reference, im_target):
    # 1. Create a folder to save the coregistered images
    output_folder = os.path.join(os.path.dirname(im_target),"coregistered")
    # This is the directory to save any of the targets that have been modified
    modified_target_folder = "modified_targets"
    modified_template_folder = "modified_templates"
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(modified_target_folder, exist_ok=True) # This is the directory to save any of the targets that have been modified
    os.makedirs(modified_template_folder, exist_ok=True)

    # Step 2a. Make a set of settings to coregitser the target to the reference
    coregister_settings = {
        "ws": (256, 256), # window size: this is the size of the window used to calculate the coregistration shifts
        "nodata":(0,0),   # This is the no data value in the images
        "max_shift":100,  # This is the maximum allowable shift in pixels
        "binary_ws":False, # This forces the window size to be a power of 2
        "progress":False,  # This shows the progress of the coregistration
        "v":False,         # This shows the verbose output
        "ignore_errors":True, # Useful for batch processing. In case of error COREG.success == False and COREG.x_shift_px/COREG.y_shift_px is None        
        "fmt_out": "GTiff",
    }

    # Step 3. Define the reference and target images
    im_reference = im_reference
    im_target = im_target # This is the image that will be coregistered to the reference image

    # preprocess the reference image
    modified_reference_path = os.path.join(modified_template_folder, os.path.basename(im_reference))
    # arosics cannot handle -inf as a no data value so we set it to 0 (Note make sure that this matches the no data value in the coregister settings)
    im_reference = update_nodata_value(im_reference, modified_reference_path, new_nodata=0)

    # Step 4. Coregister the images in the target folder
    coreg_result = coregister_file(im_reference,im_target,output_folder,modified_target_folder,coregister_settings)
    coreg_result.update({"settings":coregister_settings})

    # Step 5. Save the coregistered Result to a json file
    json_save_path = os.path.join(output_folder, "coreg_result.json")
    save_to_json(coreg_result, json_save_path)


    # Step 6. Clean Up 
    # Remove the modified reference image and the modified target image
    file_utils.delete_like_file(file_utils.get_root_name(im_target),modified_target_folder)
    file_utils.delete_like_file(file_utils.get_root_name(im_reference),modified_template_folder)