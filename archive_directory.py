import os
import shutil
import tarfile

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
        print('already archived, so making different folder')
        output_filename_new = source_directory+'_2.tar.gz'
        try:
            # Create the .tar.gz archive
            with tarfile.open(output_filename_new, "w:gz") as tar:
                arcname = os.path.basename(source_directory)+'_2'
                tar.add(source_directory, arcname=arcname)
            print(f"Successfully created {output_filename_new}")

            # Remove the original directory
            shutil.rmtree(source_directory)
            print(f"Successfully removed original directory: {source_directory}")
        except Exception as e:
            print(f"An error occurred: {e}")
            pass          



# """
# Use this to archive directories to free up space.
# """
# g='1'
# c='4'
# rrs = ['00', '01', '02', '03']
# for rr in rrs:
#     r_home = os.path.join('/', 'mnt', 'hdd_6tb', 'Alaska_Analysis_Images', 'G'+g, 'C'+c, 'RR'+rr)
#     sections = sorted(get_immediate_subdirectories(r_home))
#     for section in sections:
#         print(rr)
#         print(section)
#         compress_and_remove_directory(os.path.join(r_home, section, 'ms_tiff_paths', 'L5'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'ms_tiff_paths', 'L7'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'ms_tiff_paths', 'L8'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'ms_tiff_paths', 'L9'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'ms_tiff_paths', 'S2'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'ms_tiff_paths', 'PS'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'DEMs'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'elevation_profiles_'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'elevation_profiles_AlaskaDSM'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'elevation_profiles_TBDEM'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'ensemble_timeseries_csvs'))
#         compress_and_remove_directory(os.path.join(r_home, section, 'final_timeseries'))









