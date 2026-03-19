import pandas as pd
import os
import geopandas as gpd
import json
import numpy as np
from coregistration import filters



# convert dict to dataframe
def coreg_dict_to_dataframe(dict_data):
    """
    Converts a dictionary of coregistration data to a pandas DataFrame.
    
    Parameters:
    dict_data (dict): Dictionary containing coregistration data.
    
    Returns:
    pandas.DataFrame: DataFrame containing the coregistration data.
    """
    if "settings" in dict_data.keys():
        dict_data.pop("settings")

    try:
        df = filters.create_dataframe_with_satellites(dict_data)
    except Exception as e:
        df = pd.DataFrame(dict_data)
        # Flips the dataframe so that the image names are the rows and the columns are the values
        df = df.transpose()

    # Modify DataFrame structure
    df['filename'] = df.index
    df.reset_index(drop=True, inplace=True)
    df = df[['filename'] + [col for col in df.columns if col != 'filename']]
    return df

def filter_out_nones(df, col_name:str = 'coregistered_ssim'):
    """
    Filters out rows where the specified column has None values.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to filter.
    col_name (str): Name of the column to check for None values.
    
    Returns:
    pandas.DataFrame: DataFrame with rows filtered based on None values.
    """
    if 'filter_passed' not in df.columns:
        df['filter_passed'] = True
    
    df['filter_passed'] = df.apply(
    lambda row: False if row[col_name] is None else row['filter_passed'], axis=1
    )
    return df
    
def filter_by_shift_reliability(df, col_name:str = 'shift_reliability', threshold:float = 40):
    """
    Filters rows based on the shift reliability threshold.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to filter.
    col_name (str): Name of the column to check for shift reliability.
    threshold (float): Threshold value for shift reliability.
    
    Returns:
    pandas.DataFrame: DataFrame with rows filtered based on shift reliability.
    """
    if 'filter_passed' not in df.columns:
        df['filter_passed'] = True
    

    df['filter_passed'] = df.apply(
        lambda row: False if row[col_name] is None or row[col_name] < threshold else row['filter_passed'], axis=1
    )
    return df

    
def filter_by_max_shift_meters(df, threshold: float = 200):
    """
    Filters rows based on the maximum shift in meters.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to filter.
    threshold (float): Threshold value for maximum shift in meters.
    
    Returns:
    pandas.DataFrame: DataFrame with rows filtered based on maximum shift in meters.
    """
    if 'filter_passed' not in df.columns:
        df['filter_passed'] = True

    df['filter_passed'] = np.where(
        (df['shift_x_meters'] > threshold) | (df['shift_x_meters'].isnull()) |
        (df['shift_y_meters'] > threshold) | (df['shift_y_meters'].isnull()),
        False,
        df['filter_passed']
    )
    return df



def filter_window_size(df, col_name:str = 'window_size', threshold:int = 50):
    """
    Filters rows based on the window size threshold.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to filter.
    col_name (str): Name of the column to check for window size.
    threshold (int): Threshold value for window size.
    
    Returns:
    pandas.DataFrame: DataFrame with rows filtered based on window size.
    """
    if 'filter_passed' not in df.columns:
        df['filter_passed'] = True

    df['filter_passed'] = np.where(
        (df[col_name].apply(lambda x: x[0]) < threshold) | 
        (df[col_name].apply(lambda x: x[1]) < threshold),
        False,
        df['filter_passed']
    )
    return df

def filter_coregistration(results_path,
                          output_folder,
                          csv_path,
                          filter_settings):
    """
    Filters coregistration results based on specified filter settings and saves the filtered results to a CSV file.
    
    Parameters:
    results_path (str): Path to the JSON file containing coregistration results.
    output_folder (str): Path to the folder where the filtered CSV file will be saved.
    csv_path (str): Path to the CSV file to save the filtered results.
    filter_settings (dict): Dictionary containing filter settings with the following keys:
        - 'shift_reliability' (float, optional): Threshold for shift reliability.
        - 'window_size' (int, optional): Threshold for window size.
        - 'max_shift_meters' (float, optional): Threshold for maximum shift in meters.
        - 'filter_z_score' (bool, optional): Whether to filter by z-score.
        - 'z_score_threshold' (float, optional): Threshold for z-score filtering.
        - 'filter_z_score_filter_passed_only' (bool, optional): Whether to apply z-score filtering only to passed images.
    
    Returns:
    pandas.DataFrame: DataFrame containing the filtered coregistration results.
    """
    # read in the results
    with open(results_path, 'r') as f:
        results = json.load(f)

    df = coreg_dict_to_dataframe(results)

    shift_reliability_threshold = filter_settings.get('shift_reliability',None)
    window_size_threshold = filter_settings.get('window_size',None)
    max_shift_meters_threshold = filter_settings.get('max_shift_meters',None)
    filter_z_score = filter_settings.get('filter_z_score',False)
    z_threshold = filter_settings.get('z_score_threshold',2)
    filter_z_score_filter_passed_only = filter_settings.get('filter_z_score_filter_passed_only',False)

    # apply the filters
    df = filter_out_nones(df)

    if shift_reliability_threshold:
        df = filter_by_shift_reliability(df,threshold=shift_reliability_threshold)

    if window_size_threshold:
        df = filter_window_size(df,threshold=window_size_threshold)

    if max_shift_meters_threshold:
        df = filter_by_max_shift_meters(df,threshold=max_shift_meters_threshold)

    if filter_z_score:
        # If filter_passed is True then the zscore will only be calculated for the passed images
        df = filters.filter_zscores(df, z_threshold=z_threshold, filter_passed_only=filter_z_score_filter_passed_only)
    
    csv_path = os.path.join(output_folder, "filtered_files.csv")
    df.to_csv(csv_path,index=False)
    
    return df
