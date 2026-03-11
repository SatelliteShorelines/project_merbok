"""
Common routines for shoreline extraction in Alaska
"""

##gis
import geopandas as gpd
import shapely
from shapely import Point, LineString, MultiPoint, MultiLineString
import rioxarray

##numerical and data
import numpy as np
import pandas as pd
from scipy import stats
import time

##image processing
from PIL import Image
from skimage import measure

##plotting
import matplotlib.pyplot as plt

#from line_profiler import profile
import gc
import tarfile
import os

def convert_multilinestring_list_to_linestring_list(multilinestring_list):
    """
    Converts a list of MultiLineString objects into a list of LineString objects.
    Each MultiLineString is "exploded" into its individual LineString components.

    Args:
        multilinestring_list (list): A list containing shapely MultiLineString objects.

    Returns:
        list: A new list containing shapely LineString objects.
    """
    linestring_list = []
    for multi_line in multilinestring_list:
        if isinstance(multi_line, MultiLineString):
            for line in multi_line.geoms:
                linestring_list.append(line)
        elif isinstance(multi_line, LineString):
            # If a LineString is already present, add it directly
            linestring_list.append(multi_line)
        else:
            # Handle other geometry types or raise an error as needed
            print(f"Warning: Skipping unsupported geometry type: {type(multi_line)}")
    return linestring_list
    
def unpack_tar_gz(archive_path, extract_dir=None):
    """
    Unpacks a .tar.gz archive to a specified directory.

    Args:
        archive_path (str): The path to the .tar.gz archive file.
        extract_dir (str, optional): The directory where the archive contents
                                     will be extracted. If None, extracts to
                                     the current working directory.
    """
    if not os.path.exists(archive_path):
        print(f"Error: Archive not found at {archive_path}")
        return

    if extract_dir and not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f"Created extraction directory: {extract_dir}")

    try:
        # Open the tar.gz file in read mode ('r:gz')
        with tarfile.open(archive_path, "r:gz") as tar:
            # Extract all contents to the specified directory
            tar.extractall(path=extract_dir)
        print(f"Successfully unpacked '{archive_path}' to '{extract_dir or os.getcwd()}'")
    except tarfile.ReadError as e:
        print(f"Error reading tar.gz file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def wgs84_to_utm_df(geo_df):
    """
    Converts gdf from wgs84 to UTM
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in wgs84
    outputs:
    geo_df_utm (geopandas  dataframe): a geopandas dataframe in utm
    """
    utm_crs = geo_df.estimate_utm_crs()
    gdf_utm = geo_df.to_crs(utm_crs)
    return gdf_utm

def utm_to_wgs84_df(geo_df):
    """
    Converts gdf from utm to wgs84
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in utm
    outputs:
    geo_df_wgs84 (geopandas  dataframe): a geopandas dataframe in wgs84
    """
    wgs84_crs = 'epsg:4326'
    gdf_wgs84 = geo_df.to_crs(wgs84_crs)
    return gdf_wgs84
    
def sample_spatial_kde(kde_path, extracted_shorelines_lines_path, crs):
    """
    Samples point density map with shorelines
    assigns the mode of the sample to each shoreline as 'kde_value'

    inputs:
    kde_path (str): path to the point density map (.tif)
    extracted_shorelines_lines_path (str): path to the extracted shorelines (.geojson)

    outputs:
    extracted_shorelines_lines_path (str): path to the extracted shorelines (.geojson)
    """
    # Load extracted shorelines
    extracted_shorelines_lines = gpd.read_file(extracted_shorelines_lines_path)
    extracted_shorelines_lines = extracted_shorelines_lines.to_crs(crs)

    # Load KDE map
    kde = rioxarray.open_rasterio(kde_path).squeeze()

    # Optimize KDE sampling using `apply()`
    extracted_shorelines_lines['kde_value'] = extracted_shorelines_lines['geometry'].apply(
        lambda line: extract_mode_along_line(kde, line)
    )

    # Keep only necessary columns
    keep_columns = {'dates', 'dates_utc', 'image_suitability_score', 
                    'segmentation_suitability_score', 'simplify_param', 'year',
                    'geometry', 'kde_value', 'satname'}
    
    for col in extracted_shorelines_lines.columns:
        if col not in keep_columns:
            extracted_shorelines_lines = extracted_shorelines_lines.drop(columns=[col])

    # Convert back to WGS84 and save file
    extracted_shorelines_lines = utm_to_wgs84_df(extracted_shorelines_lines)
    extracted_shorelines_lines.to_file(extracted_shorelines_lines_path)

    return extracted_shorelines_lines_path

def convert_linestrings_to_multipoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert LineString geometries in a GeoDataFrame to MultiPoint geometries.

    Args:
    - gdf (gpd.GeoDataFrame): The input GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: A new GeoDataFrame with MultiPoint geometries. If the input GeoDataFrame
                        already contains MultiPoints, the original GeoDataFrame is returned.
    """

    # Check if all geometries in the gdf are MultiPoints
    if all(gdf.geometry.type == "MultiPoint"):
        return gdf

    def linestring_to_multipoint(linestring):
        if isinstance(linestring, LineString):
            return MultiPoint(linestring.coords)
        return linestring

    # Convert each LineString to a MultiPoint
    gdf["geometry"] = gdf["geometry"].apply(linestring_to_multipoint)

    return gdf
    
def resample_line_by_distance(line, distance):
    """
    Resamples points in shapely LineString

    inputs:
    line (shapely.LineString): the line to resample
    distance (m): resolution to sample the line at

    outputs:
    new_line (shapely.LineString): the resampled line
    """
    num_points = int(line.length / distance) + 1
    distances = np.linspace(0, line.length, num_points)
    points = [line.interpolate(distance) for distance in distances]
    if len(points)<3:
        new_line = None
    else:
        new_line = shapely.LineString(points)
    return new_line

def average_img(images_list):
    """
    Computes average array from a list of arrays

    inputs:
    images_list ([np.ndarray, ...]): list of numpy arrays

    outputs:
    arr (np.ndarray): the average array
    """
    images = np.array(images_list)
    arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)
    return arr

def chaikins_corner_cutting(coords, refinements=3):
    """
    Smooths out lines or polygons with Chaikin's method
    inputs:
    coords (list of tuples): [(x1,y1), (x..,y..), (xn,yn)]
    outputs:
    coords (list of tuples): [(x1,y1), (x..,y..), (xn,yn)],
                              this is the smooth line
    """
    i=0
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25
        i=i+1
    return coords

def check_nan_percentage(image_array):
    """
    Checks if more than 50% of the pixels in an RGB image array are NaN.

    Args:
        image_array (np.ndarray): An image array with shape (height, width, 3).

    Returns:
        bool: True if more than 50% of pixels are NaN, False otherwise.
    """
    # Create a boolean mask for NaNs across all color channels
    nan_mask = np.isnan(image_array)

    # A pixel is considered NaN if any of its R, G, or B channels are NaN.
    # We use .any(axis=2) to collapse the mask from (H, W, 3) to (H, W).
    nan_pixels = nan_mask.any(axis=2)
    
    # Count the number of pixels that have at least one NaN channel
    num_nan_pixels = np.sum(nan_pixels)
    
    # Get the total number of pixels in the image
    total_pixels = image_array.shape[0] * image_array.shape[1]
    
    # Calculate the percentage of NaN pixels
    nan_percentage = num_nan_pixels / total_pixels
    
    return nan_percentage > 0.5

def smooth_lines_constant(lines,simplify_param,refinements=2):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM (or another planar coordinate system)

    inputs:
    shorelines (gdf): gdf of extracted shorelines in UTM
    refinements (int): number of refinemnets for Chaikin's smoothing algorithm
    outputs:
    new_lines (gdf): gdf of smooth lines in UTM
    """
    lines = wgs84_to_utm_df(lines)
    lines['geometry'] = lines['geometry']
    new_geometries = [None]*len(lines)
    new_lines = lines.copy()
    for i in range(len(new_lines)):
        simplify_param = simplify_param
        line = new_lines.iloc[i]['geometry']
        line = line.simplify(simplify_param)
        coords = LineString_to_arr(line)
        refined = chaikins_corner_cutting(coords, refinements=refinements)
        refined_geom = arr_to_LineString(refined)
        new_geometries[i] = refined_geom
    new_lines['geometry'] = new_geometries
    new_lines = utm_to_wgs84_df(new_lines)
    return new_lines

def smooth_lines(lines,refinements=2):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM (or another planar coordinate system)

    inputs:
    shorelines (gdf): gdf of extracted shorelines in UTM
    refinements (int): number of refinemnets for Chaikin's smoothing algorithm
    outputs:
    new_lines (gdf): gdf of smooth lines in UTM
    """
    lines = wgs84_to_utm_df(lines)
    lines['geometry'] = lines['geometry']
    new_geometries = [None]*len(lines)
    new_lines = lines.copy()
    for i in range(len(new_lines)):
        simplify_param = new_lines['simplify_param'].iloc[i]
        line = new_lines.iloc[i]['geometry']
        line = line.simplify(simplify_param)
        coords = LineString_to_arr(line)
        refined = chaikins_corner_cutting(coords, refinements=refinements)
        refined_geom = arr_to_LineString(refined)
        new_geometries[i] = refined_geom
    new_lines['geometry'] = new_geometries
    return new_lines

def explode_multilinestrings(gdf):
    """
    Explodes any MultiLineString objects in a GeoDataFrame into individual LineStrings,
    and returns a new GeoDataFrame with these LineStrings replacing the original MultiLineStrings.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing various geometry types.

    Returns:
    GeoDataFrame: A new GeoDataFrame with MultiLineStrings exploded into LineStrings.
    """
    # Filter out MultiLineStrings
    multilinestrings = gdf[gdf['geometry'].type == 'MultiLineString']
    
    # Explode the MultiLineStrings, if any are present
    if not multilinestrings.empty:
        exploded_multilinestrings = multilinestrings.explode().reset_index(drop=True)
        
        # Remove original MultiLineStrings from the original DataFrame
        gdf = gdf[gdf['geometry'].type != 'MultiLineString']
        
        # Append exploded MultiLineStrings back to the original DataFrame
        final_gdf = pd.concat([gdf, exploded_multilinestrings], ignore_index=True)
    else:
        # No MultiLineStrings present, return original DataFrame unchanged
        final_gdf = gdf

    return final_gdf

def split_line(input_lines_or_multipoints_path,
               linestrings_or_multi_points,
               smooth=True):
    """
    Breaks up linestring into multiple linestrings if point to point distance is too high
    inputs:
    input_lines_or_multipoints_path (str): path to the output from output_gdf (
                                           extracted_shorelines_lines.geojson or extracted_shorelines_points.geojson)
    output_path (str): path to save output to (
                       extracted_shorelines_lines.geojson or extracted_shorelines_points.geojson)
    linestrings_or_multi_points (str): 'LineString' to make LineStrings, 'MultiPoint' to make MultiPoints
    smooth (bool): True to smooth the lines, False to not
    returns:
    output_path (str): path to the geodataframe with the new broken up lines
    """

    ##load shorelines, project to utm, get crs
    input_lines_or_multipoints = input_lines_or_multipoints_path
    input_lines_or_multipoints = wgs84_to_utm_df(input_lines_or_multipoints).dropna()
    source_crs = input_lines_or_multipoints.crs

    # Break any MultiLineStrings into individual LineStrings
    input_lines_or_multipoints = explode_multilinestrings(input_lines_or_multipoints)

    ##these lists are gonna hold the broken up lines and their simplified tolerance
    simplify_params = []
    all_lines = []
    for idx,row in input_lines_or_multipoints.iterrows():
        line = input_lines_or_multipoints[input_lines_or_multipoints.index==idx].reset_index(drop=True)

        ##setting distance threshold and simplify tolerance based on satellite
        satname = line['satname'].iloc[0]
        if (satname == 'L5') or (satname == 'L7') or (satname == 'L8') or (satname == 'L9'):
            dist_threshold = 45
            simplify_param = np.sqrt(30**2 + 30**2 + 30**2)/2
        elif (satname=='S2'):
            dist_threshold = 15
            simplify_param = np.sqrt(10**2 + 10**2 + 10**2)/2
        elif (satname=='PS'):
            dist_threshold = 8
            simplify_param = np.sqrt(5**2 + 5**2 + 5**2)/2

        column_names = list(line.columns)
        column_names.remove('geometry')
        points_geometry = [shapely.Point(x,y) for x,y in line['geometry'].iloc[0].coords]
        attributes = [[line[column_name].values[0]]*len(points_geometry) for column_name in column_names]
        input_coords_dict = dict(zip(column_names, attributes))
        input_coords_dict['geometry'] = points_geometry
        input_coords = gpd.GeoDataFrame(input_coords_dict, crs=source_crs)
        
        ##make the shifted geometries to compute point to point distance
        input_coords_columns = input_coords.columns[:]
        new_geometry_column = 'geom_2'
        input_coords[new_geometry_column] = input_coords['geometry'].shift(-1)

        ##compute distance
        def my_dist(in_row):
            return in_row['geometry'].distance(in_row['geom_2'])
        input_coords['dist'] = input_coords.loc[:input_coords.shape[0]-2].apply(my_dist, axis=1)
        ##break up line into multiple lines
        input_coords['break'] = (input_coords['dist'] > dist_threshold).shift(1)
        input_coords.loc[0,'break'] = True
        input_coords['line_id'] = input_coords['break'].astype(int).cumsum()

        ##make the lines
        def my_line_maker(in_grp):
            if len(in_grp) == 1:
                return list(in_grp)[0]
            elif linestrings_or_multi_points == 'LineString':
                return shapely.geometry.LineString(list(in_grp))
            elif linestrings_or_multi_points == 'MultiPoint':
                return shapely.geometry.MultiPoint(list(in_grp))
        new_lines_gdf = input_coords.groupby(['line_id']).agg({'geometry':my_line_maker}).reset_index()
        
        ##drop points and only keep linestrings
        new_lines_gdf['geom_type'] = [type(a) for a in new_lines_gdf['geometry']]
        new_lines_gdf = new_lines_gdf[new_lines_gdf['geom_type']!=shapely.Point].reset_index(drop=True)
        for column in column_names:
            new_lines_gdf[column] = [line[column].values[0]]*len(new_lines_gdf)
        new_lines_gdf = new_lines_gdf.drop(columns=['geom_type', 'line_id'])
        all_lines.append(new_lines_gdf)
        simplify_params.append(simplify_param)

    ##concatenate everything into one gdf, set geometry and crs
    all_lines_gdf = pd.concat(all_lines)
    all_lines_gdf['simplify_param'] = simplify_param
    #all_lines_gdf['dates'] = pd.to_datetime(all_lines_gdf['dates'], utc=True)
    #all_lines_gdf['year'] = all_lines_gdf['date'].dt.year
    all_lines_gdf = all_lines_gdf.set_geometry('geometry')
    all_lines_gdf = all_lines_gdf.set_crs(source_crs)

    ##smooth the lines
    if smooth==True:
        smooth_lines_gdf = smooth_lines(all_lines_gdf)

        ##put back in wgs84, save new file
        smooth_lines_gdf = utm_to_wgs84_df(smooth_lines_gdf)
        #smooth_lines_gdf.to_file(output_path)
        return smooth_lines_gdf

    else:
        ##put back in wgs84, save new file
        all_lines_gdf = utm_to_wgs84_df(all_lines_gdf)
        #all_lines_gdf.to_file(output_path)
        return all_lines_gdf
    
def LineString_to_arr(line):
    """
    Makes an array from linestring
    inputs:
    line (shapely.geometry.LineString): shapely linestring
    outputs:
    coords (List[tuples]): list of x,y coordinate pairs
    """
    listarray = []
    for pp in line.coords:
        listarray.append(pp)
    nparray = np.array(listarray)
    return nparray

def arr_to_LineString(coords):
    """
    Makes a line feature from an array of xy tuples
    inputs:
    coords (List[tuples]): list of x,y coordinate pairs
    outputs:
    line (shapely.geometry.LineString): shapely linestring
    """
    points = [None]*len(coords)
    i=0
    for xy in coords:
        points[i] = shapely.geometry.Point(xy)
        i=i+1
    line = shapely.geometry.LineString(points)
    return line

def contour_to_geo_coords(contour, xmin, ymax, xres, yres):
    """
    Converts a contour from skimage.measure.find_contours to geographic coordinates
    returns as a shapely LineString

    inputs:
    contour: output from find_contours
    xmin: minimum x coordinate of image
    ymax: maximum y coordinate of image
    xres: x resolution of image
    yres: y resolution of image

    outputs:
    line (shapely.LineString): the contour as a LineString in geographic coordinates
    """
    if len(contour)>1:
        points = [None]*len(contour)
        i=0
        for xy in contour:
            x = xy[1]
            y = xy[0]
            x = (x*xres+xmin)
            y = ymax-(y*yres)
            point = np.array([x, y])
            points[i] = point
            i=i+1
        nparray = np.array(points)
        line = arr_to_LineString(nparray)
    else:
        line = None
    return line

def extract_mode_along_line(xarr, line):
    """
    Profiles a raster with a line, returns mode of the profile
    """
    try:
        n_samples = 128

        # Generate equidistant points along the line in one step
        distances = np.linspace(0, 1, n_samples)
        points = [line.interpolate(dist, normalized=True) for dist in distances]

        # Extract pixel values in a vectorized manner
        x_coords = [point.x for point in points]
        y_coords = [point.y for point in points]
        values = xarr.sel(x=x_coords, y=y_coords, method="nearest").data
        # Round values and compute mode efficiently
        values = np.round(values, decimals=2)
    
        mode = stats.mode(values.ravel(), keepdims=True).mode[0]
    except:
        mode = 0

    return mode


def cross_distance(start_x, start_y, end_x, end_y):
    """distance formula, sqrt((x_1-x_0)^2 + (y_1-y_0)^2)"""
    dist = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
    return dist


def transect_timeseries(shorelines_path,
                        transects_path,
                        polygon_path,
                        crs,
                        output_merged_path,
                        output_mat_path):
    """
    Generates timeseries of shoreline cross-shore position
    given a geojson/shapefile containing shorelines and a
    geojson/shapefile containing cross-shore transects.
    Computes interesection points between shorelines
    and transects. Saves the merged transect timeseries.
    
    inputs:
    shoreline_path (str): path to file containing shorelines
    transect_path (str): path to file containing cross-shore transects
    output_merged path (str): path to save the merged csv file 
    output_mat_path (str): path to save the matrix csv file
    """
    # load transects, project to utm, get start x and y coords
    print('Loading transects, computing start coordinates')
    reference_polygon = gpd.read_file(polygon_path)
    joined_mats = [None]*len(reference_polygon)
    joined_dfs = [None]*len(reference_polygon)
    for idx,row in reference_polygon.iterrows():
        transects_gdf = gpd.read_file(transects_path, mask=row.geometry)
        transects_gdf = transects_gdf.to_crs(crs)
        crs = transects_gdf.crs
        transects_gdf = transects_gdf.reset_index(drop=True)
        transects_gdf['geometry_saved'] = transects_gdf['geometry']
        coords = transects_gdf['geometry_saved'].get_coordinates()
        coords = coords[~coords.index.duplicated(keep='first')]
        transects_gdf['x_start'] = coords['x']
        transects_gdf['y_start'] = coords['y']
        gc.collect()
        
        # load shorelines, project to utm, smooth
        shorelines_gdf = gpd.read_file(shorelines_path, mask=row.geometry)
        shorelines_gdf = shorelines_gdf.to_crs(crs)

        # join all the shorelines that occured on the same date together
        shorelines_gdf = shorelines_gdf.dissolve(by='dates')
        shorelines_gdf = shorelines_gdf.reset_index()
        gc.collect()

        print('computing intersections')
        # spatial join shorelines to transects
        joined_gdf = gpd.sjoin(shorelines_gdf, transects_gdf, predicate='intersects')
        
        # get points, keep highest cross distance point if multipoint (most seaward intersection)
        joined_gdf['intersection_point'] = joined_gdf.geometry.intersection(joined_gdf['geometry_saved'])
        # reset the index because running the intersection function changes the index
        joined_gdf = joined_gdf.reset_index(drop=True)

        for i in range(len(joined_gdf['intersection_point'])):
            point = joined_gdf['intersection_point'].iloc[i]
            start_x = joined_gdf['x_start'].iloc[i]
            start_y = joined_gdf['y_start'].iloc[i]
            if type(point) == shapely.MultiPoint:
                points = [shapely.Point(coord) for coord in point.geoms]
                points = gpd.GeoSeries(points, crs=crs)
                coords = points.get_coordinates()
                dists = [None]*len(coords)
                for j in range(len(coords)):
                    dists[j] = cross_distance(start_x, start_y, coords['x'].iloc[j], coords['y'].iloc[j])
                max_dist_idx = np.argmax(dists)
                last_point = points[max_dist_idx]
            # This new line  updates the shoreline at index (i) for a single value. We only want to update a single shoreline
                joined_gdf.at[i, 'intersection_point'] = last_point # is only edits a single intersection point

        # get x's and y's for intersections
        intersection_coords = joined_gdf['intersection_point'].get_coordinates()
        joined_gdf['intersect_x'] = intersection_coords['x']
        joined_gdf['intersect_y'] = intersection_coords['y']
        
        # get cross distance
        joined_gdf['cross_distance'] = cross_distance(joined_gdf['x_start'], 
                                                    joined_gdf['y_start'], 
                                                    joined_gdf['intersect_x'], 
                                                    joined_gdf['intersect_y'])
        ##clean up columns
        joined_gdf = joined_gdf.rename(columns={'date':'dates'})
        keep_columns = ['dates','satname','geoaccuracy','cloud_cover','transect_id',
                        'intersect_x','intersect_y','cross_distance', 
                        'image_suitability_score', 'segmentation_suitability_score', 'kde_value',
                        'year', 'dates_utc']

        # convert the x and y intersection points to the final crs (4326) to match the rest of joined_df
        points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(joined_gdf['intersect_x'], joined_gdf['intersect_y']),crs=crs)
        points_gdf = points_gdf.to_crs('epsg:4326')

        # you have to reset the index here otherwise the intersection point won't match the row correctly
        # recall that the shorelines were group by dates and that changed the index
        joined_gdf = joined_gdf.rename(columns={'date':'dates'}).reset_index(drop=True)

        joined_gdf['intersect_x'] = points_gdf.geometry.x
        joined_gdf['intersect_y'] = points_gdf.geometry.y

        # convert the joined_df back to CRS 4326
        joined_gdf = utm_to_wgs84_df(joined_gdf)

        for col in joined_gdf.columns:
            if col not in keep_columns:
                joined_gdf = joined_gdf.drop(columns=[col])

        joined_df = joined_gdf.reset_index(drop=True)
        joined_df = joined_df.drop_duplicates(subset=['dates', 'transect_id', 'satname'], keep='first')
        ##pivot to make the matrix
        joined_mat = joined_df.pivot(index='dates', columns='transect_id', values='cross_distance')
        joined_mat.columns.name = None
        joined_dfs[idx] = joined_df
        joined_mats[idx] = joined_mat

    joined_mat = pd.concat(joined_mats)
    joined_df = pd.concat(joined_dfs)

    #joined_mat.to_csv(output_mat_path)
    
    ##save file
    joined_df.to_csv(output_merged_path,index=False)
    print('intersections computed')

def scale_percentile(image_array, lower_percentile=2, upper_percentile=98):
    """Scales an image's pixel values to a new range using percentiles."""
    # Find the min and max values based on the specified percentiles
    min_val = np.nanpercentile(image_array, lower_percentile)
    max_val = np.nanpercentile(image_array, upper_percentile)

    # Clip values to the new range
    clipped_array = np.clip(image_array, min_val, max_val)

    # Perform min-max scaling on the clipped data
    scaled_array = ((clipped_array - min_val) / (max_val - min_val))
    return scaled_array

def rescale(arr):
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
    
def moving_average_with_boundary_averaging(data, window_size):
    """Calculates moving average, averaging partial windows at boundaries."""
    weights = np.repeat(1.0, window_size) / window_size
    ma_valid = np.convolve(data, weights, 'valid')

    # For left boundary:
    ma_left = []
    for i in range(window_size - 1):
        ma_left.append(np.mean(data[0:i+1]))

    # For right boundary
    ma_right = []
    for i in range(window_size-1, 0, -1):
        ma_right.append(np.mean(data[-i:]))

    return np.concatenate([ma_left, ma_valid, ma_right])
    
def moving_average_with_edge_padding(data, window_size):
    """Calculates the moving average with edge padding to maintain original length."""
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode='edge')
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(padded_data, weights, 'valid')

def resize_image_numpy(image_array, new_size):
  """
  Resizes a NumPy array representing an image.

  Args:
    image_array: A NumPy array representing the image (e.g., shape (height, width, channels) or (height, width)).
    new_size: A tuple (new_width, new_height) specifying the desired size.

  Returns:
    A NumPy array of the resized image.
  """
  image = Image.fromarray(image_array.astype(np.uint8))
  resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
  return np.array(resized_image)

def remove_nones(my_list):
    """
    removes nans from numpy array

    inputs:
    my_list (list): list with nones

    outputs:
    new_list (list): list without nones
    """
    new_list = [x for x in my_list if x is not None]
    return new_list

# @profile
# def get_contours(seg_lab,
#                  satname,
#                  xmin,
#                  ymax,
#                  x_res,
#                  y_res,
#                  data_polygon,
#                  reference_shoreline_gdf,
#                  ref_shore_buffer,
#                  reference_polygon,
#                  data_mask,
#                  crs):
#     """
#     Getting the contours from a two-class segmented image

#     inputs:
#     seg_lab (np.ndarray): the two-class segmented output from the Zoo model
#     rgb (np.ndarray): the rgb image that was segmented
#     detect_path (str): path to save an image to with the shoreline and segmentation over the rgb image
#     satname (str): which satellite the image came from ('L5', 'L7', 'L8', 'L9', 'S2', 'PS')
#     xmin (float): minimum coordinate of image
#     ymax (float): maximum coordinate of image
#     x_res (float): x resolution of image
#     y_res (float): y resolution of image
#     data_polygon (gpd.GeoDataFrame): polygon representing where image data is
#     reference_shoreline_gdf (gpd.GeoDataFrame): reference shoreline as gdf
#     ref_shore_buffer (float): buffer for reference shoreline
#     reference_polygon (str): path to the reference polygon
#     crs: coordinate reference system
#     data_mask (np.ndarray): array holding where data values are in the image

#     outputs:
#     linestrings (List[shapely.LineString, ...]): list of the contours as LineStrings in geographic coordinates
#     """
#     ##follow contours
#     contours = measure.find_contours(seg_lab, 0.5, mask=data_mask)
#     ##get rid of short and extra contours
#     #contour_filtered = max(contours, key=len)


#     ##clip 50 m off each end
#     clip_length=50
#     if satname == 'S2':
#         clip_units = int(clip_length/10)
#     elif satname == 'PS':
#         clip_units = int(clip_length/5)
#     else:
#         clip_units = int(clip_length/30)

#     # Convert contours to geospatial LineStrings
#     features = [
#         {"id": idx, "type": "Feature", "properties": {},
#          "geometry": contour_to_geo_coords(contour[clip_units:-clip_units], xmin, ymax, x_res, y_res)}
#         for idx, contour in enumerate(contours) if len(contour) > 5
#     ]
#     features = remove_nones(features)
#     if features == []:
#         return []
#     ##dummy_gdf for breaking up multilinestrings, clip it to the data polygon 
#     dummy_gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
#     dummy_gdf = gpd.GeoDataFrame({'id':[0],
#                                   'geometry':dummy_gdf.unary_union},crs=crs)
#     dummy_gdf = gpd.clip(dummy_gdf, data_polygon)
#     # dummy_gdf = gpd.clip(dummy_gdf, reference_shoreline_gdf['geometry'].iloc[0].buffer(ref_shore_buffer))
#     # if reference_polygon!=None:
#     #     reference_polygon_gdf = gpd.read_file(reference_polygon)
#     #     reference_polygon_gdf = wgs84_to_utm_df(reference_polygon_gdf)
#     #     dummy_gdf = gpd.clip(dummy_gdf, reference_polygon_gdf)
#     dummy_gdf = dummy_gdf.explode()
#     dummy_gdf['type'] = dummy_gdf['geometry'].type
#     dummy_gdf = dummy_gdf[dummy_gdf['type']=='LineString']
#     linestrings = [line for line in dummy_gdf['geometry']]
#     return linestrings

def get_contours(seg_lab,
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
                 crs):
    """
    Extract contours from a two-class segmented image and convert to geographic LineStrings.
    """

    contours = measure.find_contours(seg_lab, 0.5, mask=data_mask)

    clip_map = {'S2': 10, 'PS': 5}
    clip_units = int(50 / clip_map.get(satname, 30))  # default for L5–L9

    features = [
        {
            "id": idx,
            "type": "Feature",
            "properties": {},
            "geometry": contour_to_geo_coords(contour[clip_units:-clip_units], xmin, ymax, x_res, y_res)
        }
        for idx, contour in enumerate(contours) if len(contour) > 5
    ]
    features = remove_nones(features)
    if not features:
        return []

    # Wrap clipped features and process geometry
    dummy_gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    dummy_gdf = gpd.GeoDataFrame(
        {'id': [0], 'geometry': [dummy_gdf.unary_union]},
        crs=crs
    )
    dummy_gdf = gpd.clip(dummy_gdf, data_polygon)
    dummy_gdf = dummy_gdf.explode(index_parts=False)
    dummy_gdf = dummy_gdf[dummy_gdf.geometry.type == 'LineString']
    dummy_gdf['len'] = dummy_gdf.geometry.apply(lambda geom: len(geom.coords))
    dummy_gdf = dummy_gdf[dummy_gdf['len']>1]

    return list(dummy_gdf.geometry)

def clip_extracted_shoreline(shoreline_geojson, data_polygon, reference_shoreline, reference_polygon, ref_shore_buffer):
    """
    clips extracted shoreline from single image
    inputs:
    shoreline_geojson (str): path to the extracted shoreline geojson
    data_polygon (gpd.GeoDataFrame): polygon representing where to keep data
    reference_shoreline (str): path to the reference shoreline geojson
    reference_polygon (str): path to the reference polygon geojson
    ref_shore_buffer (float): reference shoreline buffer radius
    """
    reference_shoreline_gdf = gpd.read_file(reference_shoreline)
    reference_polygon_gdf = gpd.read_file(reference_polygon)
    shoreline_gdf = gpd.read_file(shoreline_geojson)

    reference_shoreline_gdf = wgs84_to_utm_df(reference_shoreline_gdf)
    reference_polygon_gdf = wgs84_to_utm_df(reference_polygon_gdf)
    shoreline_gdf = wgs84_to_utm_df(shoreline_gdf)

    if data_polygon!=None:
        shoreline_gdf = gpd.clip(shoreline_gdf, data_polygon.unary_union)
    shoreline_gdf = gpd.clip(shoreline_gdf, reference_shoreline_gdf['geometry'].iloc[0].buffer(ref_shore_buffer))
    shoreline_gdf = gpd.clip(shoreline_gdf, reference_polygon_gdf)

    shoreline_gdf = shoreline_gdf.explode()
    shoreline_gdf = utm_to_wgs84_df(shoreline_gdf)
    shoreline_gdf['type'] = shoreline_gdf['geometry'].type
    shoreline_gdf = shoreline_gdf[shoreline_gdf['type']=='LineString']

    if len(shoreline_gdf)>0:
        shoreline_gdf.to_file(shoreline_geojson)

def reference_shoreline_to_rois(reference_shoreline_path, reference_polygon_path, rois_path, distance=3000):
    """
    Computes ROIs from reference shoreline.
    Resamples the reference shoreline to a point every distance meters.
    Creates a square buffer at each point with radius distance meters.
    Saves these as the ROIs to a geojson

    inputs:
    reference_shoreline_path (str): path to the reference shoreline (.geojson)
    rois_path (str): path to save the rois to
    distance (float, m): radius of roi and resampling distance for reference shoreline

    outputs:
    rois_path (str): path to the ROIs
    """
    reference_shoreline = gpd.read_file(reference_shoreline_path)
    reference_polygon = gpd.read_file(reference_polygon_path)
    reference_polygon = gpd.GeoSeries(reference_polygon['geometry'], crs=reference_polygon.crs).unary_union
    centroid = reference_polygon.centroid
    centroid_gdf = gpd.GeoDataFrame({'id':[0]}, geometry=[centroid], crs=reference_shoreline.crs)
    centroid_gdf = wgs84_to_utm_df(centroid_gdf)
    square_buffer = centroid_gdf.buffer(distance, cap_style='square')
    square_buffer = utm_to_wgs84_df(square_buffer)
    if square_buffer.contains(reference_polygon).iloc[0]==True:
        final_rois = square_buffer
        final_rois.to_file(rois_path)
    else:
        ##if there are multiple ref shorelines, we merge into one
        reference_shoreline['OBJECTID'] = list(range(len(reference_shoreline)))
        ref_shore = reference_shoreline.sort_values('OBJECTID',ascending=True).reset_index()
        points = []
        for shore in ref_shore['geometry']:
            for point in shore.coords:
                points.append(point)
        ref_shore_real = shapely.LineString(points)
        reference_shoreline.at[0,'geometry'] = [ref_shore_real]
        reference_shoreline = reference_shoreline.loc[[0]]
        reference_shoreline = wgs84_to_utm_df(reference_shoreline)
        crs = reference_shoreline.crs
        reference_shoreline.at[0,'geometry'] = resample_line_by_distance(reference_shoreline['geometry'].iloc[0], 
                                                                        distance+distance/4)
        reference_shoreline = convert_linestrings_to_multipoints(reference_shoreline)
        
        reference_shoreline = reference_shoreline.explode(index_parts=True)
        buffers = [None]*len(reference_shoreline)
        for i in range(len(buffers)):
            gdf = reference_shoreline.iloc[i]
            gdf = gpd.GeoSeries(reference_shoreline.iloc[i]['geometry'], crs=crs)
            square_buffer = gdf.buffer(distance, cap_style='square')
            buffers[i] = square_buffer
        final_rois = pd.concat(buffers)
        final_rois = utm_to_wgs84_df(final_rois)
        final_rois.to_file(rois_path)
        return rois_path

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

def vertex_filter(shorelines):
    """
    Recursive 3-sigma filter on vertices in shorelines
    Will filter out shorelines that have too many or too few
    vertices until all of the shorelines left in the file are within
    Mean+/-3*std
    
    Saves output to the same directory with same name but with (_vtx) appended.

    inputs:
    shorelines (str): path to the extracted shorelines geojson
    outputs:
    new_path (str): path to the filtered file 
    """
    gdf = shorelines
    
    count = len(gdf)
    new_count = None
    for index, row in gdf.iterrows():
        gdf.at[index,'vtx'] = len(row['geometry'].coords)/row['geometry'].length
    filter_gdf = gdf.copy()

    while count!=new_count:
        count = len(filter_gdf)
        sigma = np.std(filter_gdf['vtx'])
        mean = np.mean(filter_gdf['vtx'])

        high_limit = mean+3*sigma
        low_limit = mean-3*sigma
        filter_gdf = gdf[gdf['vtx']< high_limit]
        filter_gdf = filter_gdf[filter_gdf['vtx']> low_limit]
        new_count = len(filter_gdf)
    filter_gdf = filter_gdf.reset_index(drop=True)

    return filter_gdf