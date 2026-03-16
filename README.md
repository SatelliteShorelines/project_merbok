# project_merbok

Batch processing code for CoastSeg in northern Alaska. Utilizes deep learning based filtering models, segmentation models, and additional post-processing steps for mapping shorelines and reducing noise.

Setup:

1) Install wsl/miniforge

2) Create conda environment

```conda env create -f project_merbok_full.yaml```

3) Setup shoreline sections (GCRRSSS) with reference shorelines and reference polygons.

Directory strucure 

Shown below is an the example for a shoreline section from:

Global Region G1 (US)

Coastal Area C4 (Alaska)

Subregion RR00 (Beaufort Sea coast)

Example processed Shoreline Section 1400000 (first section at Canadian border)

root
----G1
------C4
--------RR00
------------SSS000

This will contain the following files:

* 1400000_ms_lists (lookup tables for downloaded tiffs)
* DEMs/Arctic_DEM (2 m DEM tiff)
* elevation_profiles_ (pngs and csvs of elevation profiles with estimated foreshore slopes)
* elevation_profile_lines_ (geojsons of elevation contours)
* ms_tiff_paths/L5/pansharpened/coregistered (pansharpened and coregistered L5 imagery)
* ms_tiff_paths/L7/pansharpened (pansharpend L7 imagery)
* ms_tiff_paths/L8/pansharpened/coregistered (pansharpened and coregistered L8 imagery)
* ms_tiff_paths/L9/pansharpened/coregistered (pansharpened and coregistered L9 imagery)
* ms_tiff_paths/S2/pansharpened/coregistered (pansharpened and coregistered S2 imagery)
* ms_tiff_paths/PS (PlanetScope imagery)
* segmentation (4-class segmentation outputs, Water, Whitewater, Sand, Other)
* 1400000_extracted_shorelines.geojson (waterlines from Zoo RGB model)
* 1400000_extracted_shorelines_filter.geojson (filtered waterlines from Zoo RGB model)
* 1400000_extracted_shorelines_nir_thresh.geojson (waterlines from NIR Otsu threshold)
* 1400000_extracted_shorelines_nir_thresh_filter.geojson (filtered waterlines from NIR Otsu threshold)
* 1400000_extracted_shorelines_swir_thresh.geojson (waterlines from SWIR threshold)
* 1400000_extracted_shorelines_swir_thresh_filter.geojson (filtered waterlines from SWIR threshold)
* 1400000_filtered_tidally_corrected_points.geojson (instantaneous shoreline points after ensembling and automated timeseries filtering as geojson)
* 1400000_filtered_tidally_corrected_transect_time_series_merged.csv (instantaneous shoreline points after ensembling and automated timeseries filtering as csv)
* 1400000_max_shoreline.geojson (record maximum shoreline from resampled record (annual for Alaska data))
* 1400000_max_shoreline_points.geojson (record maximum shoreline as points form resampled record (annual for Alaska))
* 1400000_mean_shoreline.geojson (record mean shoreline from resampled record (annual for Alaska))
* 1400000_mean_shoreline_points.geojson (record mean shoreline as points from resampled record (annual for Alaska))
* 1400000_median_shoreline.geojson (record median shoreline from resampled record (annual for Alaska))
* 1400000_median_shoreline_points.geojson (record median shoreline as points from resampled record (annual for Alaska))
* 1400000_min_shoreline.geojson (record minumum shoreline from resampled record (annual for Alaska))
* 1400000_min_shoreline_points.geojson (record minimum shoreline as points from resampled record (annual for Alaska))
* 1400000_q1_shoreline.geojson (record 25th percentile shoreline from resampled record (annual for Alaska))
* 1400000_q1_shoreline_points.geojson (record 25th percentile shoreline as points from resampled record (annual for Alaska))
* 1400000_q3_shoreline.geojson (record 75th percentile shoreline from resampled record (annual for Alaska))
* 1400000_q3_shoreline_points.geojson (record 75th percentile shoreline as points from resampled record (annual for Alaska))
* 1400000_raw_transect_time_series_merged.csv (Zoo RGB waterline intersections as dataframe (each row is an entry, each column is an attribute))
* 1400000_raw_transect_time_series_merged_nir_thresh.csv (NIR waterline intersections as dataframe (each row is an entry, each column is an attribute)) 
* 1400000_raw_transect_time_series_merged_swir_thresh.csv (SWIR waterline intersections as dataframe (each row is an entry, each column is an attribute)) 
* 1400000_reference_polygon.geojson (reference polygon to limit detected waterlines to)
* 1400000_reference_shoreline.geojson (reference shoreline for generating transects and ROIs)
* 1400000_reprojected_points.geojson (resampled data as points geojson)
* 1400000_resampled_tidally_corrected_transect_time_series_matrix.csv (resampled data as a matrix (each column is a transect, each cell is a cross distance))
* 1400000_resampled_tidally_corrected_transect_time_series_merged.csv (resampled data as a dataframe (each column is an attribute, each row is an entry))
* 1400000_rois.geojson (ROIs for downloading imagery)
* 1400000_slopes_.csv (foreshore slopes)
* 1400000_spatial_kde.tif (density map from waterlines)
* 1400000_spatial_kde_otsu.geojson (polygon from Otsu threshold on density map)
* 1400000_spatial_kde_otsu.tif (binarized density map)
* 1400000_tidally_corrected_transect_time_series_mat.csv (Zoo RGB tidally corrected shoreline points as matrix (each column is a transect, each cell is a cross distance))
* 1400000_tidally_corrected_transect_time_series_mat_nir_thresh.csv (NIR tidally corrected shoreline points as matrix (each column is a transect, each cell is a cross distance))
* 1400000_tidally_corrected_transect_time_series_mat_swir_thresh.csv (SWIR tidally corrected shoreline points as matrix (each column is a transect, each cell is a cross distance))
* 1400000_tidally_corrected_transect_time_series_merged.csv (Zoo RGB tidally corrected shoreline points as dataframe (each column is an attribute, each row is an entry))
* 1400000_tidally_corrected_transect_time_series_merged_nir_thresh.csv (NIR tidally corrected shoreline points as dataframe (each column is an attribute, each row is an entry))
* 1400000_tidally_corrected_transect_time_series_merged_swir_thresh.csv (SWIR tidally corrected shoreline points as dataframe (each column is an attribute, each row is an entry))
* 1400000_tides.csv (FES22 tide data)
* 1400000_transects.geojson (shore-normal transects, spaced 50-m apart alongshore, seaward oriented)
* 1400000_transects_attributes.geojson (shore-normal transects, spaced 50-m apart alongshore, seaward oriented, joined with foreshore slopes and shore type) 
* 1400000_transects_slopes_.geojson (shore-normal transects, spaced 50-m apart alongshore, seaward oriented, joined with foreshore slopes)
* 1400000_transects_trends.geojson (shore-normal transects, spaced 50-m apart alongshore, seaward oriented, joined with trends computed on resampled data, geometry altered to have transect length proportional to trend estimate and direction oriented seaward for postive trends and shoreward for negative trends)
* 1400000_unfiltered_tidally_corrected_points.geojson (instantaneous shoreline points after ensembling as geojson)
* 1400000_unfiltered_tidally_corrected_transect_time_series_merged.csv (instantaneous shoreline points after ensembling as csv)

General Workflow:

0. Set up WSL and miniforge and the conda environment necessary to run this software. This can be challenging as dependencies change with time and individual computer hardware differs. LLMs can be very helpful when troubleshooting at this stage. My process is to request a conda .yaml environment file that finds compatible versions of each package I want to work together. I then make the conda environment and test that every libary can load and that tensorflow can see GPUs. If the machine I am working on gets errors, I give that complete error message to the LLM and prompt it to provide a new conda environment solution. My experience is that at first the LLM will try to offer ad hoc solutions. Push back on these ad hoc solutions. With persistence, a path to a working environment can usually be solved. 

1. Manually create a reference shoreline and a reference polygon. This should be a single feature for a shoreline section in WGS84. Using alternative coordinate systems (UTM, state plane, etc.) will not work with this software. If a large area is to be mapped, make sure the direction of each reference shoreline is consistent. Otherwise later on when analyzing data from the transects you will have trouble if the indexing/order of transects switches back and forth to each shoreline section. Just make this in a GIS with a satellite imagery basemap. There are automated ways to make the reference polygon (using the spatial density of unfiltered points from a shoreline mapping run or from applying a buffer to the reference shoreline). Otherwise, make it manually. Use site-specific knowledge to design this intelligently. Cycle through the Google Earth timelapse to see quickly what the coast has been doing in that section. Or download some satellite images and use that to guide how you draw the polygon. 

2. Generate transects from the reference shoreline. Then check that they look OK. Fix them if they cross or if they are not oriented the way you would prefer. Sometimes, around irregular coastalines, the indexing can get out of order. There is a function in transects/generate_transects.py that can help with this (re_index_with_ref_shoreline). Take a look at the logic of that function to understand how it works (varying smoothing and sampling resolution).

3. Generate square ROIs from the reference shoreline. Check these ROIs to make sure they look correct. Adjust if needed. These are generated as square buffers around decimated versions of the reference shoreline. They are square to avoid distortions when reized for the deep learning models. They are centered on the reference shoreline to avoid inclusion of unfamiliar terrain for the deep learning models. 

4. Download satellite imagery with ROIs through CoastSeg. 

5. Apply image suitability filter.

6. Reorganize tiffs. This system finds all of the suitable tiffs that intersect the ROIs from CoastSeg/data, and it moves these tiffs into the shoreline section directory.

7. Apply pansharpening and coregistration. Landsat 7 and Planet imagery are not coregistered with this software. The effects of these two alogirthms bring subtle improvements. 

8. Apply segmentation models and thresholding on the tiffs.

9. Apply segmentation suitability model on 4-class segmentation pngs. This model has limited effectiveness from large-scale application in Alaska, but was useful in the beginning before better filtering methods evolved.

10. Post-process data (compute spatial KDE, get FES22 tide data, get waterline intersections, tidally correct waterline intersections, ensemble data, filter data, resample data, compute trends from resampled data).

11. Compute record statistics (minimum, q1, mean, median, q3, maximum shorelines).

12. Create tiered geopackages (Index (transects, ROIs, reference shorelines, reference polygons, foreshore slopes, shore type), Tier_0 (waterlines and unfiltered data), Tier_1 (instantaneous ensembled data), Tier_2 (resampled data), Tier_3 (record statistics)).

13. Take a loot at qc_code for some functions made to check all of the data that was generated.

14. Use composite_analysis.py to create composite tiffs on an annual or decadal basis. These reduce the imagery density and remove much of the noise in the instantaneous images, making them easier to interpret as well as analyze for planform changes. Thresholding on the NIR/SWIR bands will work well on these.

Some notes:

* After segmentation, each water/land classification is saved to extra bands to the processed imagery
* ms_tiff_paths/L5/pansharpened/coregistered (band 6 = Zoo RGB output, band 7 = NIR Otsu threshold output, band 8 = SWIR Otsu threshold output)
* ms_tiff_paths/L7/pansharpened (band 6 = Zoo RGB output, band 7 = NIR Otsu threshold output, band 8 = SWIR Otsu threshold output)
* ms_tiff_paths/L8/pansharpened/coregistered (band 6 = Zoo RGB output, band 7 = NIR Otsu threshold output, band 8 = SWIR Otsu threshold output)
* ms_tiff_paths/L9/pansharpened/coregistered (band 6 = Zoo RGB output, band 7 = NIR Otsu threshold output, band 8 = SWIR Otsu threshold output)
* ms_tiff_paths/S2/pansharpened/coregistered (band 6 = Zoo RGB output, band 7 = NIR Otsu threshold output, band 8 = SWIR Otsu threshold output)
* ms_tiff_paths/PS/pansharpened (band 5 = Zoo RGB output, band 6 = NIR Otsu threshold output)
* If you want to use a custom DEM, make a folder in DEMs/MY_CUSTOM_DEM
* Check the foreshore slopes. Smoothing is usually necessary to remove noise and outliers when computing from a DEM. For very small slopes, set a floor to 0.01. Anything below that will start projecting the shoreline points too far.
* There are assumptions on the shape of the coast that go into the formulation of the foreshore slope computation that are not always compatible with the actual shape of the coast. Other times the DEM is not perfect. 
* Another option is using CoastSat.Slope. This can be a useful tool when satellite imagery is dense and the foreshore slopes are in a specified range (see the original paper from Vos et al.).  
* Something to keep in mind is that assigning a single slope to a transect is an obvious simplification that makes the math easier and scalable. There could be better ways of tidally correcting data and estimating foreshore slopes, but many times these methods/corrections are too particular for small areas and do not generalize well.
* If wave data is available, and data is dense enough in time, look into correcting for wave runup with the Stockdon equation.
* If you have data to compare, try the validation function. See the code to understand how to set up your data. Currently the code compares satellite shorelines that are within 10-days of a comparison measurement to compute "errors". Be careful with language when reporting this as an error and recognize that different techniques have costs and benefits, cover different spatial and temporal scales, and answer different questions.
* Last, review the Shannon-Nyquist theorem and the concept of aliasing before analyzing data. There are limited things one can say depending on how a natural process was sampled. Do not resample finer than what the raw data can provide.
* Look at the raw satellite imagery, segmentation outputs, and shoreline data in map form when analyzing the timeseries data. The actual imagery is usually the most obvious record of what was actually going on and whether or not the estimated shoreline is accurate. 

Running the batch processing software

1) Configure settings with config_gui.py. This will save a config.yaml and then run the merbok_workflow script.

```python config_gui.py```

2) Run various functions with merbok_workflow.py

```python merbok_workflow.py --config config.yaml```

