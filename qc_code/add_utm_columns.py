import os
import geopandas as gpd
from pyproj import Transformer

def add_utm_attributes_to_transects(input_gpkg,
                                    layer_name='01_transects_attributes',
                                    output_gpkg=None):
    """
    Adds UTM-based attributes to transect LineStrings stored in WGS84.
    
    Creates these new fields:
        - utm_zone_epsg
        - utm_start_x, utm_start_y
        - utm_end_x, utm_end_y
        - utm_midpoint_x, utm_midpoint_y
        
    Parameters
    ----------
    input_gpkg : str
        Path to the input GeoPackage.
    layer_name : str
        Name of the layer containing transect LineStrings.
    output_gpkg : str or None
        If None, overwrites the input file. Otherwise writes to new GPKG.

    Returns
    -------
    GeoDataFrame
        The modified GeoDataFrame with new UTM fields.
    """

    gdf = gpd.read_file(input_gpkg, layer=layer_name)

    # Force CRS to WGS84 if missing or incorrect
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Helper: compute UTM EPSG code from latitude & longitude
    def utm_epsg(lat, lon):
        zone = int((lon + 180) // 6) + 1
        return 32600 + zone if lat >= 0 else 32700 + zone

    # Prepare output columns
    gdf['utm_zone_epsg'] = None
    gdf['utm_start_x'] = None
    gdf['utm_start_y'] = None
    gdf['utm_end_x'] = None
    gdf['utm_end_y'] = None
    gdf['utm_midpoint_x'] = None
    gdf['utm_midpoint_y'] = None

    for idx, geom in gdf.geometry.items():

        # Extract coordinates
        x0, y0 = geom.coords[0]
        x1, y1 = geom.coords[-1]
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2

        # Determine UTM zone from midpoint
        epsg = utm_epsg(ym, xm)
        transformer = Transformer.from_crs(4326, epsg, always_xy=True)

        # Transform
        sx, sy = transformer.transform(x0, y0)
        ex, ey = transformer.transform(x1, y1)
        mx, my = transformer.transform(xm, ym)

        # Store results
        gdf.at[idx, 'utm_zone_epsg'] = epsg
        gdf.at[idx, 'utm_start_x'] = sx
        gdf.at[idx, 'utm_start_y'] = sy
        gdf.at[idx, 'utm_end_x'] = ex
        gdf.at[idx, 'utm_end_y'] = ey
        gdf.at[idx, 'utm_midpoint_x'] = mx
        gdf.at[idx, 'utm_midpoint_y'] = my

    gdf['utm_zone_epsg'] = gdf['utm_zone_epsg'].astype(int)
    gdf['utm_start_x'] = gdf['utm_start_x'].astype(float)
    gdf['utm_start_y'] = gdf['utm_start_y'].astype(float)
    gdf['utm_end_x'] = gdf['utm_end_x'].astype(float)
    gdf['utm_end_y'] = gdf['utm_end_y'].astype(float)
    gdf['utm_midpoint_x'] = gdf['utm_midpoint_x'].astype(float)
    gdf['utm_midpoint_y'] = gdf['utm_midpoint_y'].astype(float)
    # Write output
    if output_gpkg is None:
        # Overwrite existing file
        gdf.to_file(input_gpkg, layer=layer_name, driver='GPKG')
    else:
        gdf.to_file(output_gpkg, layer=layer_name, driver='GPKG')

    return gdf


# input_gpkg = os.path.join('/', 'mnt', 'f', 'SDSDataService_c_qc2', '14_index.gpkg')
# add_utm_attributes_to_transects(input_gpkg,
#                                 layer_name='01_transects_attributes',
#                                 output_gpkg=input_gpkg)