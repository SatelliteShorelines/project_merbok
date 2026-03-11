
from osgeo import gdal

def delete_gpkg_layer(gpkg_path: str, layer_name: str) -> bool:
    """
    Delete a layer from a GeoPackage using GDAL/OGR.
    Returns True if deleted, False if the layer was not found.
    Raises RuntimeError on other failures (e.g., permission).
    """
    # Open for update (read-write), vector only
    ds = gdal.OpenEx(gpkg_path, gdal.OF_VECTOR | gdal.OF_UPDATE)
    if ds is None:
        raise RuntimeError(f"Cannot open GeoPackage for update: {gpkg_path}")

    # Quick existence check
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        print(f"Layer '{layer_name}' not found in {gpkg_path}")
        return False

    # Delete the layer (GDAL cleans up gpkg metadata correctly)
    err = ds.DeleteLayer(layer_name)
    if err != gdal.CE_None:
        raise RuntimeError(f"DeleteLayer failed for '{layer_name}' (code {err})")

    print(f"Deleted layer '{layer_name}' from {gpkg_path}")
    return True
