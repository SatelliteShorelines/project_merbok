
import geopandas as gpd
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def recompute_tier1_cross_distance(
    tier_0_gpkg_path: str,
    tier_1_gpkg_path: str,
    write_back: bool = False,
    output_path: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Recompute Tier-1 cross_distance and ci using Tier-0 RGB/NIR/SWIR cross_distance_tidally_corrected
    values, matched by ['dates', 'satname'].

    Tier-0 layers (required):
        - '02_zoo_rgb_time_series'
        - '05_nir_time_series'
        - '08_swir_time_series'
      Columns required: 'dates', 'satname', 'cross_distance_tidally_corrected'

    Tier-1 layers (updated):
        - '00_unfiltered_shoreline_points'
        - '01_shoreline_points'
      Will contain/overwrite columns:
        - 'cross_distance'            (median across RGB/NIR/SWIR)
        - 'ci'                        (q90 - q10 across RGB/NIR/SWIR)
        - 'cross_distance_rgb'
        - 'cross_distance_nir'
        - 'cross_distance_swir'

    Parameters
    ----------
    tier_0_gpkg_path : str
        Path to the Tier-0 GeoPackage.
    tier_1_gpkg_path : str
        Path to the Tier-1 GeoPackage.
    write_back : bool
        If True, writes the updated Tier-1 layers to output_path (or overwrites tier_1_gpkg_path).
    output_path : Optional[str]
        Target GPKG to write; if None and write_back=True, overwrites tier_1_gpkg_path.

    Returns
    -------
    (tier1_unfiltered, tier1_filtered) : Tuple[GeoDataFrame, GeoDataFrame]
        Updated Tier-1 layers.
    """

    # ---- Helper to load and normalize a Tier-0 band layer ----
    def _load_tier0_layer(layer_name: str, band_label: str) -> pd.DataFrame:
        gdf = gpd.read_file(tier_0_gpkg_path, layer=layer_name)
        # Normalize keys
        gdf["dates"] = pd.to_datetime(gdf["dates"], utc=True, format='%Y-%m-%d-%H-%M-%S')
        gdf["satname"] = gdf["satname"].astype(str).str.upper().str.strip()
        # Keep necessary columns and rename the value column to band-specific name
        if "cross_distance_tidally_corrected" not in gdf.columns:
            raise ValueError(f"Tier-0 layer '{layer_name}' missing 'cross_distance_tidally_corrected'")
        out = (
            gdf[["dates", "satname", "cross_distance_tidally_corrected"]]
            .rename(columns={"cross_distance_tidally_corrected": f"cross_distance_{band_label}"})
        )
        # If multiple rows per (dates, satname), summarize with median
        out = (
            out.groupby(["dates", "satname"], as_index=False)
               .median(numeric_only=True)
        )
        return out

    # ---- Load Tier-0: RGB, NIR, SWIR ----
    rgb  = _load_tier0_layer("02_zoo_rgb_time_series", "rgb")
    nir  = _load_tier0_layer("05_nir_time_series",     "nir")
    swir = _load_tier0_layer("08_swir_time_series",    "swir")

    # Outer merge across bands by (dates, satname)
    tier0 = (
        rgb.merge(nir,  on=["dates", "satname"], how="outer")
            .merge(swir, on=["dates", "satname"], how="outer")
            .sort_values(["dates", "satname"])
            .reset_index(drop=True)
    )

    # ---- Compute median, q10, q90, ci across the three bands ----
    band_cols = ["cross_distance_rgb", "cross_distance_nir", "cross_distance_swir"]
    vals = tier0[band_cols].to_numpy(dtype=float)

    tier0["cross_distance"] = np.nanmedian(vals, axis=1)       # median across bands
    tier0["q10"]            = np.nanquantile(vals, 0.10, axis=1)
    tier0["q90"]            = np.nanquantile(vals, 0.90, axis=1)
    tier0["ci"]             = tier0["q90"] - tier0["q10"]

    # ---- Load Tier-1 layers ----
    layer_unfiltered = "00_unfiltered_shoreline_points"
    layer_filtered   = "01_shoreline_points"

    tier1_unfiltered = gpd.read_file(tier_1_gpkg_path, layer=layer_unfiltered)
    tier1_filtered   = gpd.read_file(tier_1_gpkg_path, layer=layer_filtered)

    for gdf in (tier1_unfiltered, tier1_filtered):
        gdf["dates"]   = pd.to_datetime(gdf["dates"], utc=True, errors="coerce")
        gdf["satname"] = gdf["satname"].astype(str).str.upper().str.strip()

    # ---- Prepare columns to merge into Tier-1 ----
    tier0_keep = ["dates", "satname", "cross_distance", "ci"] + band_cols

    # ---- Merge Tier-0 stats into Tier-1 on (dates, satname) ----
    def _merge_and_assign(tier1_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        joined = tier1_gdf.merge(
            tier0[tier0_keep],
            on=["dates", "satname"],
            how="left",
            suffixes=("", "_tier0")
        )
        # Ensure columns exist even if no match
        for col in ["cross_distance", "ci"] + band_cols:
            if col not in joined.columns:
                joined[col] = np.nan
        return joined

    tier1_unfiltered = _merge_and_assign(tier1_unfiltered)
    tier1_filtered   = _merge_and_assign(tier1_filtered)

    # ---- Optionally write back to GPKG ----
    if write_back:
        target_path = output_path or tier_1_gpkg_path
        tier1_unfiltered.to_file(target_path, layer=layer_unfiltered, driver="GPKG")
        tier1_filtered.to_file(target_path, layer=layer_filtered, driver="GPKG")
