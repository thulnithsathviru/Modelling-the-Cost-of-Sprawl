import os
import shutil
import rasterio
import numpy as np
import logging
from rasterio.enums import Resampling
from scipy.ndimage import sobel

# ===== Paths =====
RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "data/aligned"

YEARS = ["2000", "2010", "2020"]
STATIC_DIR = os.path.join(RAW_DATA_DIR, "static")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ===== Helper functions =====

def choose_resampling(fname):
    """Pick appropriate resampling based on data type."""
    categorical = ["lulc", "urban", "mask"]
    if any(c in fname.lower() for c in categorical):
        return Resampling.nearest
    return Resampling.bilinear

def copy_if_aligned(src_path, ref_profile, dst_path):
    """Check alignment; if aligned, copy; else resample."""
    with rasterio.open(src_path) as src:
        profile = ref_profile.copy()
        if (
            src.crs == ref_profile["crs"]
            and src.transform == ref_profile["transform"]
            and src.width == ref_profile["width"]
            and src.height == ref_profile["height"]
        ):
            shutil.copy(src_path, dst_path)
        else:
            logging.warning(f"{src_path} not aligned, resampling...")
            resampling = choose_resampling(os.path.basename(src_path))
            data = src.read(
                out_shape=(src.count, ref_profile["height"], ref_profile["width"]),
                resampling=resampling,
            )
            profile.update(count=src.count, dtype=src.dtypes[0], nodata=src.nodata)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)

def create_change_map(urban_prev, urban_next, out_path, profile):
    """Create change raster (1 = new urban, 0 = no change, 255 = nodata)."""
    # Ensure strictly binary 0/1 values
    prev = (urban_prev > 0).astype(np.uint8)
    nxt = (urban_next > 0).astype(np.uint8)

    # Handle nodata properly
    nodata_val = profile.get("nodata")
    if nodata_val is not None:
        mask = (urban_prev == nodata_val) | (urban_next == nodata_val)
    else:
        mask = np.zeros_like(prev, dtype=bool)

    # New urban = was 0, became 1
    change = np.where((nxt == 1) & (prev == 0), 1, 0).astype(np.uint8)

    # Apply mask → mark nodata as 255 (not 0)
    change = np.where(mask, 255, change).astype(np.uint8)

    # Update metadata
    profile.update(dtype=rasterio.uint8, count=1, nodata=255)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(change, 1)


def compute_slope(dem_array, transform):
    """Approximate slope in degrees from DEM using Sobel filter."""
    xres = transform.a
    yres = -transform.e
    dzdx = sobel(dem_array, axis=1) / (8.0 * xres)
    dzdy = sobel(dem_array, axis=0) / (8.0 * yres)
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    return slope.astype(np.float32)

# ===== Main workflow =====

def run():
    logging.info("=== Running Data Preparation ===")

    # ---- Step 1. Get reference raster ----
    ref_path = os.path.join(RAW_DATA_DIR, "2000", "urban_2000.tif")
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.meta.copy()

    # ---- Step 2. Copy yearly rasters ----
    for year in YEARS:
        year_in = os.path.join(RAW_DATA_DIR, year)
        year_out = os.path.join(OUTPUT_DIR, year)
        os.makedirs(year_out, exist_ok=True)

        for fname in os.listdir(year_in):
            if fname.endswith(".tif"):
                src = os.path.join(year_in, fname)
                dst = os.path.join(year_out, fname.replace(".tif", "_aligned.tif"))
                logging.info(f"Processing {src} → {dst}")
                copy_if_aligned(src, ref_profile, dst)

    # ---- Step 3. Copy static rasters ----
    static_out = os.path.join(OUTPUT_DIR, "static")
    os.makedirs(static_out, exist_ok=True)

    if os.path.exists(STATIC_DIR):
        for fname in os.listdir(STATIC_DIR):
            if fname.endswith(".tif"):
                src = os.path.join(STATIC_DIR, fname)
                dst = os.path.join(static_out, fname.replace(".tif", "_aligned.tif"))
                logging.info(f"Processing {src} → {dst}")
                copy_if_aligned(src, ref_profile, dst)
    else:
        logging.error("Static folder missing!")

    # ---- Step 4. Create change maps ----
    logging.info("Creating change maps...")
    with rasterio.open(os.path.join(OUTPUT_DIR, "2000", "urban_2000_aligned.tif")) as u2000, \
         rasterio.open(os.path.join(OUTPUT_DIR, "2010", "urban_2010_aligned.tif")) as u2010, \
         rasterio.open(os.path.join(OUTPUT_DIR, "2020", "urban_2020_aligned.tif")) as u2020:

        urban2000, urban2010, urban2020 = u2000.read(1), u2010.read(1), u2020.read(1)
        profile = u2000.meta.copy()

        create_change_map(urban2000, urban2010,
                          os.path.join(OUTPUT_DIR, "change_2000_2010_aligned.tif"), profile)
        create_change_map(urban2010, urban2020,
                          os.path.join(OUTPUT_DIR, "change_2010_2020_aligned.tif"), profile)

    # ---- Step 5. Compute slope ----
    dem_path = os.path.join(static_out, "dem_aligned.tif")
    slope_path = os.path.join(static_out, "slope_aligned.tif")

    if os.path.exists(dem_path) and not os.path.exists(slope_path):
        logging.info("Computing slope from DEM...")
        with rasterio.open(dem_path) as dem:
            dem_arr = dem.read(1, masked=True)
            slope_arr = compute_slope(dem_arr, dem.transform)
            profile = dem.meta.copy()
            profile.update(dtype=rasterio.float32, count=1, nodata=None)
            with rasterio.open(slope_path, "w", **profile) as dst:
                dst.write(slope_arr, 1)
    else:
        logging.info("Slope already exists or DEM missing, skipping computation.")

    logging.info("=== Data Preparation Completed ===")

if __name__ == "__main__":
    run()