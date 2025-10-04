import os
import json
import numpy as np
import rasterio
from scipy.ndimage import convolve

OUT_DIR = "outputs/abm_ca"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------
# Helpers
# ------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _make_kernel(size: int) -> np.ndarray:
    if size % 2 == 0 or size < 3:
        raise ValueError("neighborhood_size must be odd and >= 3")
    k = np.ones((size, size), dtype=np.float32)
    k[size // 2, size // 2] = 0.0
    return k

def neighborhood_index(urban_bool, size: int):
    kernel = _make_kernel(size)
    denom = kernel.sum()
    nb_cnt = convolve(urban_bool.astype(np.float32), kernel, mode="constant", cval=0.0)
    return nb_cnt / denom

def write_raster_like(template_ds, out_path, array, dtype, nodata=None):
    profile = template_ds.meta.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

def norm01(arr):
    arr = arr.astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

# ------------------------
# Developer & household rules
# ------------------------
def developer_profit(prob, dist_roads, land_price, w_road, w_land):
    road_factor = 1 - norm01(dist_roads)   # closer to roads = better
    land_factor = 1 - norm01(land_price)   # cheaper land = better
    return prob * (w_road * road_factor + w_land * land_factor)

def household_utility(prob, dist_cbds, env_quality, w_s, w_d, w_e):
    dist_factor = 1 - norm01(dist_cbds)  # closer to CBD = better
    return (w_s * prob + w_d * dist_factor + w_e * env_quality)

# ------------------------
# ABM-CA engine
# ------------------------
def run(cfg=None):
    """
    ABM-CA hybrid simulation.
    If cfg=None, load from configs/abm_config.json
    """
    if cfg is None:
        with open("configs/abm_config.json", "r") as f:
            cfg = json.load(f)

    start_year = int(cfg["start_year"])
    end_year   = int(cfg["end_year"])
    steps = end_year - start_year

    out_dir = os.path.join(OUT_DIR, f"{cfg['run_name']}_{start_year}_{end_year}")
    os.makedirs(out_dir, exist_ok=True)

    # Load rasters
    with rasterio.open(cfg["start_urban_path"]) as ds_u0, \
         rasterio.open(cfg["suitability_path"]) as ds_s:

        u = ds_u0.read(1)
        suit = ds_s.read(1).astype(np.float32)

        if cfg.get("mask_path") and os.path.exists(cfg["mask_path"]):
            with rasterio.open(cfg["mask_path"]) as ds_mask:
                mask = ds_mask.read(1)
        else:
            mask = np.ones_like(u, dtype=np.uint8)

        urban = (u > 0)
        developable = (mask > 0)

        # Static layers for ABM
        dist_roads = rasterio.open(cfg["dist_roads"]).read(1)
        dist_cbds  = rasterio.open(cfg["dist_cbds"]).read(1)
        land_price = rasterio.open(cfg["land_price"]).read(1) if cfg.get("land_price") else np.ones_like(suit)
        env_quality = suit.copy()  # proxy: suitability

        # Determine total cells to convert (area-match)
        if "observed_change_path" in cfg and os.path.exists(cfg["observed_change_path"]):
            with rasterio.open(cfg["observed_change_path"]) as ds_chg:
                change_obs = ds_chg.read(1)
            total_new = int((change_obs == 1).sum())
            per_step  = total_new // steps
            spillover = total_new - per_step * steps
        else:
            per_step, spillover = 1000, 0  # default growth

        for t in range(steps):
            # Step 1: CA probability
            nb = neighborhood_index(urban, size=int(cfg["neighborhood_size"]))
            z = cfg["a"] * suit + cfg["b"] * nb + cfg["c"]
            prob = sigmoid(z / float(cfg["temperature"]))

            # Step 2: Developer layer
            dev = developer_profit(prob, dist_roads, land_price,
                                   cfg["w_road"], cfg["w_land"])

            # Step 3: Household layer
            hh = household_utility(prob, dist_cbds, env_quality,
                                   cfg["w_s"], cfg["w_d"], cfg["w_e"])

            # Step 4: Combine into ABM weight
            abm_weight = prob * dev * hh
            abm_weight = np.nan_to_num(abm_weight, nan=0.0, posinf=0.0, neginf=0.0)

            # Step 5: Select k cells
            can_convert = (~urban) & developable
            k = per_step + (spillover if t == steps - 1 else 0)
            idx_candidates = np.flatnonzero(can_convert)
            if idx_candidates.size > 0 and k > 0:
                weights = abm_weight.flat[idx_candidates]
                weights = np.nan_to_num(weights, nan=0.0)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    sel_local = np.random.choice(idx_candidates, size=min(k, idx_candidates.size),
                                                 replace=False, p=weights)
                else:
                    sel_local = np.random.choice(idx_candidates, size=min(k, idx_candidates.size),
                                                 replace=False)
                urban.flat[sel_local] = True

            if cfg.get("save_yearly", False):
                year = start_year + t + 1
                out_path = os.path.join(out_dir, f"urban_{year}_sim.tif")
                write_raster_like(ds_u0, out_path, urban.astype(np.uint8), dtype=rasterio.uint8, nodata=0)

        # Save final
        final_out = os.path.join(out_dir, f"urban_{end_year}_sim.tif")
        write_raster_like(ds_u0, final_out, urban.astype(np.uint8), dtype=rasterio.uint8, nodata=0)
        print(f"Saved ABM-CA final map: {final_out}")

if __name__ == "__main__":
    run()
