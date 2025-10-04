import os
import json
import numpy as np
import rasterio
import pandas as pd
from scipy.ndimage import convolve

RESULTS_DIR = "outputs/scenarios"
SUIT_DIR = "outputs/suitability_ml_lgbm"
CAL_CA_JSON = "outputs/calibration/bayes_joint_best.json"  # optional: CA params
CAL_ABM_JSON = "outputs/calibration_abm/abm_best.json"     # optional: ABM params

MASK_PATH   = "data/aligned/static/mask_restricted_aligned.tif"
URBAN_2020  = "data/aligned/2020/urban_2020_aligned.tif"
CHANGE_1020 = "data/aligned/change_2010_2020_aligned.tif"
DIST_ROADS  = "data/aligned/static/dist_roads_aligned.tif"
DIST_RAIL   = "data/aligned/static/dist_rail_aligned.tif"
DIST_CBDS   = "data/aligned/static/dist_cbds_aligned.tif"
LULC_2020   = "data/aligned/2020/lulc_2020_aligned.tif"
LAND_PRICE  = "data/aligned/static/land_price_aligned.tif"  # optional

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- scenarios you run ----------
SCENARIOS = [
    "BAU",
    "Compact",
    "TransitOriented",
    "HighwayBias",
    "FarmlandProtection",
]

# ---------- growth control ----------
# mode: "infer" (from 2010→2020), "total_cells", "percent_increase", "logistic"
GROWTH = {
    "mode": "logistic",
    "total_cells": None,          # if mode == "total_cells"
    "percent_increase": None,     # if mode == "percent_increase" (relative to 2020 urban)
    # logistic params (used if mode == "logistic")
    "K_factor": 2.0,              # carrying capacity = K_factor * (2020 urban cells)
    "r": 0.08,                    # growth rate
    "t0": 15                      # inflection at ~2035 (15 years after 2020)
}

# ---------- CA core params (fallback if calibration JSON not present) ----------
CA_FALLBACK = {
    "a": 1.2, "b": 0.25, "c": 0.0, "T": 1.0,
    "sizes": [3], "weights": [1.0],
    "kernel_mode": "anisotropic", "anis_axis": "ns", "anis_strength": 2.0,
    "min_prob": 0.35,             # hybrid minimum probability gate
    "noise_sigma": 0.05           # <-- stronger noise to allow MC variation
}

# ---------- ABM defaults (fallback if calibration JSON not present) ----------
ABM_FALLBACK = {"w_road": 0.6, "w_land": 0.4, "w_s": 0.5, "w_d": 0.3, "w_e": 0.2}
ABM_JITTER = 1e-3   # weight jitter before sampling to avoid deterministic ties

# ---------- utilities ----------
def load_raster(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds

def write_raster_like(template_ds, out_path, array, dtype=rasterio.uint8, nodata=0):
    prof = template_ds.meta.copy()
    prof.update(dtype=dtype, count=1, nodata=nodata)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(array.astype(dtype), 1)

def write_float_raster_like(template_ds, out_path, array):
    prof = template_ds.meta.copy()
    prof.update(dtype=rasterio.float32, count=1, nodata=None)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(array.astype(np.float32), 1)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def kernel(size: int, mode="isotropic", axis="ew", strength=1.5):
    if size % 2 == 0 or size < 3:
        raise ValueError("Neighborhood size must be odd and >=3")
    k = np.ones((size, size), dtype=np.float32)
    cy = cx = size // 2
    k[cy, cx] = 0.0
    if mode == "isotropic":
        return k
    # anisotropic emphasis
    yy, xx = np.indices(k.shape)
    dx, dy = np.abs(xx - cx), np.abs(yy - cy)
    if axis == "ew":
        w = 1.0 + (dx > dy) * (strength - 1.0)
    else:
        w = 1.0 + (dy > dx) * (strength - 1.0)
    k = k * w
    k[cy, cx] = 0.0
    return k

def neighborhood_index(urban_bool, sizes, weights, mode="isotropic", axis="ew", strength=1.5):
    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-9)
    acc = np.zeros_like(urban_bool, dtype=np.float32)
    for s, w in zip(sizes, weights):
        k = kernel(s, mode, axis, strength)
        denom = k.sum()
        nb = convolve(urban_bool.astype(np.float32), k, mode="constant", cval=0.0) / (denom + 1e-9)
        acc += w * nb
    return acc

def norm01(arr):
    arr = arr.astype(np.float32)
    m, M = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - m) / (M - m), 0, 1)

def dist_to_prox(dist, cap=None):
    dist = dist.astype(np.float32)
    if cap is None or not np.isfinite(cap):
        cap = np.nanmax(dist) if np.isfinite(np.nanmax(dist)) else 1.0
    dist = np.clip(dist, 0, cap)
    return np.clip(1.0 - dist / (cap + 1e-9), 0, 1).astype(np.float32)

def safe_weights(w):
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    s = w.sum()
    return (w / s) if s > 0 else None

# ---------- growth targets ----------
def infer_per_year_from_2010_2020():
    ch, _ = load_raster(CHANGE_1020)
    total_new = int((ch == 1).sum())
    per_year = total_new // 10
    return per_year

def make_yearly_targets(mode_dict, start_urban, steps=30):
    """
    Returns an array of length `steps` with k[t] = number of cells to add in year t (2020+t+1).
    """
    base = int((start_urban == 1).sum())
    if mode_dict["mode"] == "infer":
        per = infer_per_year_from_2010_2020()
        return np.array([per] * steps, dtype=int)

    if mode_dict["mode"] == "total_cells":
        tot = int(mode_dict["total_cells"])
        per, spill = tot // steps, tot - (tot // steps) * steps
        out = [per] * steps
        if spill > 0: out[-1] += spill
        return np.array(out, dtype=int)

    if mode_dict["mode"] == "percent_increase":
        inc = int(base * float(mode_dict["percent_increase"]))
        per, spill = inc // steps, inc - (inc // steps) * steps
        out = [per] * steps
        if spill > 0: out[-1] += spill
        return np.array(out, dtype=int)

    if mode_dict["mode"] == "logistic":
        K = base * float(mode_dict.get("K_factor", 2.0))  # carrying capacity in cells
        r = float(mode_dict.get("r", 0.08))
        t0 = float(mode_dict.get("t0", 15))
        years = np.arange(steps + 1)  # 0..30
        U = K / (1.0 + np.exp(-r * (years - t0)))  # S-curve
        # scale so that U[0] ~= base
        scale = base / max(U[0], 1e-9)
        U = U * scale
        yearly = np.diff(U)                 # additions each year
        yearly = np.maximum(yearly, 0.0)    # guard
        return np.round(yearly).astype(int)

    raise ValueError("Unknown GROWTH['mode']")

# ---------- scenario suitability modifiers ----------
def build_scenario_suitability(base_suit, scenario_name):
    """Return modified suitability array and a template dataset."""
    suit, ds = load_raster(base_suit)

    if scenario_name == "BAU":
        mod = suit.copy()

    elif scenario_name == "Compact":
        u20, _ = load_raster(URBAN_2020)
        kernel7 = np.ones((7, 7), dtype=np.float32) / 49.0
        density = convolve(u20.astype(np.float32), kernel7, mode="constant", cval=0.0)
        mod = np.clip(suit + 0.20 * density, 0, 1)

    elif scenario_name == "TransitOriented":
        drail, _ = load_raster(DIST_RAIL)
        near_rail = dist_to_prox(drail, cap=np.nanpercentile(drail, 95))
        mod = np.clip(suit + 0.30 * near_rail, 0, 1)

    elif scenario_name == "HighwayBias":
        droads, _ = load_raster(DIST_ROADS)
        near_roads = dist_to_prox(droads, cap=np.nanpercentile(droads, 95))
        mod = np.clip(suit + 0.30 * near_roads, 0, 1)

    elif scenario_name == "FarmlandProtection":
        lulc, _ = load_raster(LULC_2020)
        farmland = (lulc == 3)  # adjust class if needed
        mod = suit.copy()
        mod[farmland] = 0.0

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    return mod, ds

# ---------- load best params (if available) ----------
def load_calibrated_params():
    # CA (from bayes_joint_best.json)
    ca = CA_FALLBACK.copy()
    if os.path.exists(CAL_CA_JSON):
        try:
            with open(CAL_CA_JSON, "r") as f:
                best = json.load(f)
            p = best.get("params", {})
            ca["a"] = p.get("a", ca["a"])
            ca["b"] = p.get("b", ca["b"])
            ca["c"] = p.get("c", ca["c"])
            ca["T"] = p.get("T", ca["T"])
            # multi-scale neighborhood if present
            if "sizes" in p and "weights" in p:
                ca["sizes"] = p["sizes"]
                ca["weights"] = p["weights"]
            else:
                # single scale fallback from N
                if "N" in p:
                    ca["sizes"] = [p["N"]]
                    ca["weights"] = [1.0]
            ca["kernel_mode"]   = p.get("kernel_mode", ca["kernel_mode"])
            ca["anis_axis"]     = p.get("anis_axis", p.get("anisotropy_axis", ca["anis_axis"]))
            ca["anis_strength"] = p.get("anis_strength", p.get("anisotropy_strength", ca["anis_strength"]))
            # keep min_prob and noise_sigma from fallback unless you want to expose in calibration
        except Exception:
            pass

    # ABM (from abm_best.json)
    abm = ABM_FALLBACK.copy()
    if os.path.exists(CAL_ABM_JSON):
        try:
            with open(CAL_ABM_JSON, "r") as f:
                best = json.load(f)
            abm.update(best.get("params", {}))
        except Exception:
            pass

    return ca, abm

# ---------- ABM–CA yearly engine ----------
def run_abm_ca_2020_2050(scenario_name, suit_mod, template_ds, ca_params, abm_params, out_dir, seed=None):
    """Yearly loop: CA probability → ABM weighted sampling → convert k per year."""
    # Load base layers
    start_urban, ds_u = load_raster(URBAN_2020)
    mask_arr, _ = load_raster(MASK_PATH)
    droads, _ = load_raster(DIST_ROADS)
    dcbd, _ = load_raster(DIST_CBDS)

    if os.path.exists(LAND_PRICE):
        land, _ = load_raster(LAND_PRICE)
    else:
        land = np.ones_like(start_urban, dtype=np.float32)

    # growth targets (array length 30)
    yearly_targets = make_yearly_targets(GROWTH, start_urban, steps=30)

    # CA params
    a = float(ca_params["a"]); b = float(ca_params["b"]); c = float(ca_params["c"]); T = float(ca_params["T"])
    sizes = ca_params.get("sizes", [3])
    weights = ca_params.get("weights", [1.0])
    k_mode = ca_params.get("kernel_mode", "isotropic")
    k_axis = ca_params.get("anis_axis", "ew")
    k_strength = float(ca_params.get("anis_strength", 1.5))
    min_prob = float(ca_params.get("min_prob", 0.35))
    noise_sigma = float(ca_params.get("noise_sigma", 0.05))  # stronger noise

    # ABM params (calibrated or defaults)
    w_road = float(abm_params.get("w_road", ABM_FALLBACK["w_road"]))
    w_land = float(abm_params.get("w_land", ABM_FALLBACK["w_land"]))
    w_s    = float(abm_params.get("w_s", ABM_FALLBACK["w_s"]))
    w_d    = float(abm_params.get("w_d", ABM_FALLBACK["w_d"]))
    w_e    = float(abm_params.get("w_e", ABM_FALLBACK["w_e"]))

    # RNG
    rng = np.random.default_rng(seed if seed is not None else 42)

    # normalize statics
    road_fac = dist_to_prox(droads, cap=np.nanpercentile(droads, 95))
    land_fac = 1.0 - norm01(land)
    cbd_fac  = dist_to_prox(dcbd,   cap=np.nanpercentile(dcbd, 95))

    developable = (mask_arr > 0)
    urban = (start_urban > 0)

    # save modified suitability to scenario folder (for transparency)
    suit_path = os.path.join(out_dir, f"suitability_{scenario_name}.tif")
    write_float_raster_like(template_ds, suit_path, suit_mod)

    steps = 30  # 2020→2050
    yearly_totals = []

    for t in range(steps):
        # CA probability
        nb = neighborhood_index(urban, sizes, weights, k_mode, k_axis, k_strength)
        z = a * suit_mod + b * nb + c
        prob = sigmoid(z / T)
        # ADD stochastic noise (important for uncertainty)
        prob = np.clip(prob + rng.normal(0.0, noise_sigma, prob.shape).astype(np.float32), 0, 1)

        # ABM combining
        dev = (w_road * road_fac + w_land * land_fac)
        hh  = (w_s * prob + w_d * cbd_fac + w_e * suit_mod)  # suit_mod as env proxy
        abm_w = np.clip(prob * dev * hh, 0, None)
        abm_w = np.nan_to_num(abm_w, nan=0.0, posinf=0.0, neginf=0.0)

        # candidates & yearly k
        can = (~urban) & developable & (prob >= min_prob)

        # stochastic yearly target ±10% (optional; comment if not desired)
        k_nominal = int(yearly_targets[t])
        k = int(k_nominal * rng.uniform(0.9, 1.1))
        if k <= 0:
            # still write out the year & continue
            year = 2020 + t + 1
            write_raster_like(ds_u, os.path.join(out_dir, f"urban_{year}_sim.tif"), urban.astype(np.uint8))
            yearly_totals.append(int((urban == 1).sum()))
            continue

        # weighted sampling
        idx_c = np.flatnonzero(can)
        if idx_c.size > 0:
            w = abm_w.flat[idx_c].astype(np.float64)
            # ADD jitter to avoid identical weight ties
            w = w + rng.random(w.shape) * ABM_JITTER
            w = safe_weights(w)

            k_eff = min(k, idx_c.size)
            if w is None:
                sel = rng.choice(idx_c, size=k_eff, replace=False)
            else:
                sel = rng.choice(idx_c, size=k_eff, replace=False, p=w)
            urban.flat[sel] = True

        # save yearly
        year = 2020 + t + 1
        write_raster_like(ds_u, os.path.join(out_dir, f"urban_{year}_sim.tif"), urban.astype(np.uint8))
        yearly_totals.append(int((urban == 1).sum()))

    # final save
    write_raster_like(ds_u, os.path.join(out_dir, "urban_2050_sim.tif"), urban.astype(np.uint8))

    # summary CSV
    df = pd.DataFrame({"year": list(range(2021, 2051)), "urban_total_cells": yearly_totals})
    df.to_csv(os.path.join(out_dir, f"summary_{scenario_name}.csv"), index=False)

    return os.path.join(out_dir, "urban_2050_sim.tif"), os.path.join(out_dir, f"summary_{scenario_name}.csv")

# ---------- main ----------
def run():
    # load calibrated params if available
    ca_params, abm_params = load_calibrated_params()

    # base suitability (LGBM)
    base_suit_path = os.path.join(SUIT_DIR, "suitability_2020.tif")
    if not os.path.exists(base_suit_path):
        raise FileNotFoundError(f"Missing suitability: {base_suit_path}. Run suitability_lgbm.py first.")

    # scenario loop
    for sc in SCENARIOS:
        print(f"\n=== Scenario: {sc} (2020→2050) ===")
        suit_mod, temp_ds = build_scenario_suitability(base_suit_path, sc)

        out_dir = os.path.join(RESULTS_DIR, f"{sc}_2020_2050")
        os.makedirs(out_dir, exist_ok=True)

        final_tif, csv_path = run_abm_ca_2020_2050(
            sc, suit_mod, temp_ds, ca_params, abm_params, out_dir, seed=None
        )
        print(f"Saved: {final_tif}")
        print(f"Saved: {csv_path}")

    print("\n=== All scenarios complete ===")

if __name__ == "__main__":
    run()
