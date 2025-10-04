import os
import json
import numpy as np
import rasterio
from scipy.ndimage import convolve
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

RESULTS_DIR = "outputs/validation"
SUIT_DIR = "outputs/suitability_ml_lgbm"  # LightGBM suitability
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Small helpers
# -------------------------
def load_raster(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds

def write_raster_like(template_ds, out_path, array, dtype=rasterio.uint8, nodata=0):
    profile = template_ds.meta.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def kernel_isotropic(size: int) -> np.ndarray:
    if size % 2 == 0 or size < 3:
        raise ValueError("neighborhood size must be odd and >=3")
    k = np.ones((size, size), dtype=np.float32)
    k[size // 2, size // 2] = 0.0
    return k

def kernel_anisotropic(size: int, axis: str, strength: float) -> np.ndarray:
    k = kernel_isotropic(size)
    yy, xx = np.indices(k.shape)
    cy, cx = size // 2, size // 2
    dx = np.abs(xx - cx)
    dy = np.abs(yy - cy)
    if axis == "ew":
        w = (1.0 + (dx > dy) * (strength - 1.0))
    else:
        w = (1.0 + (dy > dx) * (strength - 1.0))
    k = k * w
    k[cy, cx] = 0.0
    return k

def neighborhood_index(urban_bool: np.ndarray,
                       sizes, weights,
                       mode="isotropic", anis_axis="ew", anis_strength=1.5) -> np.ndarray:
    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-9)
    acc = np.zeros_like(urban_bool, dtype=np.float32)
    for s, w in zip(sizes, weights):
        k = kernel_isotropic(s) if mode == "isotropic" else kernel_anisotropic(s, anis_axis, anis_strength)
        denom = k.sum()
        nb_cnt = convolve(urban_bool.astype(np.float32), k, mode="constant", cval=0.0)
        acc += w * (nb_cnt / (denom + 1e-9))
    return acc

def norm01(arr):
    arr = arr.astype(np.float32)
    m = np.nanmin(arr)
    M = np.nanmax(arr)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return np.zeros_like(arr, dtype=np.float32)
    x = (arr - m) / (M - m)
    x = np.clip(x, 0, 1)
    return x

def dist_to_proximity(dist, max_cap=None):
    dist = dist.astype(np.float32)
    if max_cap is None or not np.isfinite(max_cap):
        max_cap = np.nanmax(dist) if np.isfinite(np.nanmax(dist)) else 1.0
    dist = np.clip(dist, 0, max_cap)
    prox = 1.0 - (dist / (max_cap + 1e-9))
    return np.clip(prox, 0, 1).astype(np.float32)

def quantity_allocation_disagreement(obs_change, sim_change):
    y_true = (obs_change == 1).ravel()
    y_pred = (sim_change == 1).ravel()
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    N = tp + fp + fn + tn
    if N == 0:
        return 0.0, 0.0
    Q = abs(fp - fn) / N
    A = (2.0 * min(fp, fn)) / N
    return Q, A

def compute_fom(obs_change, sim_change):
    obs = (obs_change == 1)
    pred = (sim_change == 1)
    inter = np.logical_and(obs, pred).sum()
    union = obs.sum() + pred.sum() - inter
    return inter / union if union > 0 else 0.0

# Weighted sampling without replacement over candidate cells
def select_k_weighted(prob_weights, candidate_mask, k, rng):
    idx_candidates = np.flatnonzero(candidate_mask)
    if idx_candidates.size == 0 or k <= 0:
        return np.zeros_like(candidate_mask, dtype=bool)
    k = min(k, idx_candidates.size)

    # Extract weights for candidates
    w = prob_weights.flat[idx_candidates].astype(np.float64)

    # Clean up NaNs / infs
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    if w.sum() <= 0:
        # Fallback: uniform random selection
        sel_local = rng.choice(idx_candidates.size, size=k, replace=False)
    else:
        w = w / w.sum()
        sel_local = rng.choice(idx_candidates.size, size=k, replace=False, p=w)

    sel_mask = np.zeros_like(candidate_mask, dtype=bool)
    sel_mask.flat[idx_candidates[sel_local]] = True
    return sel_mask

# -------------------------
# ABM-CA hybrid yearly loop
# -------------------------
def run_validation_abm(save_outputs=True, also_run_baseline=False):
    # 1) Load best params from calibration
    best_path = "outputs/calibration/bayes_joint_best.json"
    if not os.path.exists(best_path):
        raise FileNotFoundError("Run calibration first (outputs/calibration/bayes_joint_best.json)")
    with open(best_path, "r") as f:
        best = json.load(f)
    p = best["params"]

    # Multi-scale params (fallbacks for older result files)
    sizes = p.get("sizes", [p.get("N", 3)])
    weights = p.get("weights", [1.0] if "weights" not in p else p["weights"])
    kernel_mode = p.get("kernel_mode", "isotropic")
    anis_axis = p.get("anis_axis", p.get("anisotropy_axis", "ns"))
    anis_strength = p.get("anis_strength", p.get("anisotropy_strength", 1.5))

    # 2) Load inputs for 2010→2020
    start_urban, ds_u = load_raster("data/aligned/2010/urban_2010_aligned.tif")
    suit2010, _ = load_raster(os.path.join(SUIT_DIR, "suitability_2010.tif"))
    mask_arr, _ = load_raster("data/aligned/static/mask_restricted_aligned.tif")
    obs_change, _ = load_raster("data/aligned/change_2010_2020_aligned.tif")

    # ABM factors
    dist_roads, _ = load_raster("data/aligned/static/dist_roads_aligned.tif")
    dist_cbds, _ = load_raster("data/aligned/static/dist_cbds_aligned.tif")
    # optional land price
    land_price_path = "data/aligned/static/land_price_aligned.tif"
    if os.path.exists(land_price_path):
        land_price, _ = load_raster(land_price_path)
    else:
        land_price = np.ones_like(dist_roads, dtype=np.float32)

    # environment proxy (use NDVI 2010 if available, otherwise suitability)
    ndvi_path = "data/aligned/2010/ndvi_2010_aligned.tif"
    if os.path.exists(ndvi_path):
        env_quality, _ = load_raster(ndvi_path)
        env_quality = norm01(env_quality)
    else:
        env_quality = norm01(suit2010)

    # masks & types
    developable = (mask_arr > 0)
    urban = (start_urban > 0)

    # 3) Compute area-match targets per step from observed change
    total_new = int((obs_change == 1).sum())
    steps = 2020 - 2010
    per_step = total_new // steps
    spillover = total_new - per_step * steps

    # 4) Constants / knobs
    a = p["a"]; b = p["b"]; c = p["c"]; T = p["T"]
    min_prob = p.get("hybrid_min_prob", 0.35)
    rng = np.random.default_rng(42)

    # ABM weights
    dev_w_road = 0.6
    dev_w_land = 0.4
    hh_w_suit  = 0.5
    hh_w_cbd   = 0.3
    hh_w_env   = 0.2

    # Precompute ABM static layers (normalized)
    road_factor = dist_to_proximity(dist_roads, max_cap=p.get("road_max_m", 8000.0))
    land_factor = 1.0 - norm01(land_price)
    cbd_prox    = dist_to_proximity(dist_cbds, max_cap=np.nanmax(dist_cbds))

    # 5) Optional baseline (CA-only) parallel run for comparison
    do_baseline = bool(also_run_baseline)
    if do_baseline:
        urban_base = urban.copy()

    # 6) Yearly loop with ABM
    for t in range(steps):
        # CA probability from suitability + multi-scale neighborhood
        nb = neighborhood_index(urban, sizes=sizes, weights=weights,
                                mode=kernel_mode, anis_axis=anis_axis, anis_strength=anis_strength)
        z = a * suit2010 + b * nb + c
        prob = sigmoid(z / T)

        # stochastic smoothing (small noise)
        prob = np.clip(prob + rng.normal(0.0, 1e-2, size=prob.shape).astype(np.float32), 0, 1)

        # Candidates: not yet urban, developable, and above min prob
        can_convert = (~urban) & developable & (prob >= float(min_prob))

        # ABM scoring on candidates
        developer_profit = (dev_w_road * road_factor + dev_w_land * land_factor)
        household_util   = (hh_w_suit * prob + hh_w_cbd * cbd_prox + hh_w_env * env_quality)

        # Combine into a single weight for sampling
        # Multiplicative blend gives strong agreement emphasis:
        abm_weight = np.clip(prob * developer_profit * household_util, 0, None)

        # How many to convert this year?
        k = per_step + (spillover if t == steps - 1 else 0)

        # ABM selection
        to_convert = select_k_weighted(abm_weight, can_convert, k, rng)
        urban[to_convert] = True

        # Baseline CA-only (no ABM weighting): pick top-k by prob
        if do_baseline:
            can_convert_b = (~urban_base) & developable & (prob >= float(min_prob))
            # select top-k by prob among candidates
            idx_c = np.flatnonzero(can_convert_b)
            k_b = min(k, idx_c.size)
            if k_b > 0:
                pvals = prob.flat[idx_c]
                kth = np.argpartition(pvals, -k_b)[-k_b:]
                sel_idx = idx_c[kth]
                mask_sel = np.zeros_like(can_convert_b, dtype=bool)
                mask_sel.flat[sel_idx] = True
                urban_base[mask_sel] = True

    # 7) Save final map(s)
    out_dir = os.path.join("outputs/ca", "validation_abm_2010_2020")
    write_raster_like(ds_u, os.path.join(out_dir, "urban_2020_sim_abm.tif"), urban.astype(np.uint8))
    if do_baseline:
        write_raster_like(ds_u, os.path.join(out_dir, "urban_2020_sim_baseline.tif"), urban_base.astype(np.uint8))

    # 8) Metrics (ABM)
    sim_change_abm = ((urban == 1) & (start_urban == 0)).astype(np.uint8)
    fom_abm = compute_fom(obs_change, sim_change_abm)
    Q_abm, A_abm = quantity_allocation_disagreement(obs_change, sim_change_abm)
    y_true = obs_change.flatten()
    y_pred = sim_change_abm.flatten()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    results = {
        "FoM": float(fom_abm),
        "Q": float(Q_abm),
        "A": float(A_abm),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "sim_new": int(sim_change_abm.sum()),
        "obs_new": int((obs_change == 1).sum()),
        "note": "ABM-CA hybrid validation (2010→2020) using LGBM suitability and yearly ABM selection"
    }

    # 9) Optional baseline metrics
    if do_baseline:
        sim_change_base = ((urban_base == 1) & (start_urban == 0)).astype(np.uint8)
        fom_b = compute_fom(obs_change, sim_change_base)
        Q_b, A_b = quantity_allocation_disagreement(obs_change, sim_change_base)
        y_pred_b = sim_change_base.flatten()
        precision_b = precision_score(y_true, y_pred_b, zero_division=0)
        recall_b    = recall_score(y_true, y_pred_b, zero_division=0)
        f1_b        = f1_score(y_true, y_pred_b, zero_division=0)

        results["baseline"] = {
            "FoM": float(fom_b),
            "Q": float(Q_b),
            "A": float(A_b),
            "precision": float(precision_b),
            "recall": float(recall_b),
            "f1": float(f1_b),
            "sim_new": int(sim_change_base.sum())
        }

    # 10) Save report
    out_json = os.path.join(RESULTS_DIR, "validation_abm_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print("\n=== ABM-CA Validation (2010→2020) ===")
    for k, v in results.items():
        if k != "baseline":
            print(f"{k}: {v}")
    if "baseline" in results:
        print("\n--- Baseline (CA-only, no ABM weighting) ---")
        for k, v in results["baseline"].items():
            print(f"{k}: {v}")
    print(f"\nSaved results to {out_json}")
    print(f"Saved rasters to {out_dir}")

if __name__ == "__main__":
    # Set also_run_baseline=True to compare with a CA-only run
    run_validation_abm(save_outputs=True, also_run_baseline=True)
