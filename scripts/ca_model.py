import os
import json
import numpy as np
import rasterio
from scipy.ndimage import convolve

OUT_DIR = "outputs/ca"
os.makedirs(OUT_DIR, exist_ok=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------- Neighborhoods ----------
def _kernel_isotropic(size: int) -> np.ndarray:
    if size % 2 == 0 or size < 3:
        raise ValueError("neighborhood size must be odd and >=3")
    k = np.ones((size, size), dtype=np.float32)
    k[size // 2, size // 2] = 0.0
    return k

def _kernel_anisotropic(size: int, axis: str, strength: float) -> np.ndarray:
    """
    axis: 'ew' or 'ns'
    strength: >1 emphasizes preferred axis; 1.0 → isotropic
    """
    k = _kernel_isotropic(size)
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
                       sizes: list,
                       weights: list,
                       mode: str = "isotropic",
                       anis_axis: str = "ew",
                       anis_strength: float = 1.5) -> np.ndarray:
    """Weighted sum of (normalized) neighborhood densities across scales."""
    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-9)
    acc = np.zeros_like(urban_bool, dtype=np.float32)
    for s, w in zip(sizes, weights):
        k = _kernel_isotropic(s) if mode == "isotropic" else _kernel_anisotropic(s, anis_axis, anis_strength)
        denom = k.sum()
        nb_cnt = convolve(urban_bool.astype(np.float32), k, mode="constant", cval=0.0)
        acc += w * (nb_cnt / (denom + 1e-9))
    return acc

# ---------- IO helpers ----------
def write_raster_like(template_ds, out_path, array, dtype, nodata=None):
    profile = template_ds.meta.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

# ---------- Selection rules ----------
def select_conversions_area_match(prob, can_convert, n_to_convert, stochastic=False, rng=None):
    if n_to_convert <= 0:
        return np.zeros_like(can_convert, dtype=bool)
    flat_prob = prob[can_convert]
    if flat_prob.size == 0:
        return np.zeros_like(can_convert, dtype=bool)

    if stochastic:
        # probability-proportional sampling without replacement
        p = flat_prob - flat_prob.min()
        p = p / (p.sum() + 1e-12)
        n = min(n_to_convert, flat_prob.size)
        idx_local = rng.choice(flat_prob.size, size=n, replace=False, p=p)
    else:
        n = min(n_to_convert, flat_prob.size)
        idx_local = np.argpartition(flat_prob, -n)[-n:]

    sel_mask_filtered = np.zeros(flat_prob.shape, dtype=bool)
    sel_mask_filtered[idx_local] = True

    sel_mask_full = np.zeros_like(can_convert, dtype=bool)
    idxs = np.flatnonzero(can_convert)
    sel_idxs = idxs[sel_mask_filtered]
    sel_mask_full.flat[sel_idxs] = True
    return sel_mask_full

# ---------- Main run ----------
def run(cfg=None):
    """
    Config keys (in addition to your previous ones):

    # Multi-scale neighborhoods:
    "multi_neighborhood_sizes": [3,5]      # list of odd ints
    "multi_neighborhood_weights": [0.7,0.3]# same length, sum to ~1

    # Anisotropy:
    "kernel_mode": "isotropic"|"anisotropic"
    "anisotropy_axis": "ew"|"ns"
    "anisotropy_strength": float(>=1)

    # Road bias (optional):
    "use_road_bias": true|false
    "road_distance_path": "data/aligned/static/dist_roads_aligned.tif"
    "road_max_m": 8000
    "road_weight": 0.2        # added to z before sigmoid (after scaling to 0..1)

    # Stochastic & ensemble:
    "stochastic_threshold": true
    "stochastic_area_match": false
    "random_seed": 42
    "ensemble_runs": 1
    "ensemble_vote_threshold": 0.5
    """
    if cfg is None:
        with open("configs/ca_config.json", "r") as f:
            cfg = json.load(f)

    start_year = int(cfg["start_year"])
    end_year   = int(cfg["end_year"])
    steps = end_year - start_year
    if steps <= 0:
        raise ValueError("end_year must be greater than start_year")

    # Defaults for new options
    sizes = cfg.get("multi_neighborhood_sizes", [cfg.get("neighborhood_size", 3)])
    weights = cfg.get("multi_neighborhood_weights", [1.0] * len(sizes))
    kernel_mode = cfg.get("kernel_mode", "isotropic")
    anis_axis = cfg.get("anisotropy_axis", "ew")
    anis_strength = float(cfg.get("anisotropy_strength", 1.5))
    use_road_bias = bool(cfg.get("use_road_bias", False))
    road_path = cfg.get("road_distance_path", "data/aligned/static/dist_roads_aligned.tif")
    road_max_m = float(cfg.get("road_max_m", 8000.0))
    road_weight = float(cfg.get("road_weight", 0.0))
    ensemble_runs = int(cfg.get("ensemble_runs", 1))
    ensemble_vote_threshold = float(cfg.get("ensemble_vote_threshold", 0.5))

    period_dir = os.path.join(OUT_DIR, f"{cfg['run_name']}_{start_year}_{end_year}")
    os.makedirs(period_dir, exist_ok=True)

    def single_run(seed_offset=0):
        with rasterio.open(cfg["start_urban_path"]) as ds_u0, \
             rasterio.open(cfg["suitability_path"]) as ds_s:

            u0    = ds_u0.read(1)
            suit  = ds_s.read(1).astype(np.float32)

            # Mask (1=developable, 0=restricted)
            if cfg.get("mask_path") and os.path.exists(cfg["mask_path"]):
                with rasterio.open(cfg["mask_path"]) as ds_mask:
                    mask = ds_mask.read(1) > 0
            else:
                mask = np.ones_like(u0, dtype=bool)

            # Optional road bias (proximity → 0..1; closer = higher)
            road_bias = 0.0
            if use_road_bias and os.path.exists(road_path):
                with rasterio.open(road_path) as ds_road:
                    dist = ds_road.read(1).astype(np.float32)
                dist = np.clip(dist, 0, road_max_m)
                road_bias = 1.0 - (dist / (road_max_m + 1e-9))  # 0..1

            # Area-match requirements
            if cfg["conversion_rule"] == "area_match" or cfg["conversion_rule"] == "area_match_hybrid":
                with rasterio.open(cfg["observed_change_path"]) as ds_chg:
                    change_obs = ds_chg.read(1)
                total_new = int((change_obs == 1).sum())
                per_step  = int(total_new / steps)
                spillover = total_new - per_step * steps
            else:
                per_step  = 0
                spillover = 0

            rng = np.random.default_rng(int(cfg.get("random_seed", 42)) + seed_offset)

            urban       = (u0 > 0)
            developable = mask

            for t in range(steps):
                nb = neighborhood_index(urban, sizes=sizes, weights=weights,
                                        mode=kernel_mode, anis_axis=anis_axis, anis_strength=anis_strength)

                # Linear combination → logits
                z = cfg["a"] * suit + cfg["b"] * nb + cfg["c"]

                # add road bias (scaled) if requested
                if use_road_bias and road_weight > 0:
                    z = z + road_weight * np.asarray(road_bias, dtype=np.float32)

                # temperature scaling and logistic
                prob = sigmoid(z / float(cfg["temperature"]))

                # stochastic threshold (adds small noise)
                if bool(cfg.get("stochastic_threshold", True)):
                    eps = rng.normal(loc=0.0, scale=0.01, size=prob.shape).astype(np.float32)
                    prob = np.clip(prob + eps, 0.0, 1.0)

                can_convert = (~urban) & developable

                if cfg["conversion_rule"] == "threshold":
                    pth = float(cfg.get("p_thresh", 0.5))
                    to_convert = (prob >= pth) & can_convert
                elif cfg["conversion_rule"] in ("area_match", "area_match_hybrid"):
                    k = per_step + (spillover if t == steps - 1 else 0)
                    stoch_area = bool(cfg.get("stochastic_area_match", False))
                    if cfg["conversion_rule"] == "area_match_hybrid":
                        # Enforce a minimum probability to avoid extremely low-suitability conversions
                        min_p = float(cfg.get("hybrid_area_min_prob", 0.3))
                        can_convert = can_convert & (prob >= min_p)
                    to_convert = select_conversions_area_match(prob, can_convert, k,
                                                               stochastic=stoch_area, rng=rng)
                else:
                    raise ValueError(f"Unknown conversion_rule: {cfg['conversion_rule']}")

                urban[to_convert] = True

                if cfg.get("save_yearly", True):
                    year = start_year + t + 1
                    out_path = os.path.join(period_dir, f"urban_{year}_sim.tif")
                    write_raster_like(ds_u0, out_path, urban.astype(np.uint8), dtype=rasterio.uint8, nodata=0)

            # final map for this run
            final = urban.astype(np.uint8)
            return final

    # Ensemble runs (if ensemble_runs==1, this is just one run)
    runs = []
    for i in range(max(1, ensemble_runs)):
        runs.append(single_run(seed_offset=i))

    # Aggregate
    if len(runs) == 1:
        final_urban = runs[0]
    else:
        stack = np.stack(runs, axis=0).astype(np.float32)  # [R,H,W] with 0/1
        prob_mean = stack.mean(axis=0)                     # mean occupancy
        thr = float(ensemble_vote_threshold)
        final_urban = (prob_mean >= thr).astype(np.uint8)

    # Save final
    with rasterio.open(cfg["start_urban_path"]) as ds_u0:
        final_out = os.path.join(OUT_DIR, f"{cfg['run_name']}_{start_year}_{end_year}", f"urban_{end_year}_sim.tif")
        os.makedirs(os.path.dirname(final_out), exist_ok=True)
        write_raster_like(ds_u0, final_out, final_urban, dtype=rasterio.uint8, nodata=0)
        print(f"Saved final map: {final_out}")
    print("=== CA run complete ===\n")

if __name__ == "__main__":
    run()
