import os
import json
import numpy as np
import rasterio
from ca_model import run as ca_run
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

RESULTS_DIR = "outputs/validation"
SUIT_DIR = "outputs/suitability_ml_lgbm"  # use LightGBM suitability
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_fom(obs_change, sim_change):
    obs = (obs_change == 1)
    pred = (sim_change == 1)
    inter = np.logical_and(obs, pred).sum()
    union = obs.sum() + pred.sum() - inter
    return inter / union if union > 0 else 0.0

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

def load_map(path):
    with rasterio.open(path) as ds:
        return ds.read(1)

def run():
    best_path = "outputs/calibration/bayes_joint_best.json"
    if not os.path.exists(best_path):
        raise FileNotFoundError("Run calibration first (outputs/calibration/bayes_joint_best.json)")
    with open(best_path, "r") as f:
        best = json.load(f)
    params = best["params"]
    print(f"Loaded best parameters from {best_path}")

    cfg = {
        "start_year": 2010,
        "end_year": 2020,
        "start_urban_path": "data/aligned/2010/urban_2010_aligned.tif",
        "suitability_path": os.path.join(SUIT_DIR, "suitability_2010.tif"),
        "mask_path": "data/aligned/static/mask_restricted_aligned.tif",
        "observed_change_path": "data/aligned/change_2010_2020_aligned.tif",

        "a": params["a"],
        "b": params["b"],
        "c": params["c"],
        "temperature": params["T"],

        # multi-scale
        "multi_neighborhood_sizes": params.get("sizes", [params.get("N", 3)]),
        "multi_neighborhood_weights": params.get("weights", [1.0]),

        # anisotropy
        "kernel_mode": params.get("kernel_mode", "isotropic"),
        "anisotropy_axis": params.get("anis_axis", params.get("anisotropy_axis", "ns")),
        "anisotropy_strength": params.get("anis_strength", params.get("anisotropy_strength", 1.5)),

        # road bias
        "use_road_bias": params.get("use_road_bias", False),
        "road_distance_path": "data/aligned/static/dist_roads_aligned.tif",
        "road_weight": params.get("road_weight", 0.0),
        "road_max_m": params.get("road_max_m", 8000.0),

        "conversion_rule": "area_match_hybrid",
        "hybrid_area_min_prob": params.get("hybrid_min_prob", 0.35),
        "stochastic_threshold": True,
        "stochastic_area_match": params.get("stoch_area", False),

        "random_seed": 42,
        "save_yearly": True,
        "save_prob_yearly": False,
        "run_name": "validation_2010_2020"
    }

    ca_run(cfg)

    obs_change = load_map(cfg["observed_change_path"])
    sim_path = os.path.join("outputs/ca",
                            f"{cfg['run_name']}_{cfg['start_year']}_{cfg['end_year']}",
                            f"urban_{cfg['end_year']}_sim.tif")
    with rasterio.open(sim_path) as ds_sim:
        sim_urban = ds_sim.read(1)
    with rasterio.open(cfg["start_urban_path"]) as ds_start:
        start_urban = ds_start.read(1)

    sim_change = ((sim_urban == 1) & (start_urban == 0)).astype(np.uint8)

    fom = compute_fom(obs_change, sim_change)
    Q, A = quantity_allocation_disagreement(obs_change, sim_change)

    y_true = obs_change.flatten()
    y_pred = sim_change.flatten()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    results = {
        "FoM": float(fom),
        "Q": float(Q),
        "A": float(A),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "sim_new": int(sim_change.sum()),
        "obs_new": int(obs_change.sum())
    }

    out_path = os.path.join(RESULTS_DIR, "validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Validation (2010→2020) Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run()
