import os
import json
import itertools
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Optional

from ca_model import run as ca_run

RESULTS_DIR = "outputs/calibration"
SUIT_DIR = "outputs/suitability_ml_lgbm"  # LightGBM suitability only
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Metrics
# -------------------------
def load_map(path: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)

def compute_fom(obs_change: np.ndarray, sim_change: np.ndarray) -> float:
    obs = (obs_change == 1)
    pred = (sim_change == 1)
    inter = np.logical_and(obs, pred).sum()
    union = obs.sum() + pred.sum() - inter
    return float(inter) / float(union) if union > 0 else 0.0

def binary_confusion(obs_change: np.ndarray, sim_change: np.ndarray) -> Tuple[int,int,int,int]:
    y_true = (obs_change == 1).ravel()
    y_pred = (sim_change == 1).ravel()
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn

def quantity_allocation_disagreement(obs_change: np.ndarray, sim_change: np.ndarray) -> Tuple[float,float]:
    tp, fp, fn, tn = binary_confusion(obs_change, sim_change)
    N = tp + fp + fn + tn
    if N == 0:
        return 0.0, 0.0
    Q = abs(fp - fn) / N
    A = (2.0 * min(fp, fn)) / N
    return float(Q), float(A)

# -------------------------
# CA wrapper
# -------------------------
def simulate_change_from_ca(cfg: Dict[str, Any]) -> np.ndarray:
    ca_run(cfg)
    sim_path = os.path.join("outputs/ca",
                            f"{cfg['run_name']}_{cfg['start_year']}_{cfg['end_year']}",
                            f"urban_{cfg['end_year']}_sim.tif")
    with rasterio.open(sim_path) as ds_sim:
        sim_urban = ds_sim.read(1)
    with rasterio.open(cfg["start_urban_path"]) as ds_start:
        start_urban = ds_start.read(1)
    sim_change = ((sim_urban == 1) & (start_urban == 0)).astype(np.uint8)
    return sim_change

# -------------------------
# Period paths
# -------------------------
def period_paths(start_year: int, end_year: int) -> Dict[str, str]:
    if start_year == 2000 and end_year == 2010:
        return dict(
            start_urban="data/aligned/2000/urban_2000_aligned.tif",
            suitability=os.path.join(SUIT_DIR, "suitability_2000.tif"),
            change="data/aligned/change_2000_2010_aligned.tif",
        )
    elif start_year == 2010 and end_year == 2020:
        return dict(
            start_urban="data/aligned/2010/urban_2010_aligned.tif",
            suitability=os.path.join(SUIT_DIR, "suitability_2010.tif"),
            change="data/aligned/change_2010_2020_aligned.tif",
        )
    else:
        raise ValueError("Unsupported period; extend period_paths().")

def _verify_inputs(cfg: Dict[str, Any]):
    for key in ["start_urban_path", "suitability_path", "observed_change_path", "mask_path"]:
        p = cfg.get(key)
        if not p or not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing '{key}': {p}\nExpected LightGBM suitability in {SUIT_DIR}"
            )

def make_cfg(base_cfg: Dict[str, Any],
             a: float, b: float, c: float, T: float,
             sizes: List[int], weights: List[float],
             kernel_mode: str, anis_axis: str, anis_strength: float,
             rule: str, hybrid_min_prob: float, stoch_area: bool,
             use_road_bias: bool, road_weight: float, road_max_m: float,
             start_year: int, end_year: int, run_tag: str) -> Dict[str, Any]:
    P = period_paths(start_year, end_year)
    weights = np.array(weights, dtype=np.float32)
    weights = (weights / (weights.sum() + 1e-9)).tolist()

    cfg = base_cfg.copy()
    cfg.update({
        "a": a, "b": b, "c": c,
        "temperature": T,

        # multi-scale neighborhoods
        "multi_neighborhood_sizes": sizes,
        "multi_neighborhood_weights": weights,

        # anisotropy
        "kernel_mode": kernel_mode,
        "anisotropy_axis": anis_axis,
        "anisotropy_strength": anis_strength,

        # road bias
        "use_road_bias": use_road_bias,
        "road_distance_path": "data/aligned/static/dist_roads_aligned.tif",
        "road_weight": road_weight,
        "road_max_m": road_max_m,

        # rule / stochastic
        "conversion_rule": rule,
        "hybrid_area_min_prob": hybrid_min_prob,
        "stochastic_area_match": stoch_area,
        "stochastic_threshold": True,
        "random_seed": 42,

        # IO / period
        "save_yearly": False,
        "save_prob_yearly": False,
        "run_name": run_tag,
        "start_year": start_year, "end_year": end_year,
        "start_urban_path": P["start_urban"],
        "suitability_path": P["suitability"],
        "observed_change_path": P["change"],
        "mask_path": base_cfg.get("mask_path", "data/aligned/static/mask_restricted_aligned.tif"),
    })
    _verify_inputs(cfg)
    return cfg

# -------------------------
# Evaluate both periods
# -------------------------
def evaluate_both_periods(base_cfg: Dict[str, Any],
                          a: float, b: float, c: float, T: float,
                          sizes: List[int], weights: List[float],
                          kernel_mode: str, anis_axis: str, anis_strength: float,
                          rule: str, hybrid_min_prob: float, stoch_area: bool,
                          use_road_bias: bool, road_weight: float, road_max_m: float,
                          tag: str, weight_2010: float = 0.5, weight_2020: float = 0.5) -> Dict[str, Any]:

    cfg_0010 = make_cfg(base_cfg, a,b,c,T, sizes,weights, kernel_mode,anis_axis,anis_strength,
                        rule,hybrid_min_prob,stoch_area, use_road_bias,road_weight,road_max_m,
                        2000,2010, f"CAL_0010_{tag}")
    sim_change_0010 = simulate_change_from_ca(cfg_0010)
    obs_change_0010 = load_map(cfg_0010["observed_change_path"])
    FoM_0010 = compute_fom(obs_change_0010, sim_change_0010)
    Q_0010, A_0010 = quantity_allocation_disagreement(obs_change_0010, sim_change_0010)

    cfg_1020 = make_cfg(base_cfg, a,b,c,T, sizes,weights, kernel_mode,anis_axis,anis_strength,
                        rule,hybrid_min_prob,stoch_area, use_road_bias,road_weight,road_max_m,
                        2010,2020, f"CAL_1020_{tag}")
    sim_change_1020 = simulate_change_from_ca(cfg_1020)
    obs_change_1020 = load_map(cfg_1020["observed_change_path"])
    FoM_1020 = compute_fom(obs_change_1020, sim_change_1020)
    Q_1020, A_1020 = quantity_allocation_disagreement(obs_change_1020, sim_change_1020)

    FoM_comb = weight_2010 * FoM_0010 + weight_2020 * FoM_1020
    Q_comb   = weight_2010 * Q_0010   + weight_2020 * Q_1020
    A_comb   = weight_2010 * A_0010   + weight_2020 * A_1020
    score    = FoM_comb - 0.5 * (Q_comb + A_comb)

    return {
        "FoM_0010": FoM_0010, "Q_0010": Q_0010, "A_0010": A_0010,
        "FoM_1020": FoM_1020, "Q_1020": Q_1020, "A_1020": A_1020,
        "FoM_comb": FoM_comb, "Q_comb": Q_comb, "A_comb": A_comb,
        "score": score,
    }

# -------------------------
# Grid (optional)
# -------------------------
def default_grids(mode: str = "quick") -> Dict[str, List]:
    if mode == "quick":
        return {
            "a": [0.6, 1.0, 1.4],
            "b": [0.2, 0.5, 0.8],
            "c": [-0.2, 0.0, 0.2],
            "T": [0.9, 1.1],
            "sizes": [[3,5], [3,7], [5,7]],
            "w_small": [0.4, 0.6, 0.8],  # w_large = 1 - w_small
            "kernel_mode": ["isotropic", "anisotropic"],
            "anis_axis": ["ew", "ns"],
            "anis_strength": [1.5, 2.0],
            "rule": ["area_match_hybrid"],
            "hybrid_min_prob": [0.30, 0.36, 0.42],
            "stoch_area": [True, False],
            "use_road_bias": [True, False],
            "road_weight": [0.0, 0.1, 0.2],
            "road_max_m": [5000, 8000],
        }
    else:
        return {
            "a": [0.4, 0.8, 1.2, 1.6],
            "b": [0.1, 0.3, 0.5, 0.7, 0.9],
            "c": [-0.4, -0.2, 0.0, 0.2, 0.4],
            "T": [0.7, 0.9, 1.1, 1.3],
            "sizes": [[3,5], [3,7], [5,7], [3,5,7]],
            "w_small": [0.3, 0.5, 0.7, 0.85],
            "kernel_mode": ["isotropic", "anisotropic"],
            "anis_axis": ["ew", "ns"],
            "anis_strength": [1.5, 2.0, 2.5],
            "rule": ["area_match_hybrid"],
            "hybrid_min_prob": [0.25, 0.30, 0.35, 0.40, 0.45],
            "stoch_area": [True, False],
            "use_road_bias": [True, False],
            "road_weight": [0.0, 0.1, 0.2, 0.3],
            "road_max_m": [4000, 6000, 8000, 10000],
        }

def run_grid(mode: str = "quick", max_combos_warn: int = 5000):
    with open("configs/ca_config.json", "r") as f:
        base_cfg = json.load(f)

    grid = default_grids(mode)
    keys = ["a","b","c","T","sizes","w_small","kernel_mode","anis_axis","anis_strength",
            "rule","hybrid_min_prob","stoch_area","use_road_bias","road_weight","road_max_m"]
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)
    print(f"Testing {total} parameter sets ({mode} grid)")
    if total > max_combos_warn:
        print(f"WARNING: {total} combos > {max_combos_warn}. Consider Bayesian optimization.")

    rows = []
    for combo in tqdm(combos):
        params = dict(zip(keys, combo))
        sizes = params["sizes"]
        if len(sizes) == 2:
            weights = [params["w_small"], 1.0 - params["w_small"]]
        else:
            # for 3 scales: split (w_small, w_mid, w_large) from w_small; simple heuristic
            w1 = params["w_small"]
            w2 = (1.0 - w1) * 0.6
            w3 = 1.0 - w1 - w2
            weights = [w1, w2, w3]

        tag = ("GS_" +
               f"a{params['a']}_b{params['b']}_c{params['c']}_T{params['T']}_S{','.join(map(str,sizes))}_" +
               f"W{','.join([f'{w:.2f}' for w in weights])}_K{params['kernel_mode']}_AX{params['anis_axis']}_AS{params['anis_strength']}_" +
               f"RB{params['use_road_bias']}_RW{params['road_weight']}_RM{params['road_max_m']}_H{params['hybrid_min_prob']}_SA{params['stoch_area']}")

        metrics = evaluate_both_periods(
            base_cfg,
            params["a"], params["b"], params["c"], params["T"],
            sizes, weights,
            params["kernel_mode"], params["anis_axis"], params["anis_strength"],
            params["rule"], params["hybrid_min_prob"], params["stoch_area"],
            params["use_road_bias"], params["road_weight"], params["road_max_m"],
            tag
        )
        rows.append({**params, **{"weights": weights}, **metrics})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, f"calibration_joint_{mode}.csv")
    df.to_csv(out_csv, index=False)

    top = df.sort_values(["FoM_comb","FoM_1020","FoM_0010"], ascending=False).head(15)
    print("\n=== Top 15 (joint calibration) ===")
    print(top[["a","b","c","T","sizes","weights","kernel_mode","anis_axis","anis_strength",
               "use_road_bias","road_weight","road_max_m",
               "hybrid_min_prob","stoch_area",
               "FoM_0010","Q_0010","A_0010","FoM_1020","Q_1020","A_1020",
               "FoM_comb","Q_comb","A_comb","score"]])
    print(f"\nResults saved to {out_csv}")

# -------------------------
# Bayesian optimization (recommended)
# -------------------------
def run_bayes(n_trials: int = 60, w0010: float = 0.5, w1020: float = 0.5):
    try:
        import optuna
    except Exception:
        print("Optuna not available. Install with: pip install optuna")
        return run_grid(mode="quick")

    with open("configs/ca_config.json", "r") as f:
        base_cfg = json.load(f)

    def objective(trial: "optuna.Trial") -> float:
        a  = trial.suggest_float("a", 0.3, 1.8)
        b  = trial.suggest_float("b", 0.1, 0.9)
        c  = trial.suggest_float("c", -0.5, 0.5)
        T  = trial.suggest_float("T", 0.6, 1.6)

        # choose scale set
        sizes = trial.suggest_categorical("sizes", [[3,5], [3,7], [5,7], [3,5,7]])
        if len(sizes) == 2:
            w_small = trial.suggest_float("w_small", 0.2, 0.85)
            weights = [w_small, 1.0 - w_small]
        else:
            # 3-scale weights: sample two, derive third
            w1 = trial.suggest_float("w1", 0.2, 0.8)
            w2 = trial.suggest_float("w2", 0.1, 0.7)
            s  = max(w1 + w2, 1e-6)
            if s >= 1.0:
                w1 = w1 / (s + 1e-6) * 0.9
                w2 = w2 / (s + 1e-6) * 0.9
            w3 = 1.0 - (w1 + w2)
            weights = [w1, w2, w3]

        kernel_mode = trial.suggest_categorical("kernel_mode", ["isotropic","anisotropic"])
        anis_axis = trial.suggest_categorical("anis_axis", ["ew","ns"])
        anis_strength = trial.suggest_float("anis_strength", 1.2, 3.0)

        use_road_bias = trial.suggest_categorical("use_road_bias", [True, False])
        road_weight = trial.suggest_float("road_weight", 0.0, 0.35) if use_road_bias else 0.0
        road_max_m = trial.suggest_float("road_max_m", 4000, 12000) if use_road_bias else 8000.0

        rule = "area_match_hybrid"
        hybrid_min_prob = trial.suggest_float("hybrid_min_prob", 0.20, 0.50)
        stoch_area = trial.suggest_categorical("stoch_area", [True, False])

        tag = f"BO_t{trial.number}"
        metrics = evaluate_both_periods(
            base_cfg, a,b,c,T,
            sizes, weights,
            kernel_mode, anis_axis, anis_strength,
            rule, hybrid_min_prob, stoch_area,
            use_road_bias, road_weight, road_max_m,
            tag, weight_2010=w0010, weight_2020=w1020
        )
        trial.set_user_attr("metrics", metrics)
        return metrics["score"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True)

    best = study.best_trial
    best_metrics = best.user_attrs.get("metrics", {})
    out_json = os.path.join(RESULTS_DIR, "bayes_joint_best.json")
    with open(out_json, "w") as f:
        json.dump({"params": best.params, "score": best.value, "metrics": best_metrics}, f, indent=2)

    print("\n=== Joint Bayesian Calibration: Best Trial ===")
    print(json.dumps({"params": best.params, "score": best.value, "metrics": best_metrics}, indent=2))
    print(f"\nSaved to {out_json}")

# -------------------------
# Entrypoint
# -------------------------
def run(mode: str = "quick", bayes_trials: Optional[int] = None):
    if bayes_trials:
        run_bayes(n_trials=int(bayes_trials))
    else:
        run_grid(mode=mode)

if __name__ == "__main__":
    # run(mode="quick")
    run(bayes_trials=100)
