import os
import json
import numpy as np
import rasterio
import optuna
from abm_model import run as abm_run

RESULTS_DIR = "outputs/calibration_abm"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def load_map(path):
    with rasterio.open(path) as ds:
        return ds.read(1)

def compute_fom(obs, sim):
    """Figure of Merit = intersection / union for change=1."""
    obs = (obs == 1)
    sim = (sim == 1)
    inter = np.logical_and(obs, sim).sum()
    union = obs.sum() + sim.sum() - inter
    return inter / union if union > 0 else 0.0

def quantity_allocation_disagreement(obs, sim):
    """Pontius-style disagreement metrics (binary)."""
    y_true = (obs == 1).ravel()
    y_pred = (sim == 1).ravel()
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

def simulate(cfg):
    """Run ABM model and return simulated change map (new urban=1)."""
    abm_run(cfg)
    sim_path = os.path.join(
        "outputs/abm_ca",
        f"{cfg['run_name']}_{cfg['start_year']}_{cfg['end_year']}",
        f"urban_{cfg['end_year']}_sim.tif"
    )
    with rasterio.open(sim_path) as ds_sim:
        sim_urban = ds_sim.read(1)
    with rasterio.open(cfg["start_urban_path"]) as ds_start:
        start = ds_start.read(1)
    return ((sim_urban == 1) & (start == 0)).astype(np.uint8)

# -------------------------
# Bayesian Optimization
# -------------------------
def run_bayes(n_trials=50):
    with open("configs/abm_config.json", "r") as f:
        base_cfg = json.load(f)

    obs_change = load_map(base_cfg["observed_change_path"])

    def objective(trial):
        cfg = base_cfg.copy()
        cfg.update({
            "w_road": trial.suggest_float("w_road", 0.2, 1.0),
            "w_land": trial.suggest_float("w_land", 0.2, 1.0),
            "w_s": trial.suggest_float("w_s", 0.2, 0.8),
            "w_d": trial.suggest_float("w_d", 0.1, 0.5),
            "w_e": trial.suggest_float("w_e", 0.1, 0.5),
            "run_name": f"ABM_CAL_t{trial.number}"
        })

        sim_change = simulate(cfg)
        fom = compute_fom(obs_change, sim_change)
        Q, A = quantity_allocation_disagreement(obs_change, sim_change)

        # Combined score (FoM − penalty for Q + A)
        score = fom - 0.5 * (Q + A)

        # Store metrics for later inspection
        trial.set_user_attr("metrics", {
            "FoM": fom, "Q": Q, "A": A,
            "sim_new": int(sim_change.sum()),
            "obs_new": int((obs_change == 1).sum())
        })

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    metrics = best.user_attrs.get("metrics", {})
    out_json = os.path.join(RESULTS_DIR, "abm_best.json")
    with open(out_json, "w") as f:
        json.dump({
            "params": best.params,
            "score": best.value,
            "metrics": metrics
        }, f, indent=2)

    print("\n=== ABM Bayesian Calibration: Best Trial ===")
    print(json.dumps({
        "params": best.params,
        "score": best.value,
        "metrics": metrics
    }, indent=2))
    print(f"\nSaved to {out_json}")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    run_bayes(80)  # increase trials for more accuracy
