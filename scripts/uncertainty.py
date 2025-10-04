import os
import json
import numpy as np
import rasterio
import pandas as pd
from scenarios import build_scenario_suitability, run_abm_ca_2020_2050, load_calibrated_params

RESULTS_DIR = "outputs/uncertainty"
SUIT_DIR = "outputs/suitability_ml_lgbm"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_raster(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds

def save_raster_like(template_ds, out_path, array, dtype=rasterio.float32, nodata=None):
    profile = template_ds.meta.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

def run():
    # Load calibrated CA + ABM params
    ca_params, abm_params = load_calibrated_params()

    base_suit_path = os.path.join(SUIT_DIR, "suitability_2020.tif")
    scenarios = ["BAU", "Compact", "TransitOriented", "HighwayBias", "FarmlandProtection"]

    N = 20  # number of Monte Carlo runs per scenario
    summary = []
    global_stack = []   # 🔹 store all runs across all scenarios
    template_ds = None

    for sc in scenarios:
        print(f"\n=== Uncertainty: {sc} ===")

        # Build scenario suitability
        suit_mod, temp_ds = build_scenario_suitability(base_suit_path, sc)
        if template_ds is None:
            template_ds = temp_ds  # keep reference for saving rasters

        # Collect runs into stack
        totals = []
        for i in range(N):
            out_dir = os.path.join(RESULTS_DIR, f"{sc}_run{i}")
            os.makedirs(out_dir, exist_ok=True)

            final_tif, _ = run_abm_ca_2020_2050(
                sc, suit_mod, temp_ds, ca_params, abm_params, out_dir, seed=i
            )

            urban2050, _ = load_raster(final_tif)
            global_stack.append(urban2050.astype(np.uint8))
            totals.append(int((urban2050 == 1).sum()))

        # Scenario-level stats
        mean_area = int(np.mean(totals))
        min_area = int(np.min(totals))
        max_area = int(np.max(totals))

        summary.append({
            "scenario": sc,
            "runs": N,
            "mean_urban_cells": mean_area,
            "min_urban_cells": min_area,
            "max_urban_cells": max_area
        })

    # 🔹 Combine all runs across all scenarios
    global_stack = np.stack(global_stack, axis=0)

    prob_map = np.mean(global_stack, axis=0)   # overall probability
    std_map = np.std(global_stack, axis=0)     # overall uncertainty

    # Save overall probability & uncertainty maps
    prob_out = os.path.join(RESULTS_DIR, "overall_probability_2050.tif")
    std_out = os.path.join(RESULTS_DIR, "overall_uncertainty_2050.tif")
    save_raster_like(template_ds, prob_out, prob_map, dtype=rasterio.float32, nodata=None)
    save_raster_like(template_ds, std_out, std_map, dtype=rasterio.float32, nodata=None)

    print(f"\n✅ Saved overall probability map → {prob_out}")
    print(f"✅ Saved overall uncertainty map → {std_out}")

    # Save summary CSV
    df = pd.DataFrame(summary)
    out_csv = os.path.join(RESULTS_DIR, "uncertainty_summary.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== Uncertainty analysis complete ===")
    print(df.to_string(index=False))
    print(f"\nScenario-wise results saved to {out_csv}")

if __name__ == "__main__":
    run()
