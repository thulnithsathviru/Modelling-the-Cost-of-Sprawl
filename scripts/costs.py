import os
import json
import numpy as np
import pandas as pd
import rasterio

RESULTS_DIR = "outputs/costs"
os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------------
# Utility functions
# -------------------------
def load_raster(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds


# -------------------------
# Cost calculation
# -------------------------
def compute_costs(new_cells, dist_roads, pop_density, ndvi, land_value, params):
    """
    Compute TOTAL costs for newly urbanized cells (2020→2050).
    Returns values in millions of Sri Lankan Rupees (LKR).
    """

    # --- Clean distance raster ---
    dist = dist_roads.astype(float)
    dist[dist <= 0] = np.nan
    dist[dist > 1e6] = np.nan  # ignore absurd distances

    # --- Infrastructure cost ---
    infra_cost = np.nansum(dist[new_cells] / 1000 * params["road_cost_per_km"])

    # --- Population-related costs ---
    pop_new = np.nansum(pop_density[new_cells])
    service_cost = pop_new * params["service_cost_per_person"] if np.isfinite(pop_new) else 0
    revenue = pop_new * params["tax_revenue_per_person"] if np.isfinite(pop_new) else 0

    # --- Environmental cost ---
    env_cost = np.nansum(ndvi[new_cells]) * params["env_cost_per_ndvi"]

    # --- Land acquisition / opportunity cost ---
    # Land value raster is Rs per perch. Convert to per cell using area_per_cell_perches.
    land_cost = np.nansum(land_value[new_cells]) * params["area_per_cell_perches"]

    # --- Totals ---
    total_cost = infra_cost + service_cost + env_cost + land_cost
    fia = revenue - service_cost

    # Replace NaNs with 0
    def safe_val(x): return 0 if (x is None or not np.isfinite(x)) else x

    return {
        "infra_cost_million": round(safe_val(infra_cost) / 1e6, 2),
        "service_cost_million": round(safe_val(service_cost) / 1e6, 2),
        "revenue_million": round(safe_val(revenue) / 1e6, 2),
        "env_cost_million": round(safe_val(env_cost) / 1e6, 2),
        "land_cost_million": round(safe_val(land_cost) / 1e6, 2),
        "total_cost_million": round(safe_val(total_cost) / 1e6, 2),
        "fia_million": round(safe_val(fia) / 1e6, 2),
        "pop_new": int(round(safe_val(pop_new)))
    }


# -------------------------
# Main routine
# -------------------------
def run():
    # Load parameters
    with open("configs/costs_config.json", "r") as f:
        params = json.load(f)

    # Static layers
    dist_roads, _ = load_raster("data/aligned/static/dist_roads_aligned.tif")
    pop2020, _ = load_raster("data/aligned/2020/pop_2020_aligned.tif")
    ndvi2020, _ = load_raster("data/aligned/2020/ndvi_2020_aligned.tif")
    land_value, _ = load_raster("data/aligned/static/land_price_aligned.tif")  # Rs per perch

    # Scenarios (ABM-CA runs)
    scenarios = ["BAU", "Compact", "TransitOriented", "HighwayBias", "FarmlandProtection"]

    results = []

    for sc in scenarios:
        print(f"\n=== Calculating TOTAL costs for scenario: {sc} ===")
        sc_dir = os.path.join("outputs/scenarios", f"{sc}_2020_2050")

        # Start from 2020 urban baseline
        prev_path = os.path.join(sc_dir, "urban_2020_sim.tif")
        if not os.path.exists(prev_path):
            prev_path = "data/aligned/2020/urban_2020_aligned.tif"
        prev_urban, ds_ref = load_raster(prev_path)

        # Load final year (2050)
        final_path = os.path.join(sc_dir, "urban_2050_sim.tif")
        if not os.path.exists(final_path):
            print(f"⚠ Skipping {sc}, no 2050 output found")
            continue
        final_urban, _ = load_raster(final_path)

        # All new urban cells from 2020→2050
        new_cells = (final_urban == 1) & (prev_urban == 0)

        # Population growth factor by 2050 (linear)
        pop_factor = (2050 - 2020) / 30.0 + 1.0  # ~2x by 2050
        pop_density = pop2020 * pop_factor

        # Normalize NDVI for environmental cost
        ndvi = ndvi2020 / np.nanmax(ndvi2020)

        cost_dict = compute_costs(new_cells, dist_roads, pop_density, ndvi, land_value, params)
        cost_dict.update({
            "scenario": sc,
            "new_cells": int(new_cells.sum())
        })

        results.append(cost_dict)

    # Save results
    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "costs_summary_2050.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n=== TOTAL Costs (2020→2050, in millions of LKR) ===")
    print(df.to_string(index=False))
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    run()
