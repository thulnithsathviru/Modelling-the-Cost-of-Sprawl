import os
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import convolve
import lightgbm as lgb
import joblib

ALIGNED_DIR = "data/aligned"
OUTPUT_DIR = "outputs/suitability_ml_lgbm"
YEARS = ["2000", "2010", "2020"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_neighborhood_density(urban_array, size):
    kernel = np.ones((size, size), dtype=np.float32)
    kernel[size // 2, size // 2] = 0.0
    counts = convolve((urban_array > 0).astype(np.float32), kernel, mode="constant", cval=0.0)
    return counts / kernel.sum()

def load_factors(year):
    year_dir = os.path.join(ALIGNED_DIR, year)
    rasters = {
        "lulc": os.path.join(year_dir, f"lulc_{year}_aligned.tif"),
        "population": os.path.join(year_dir, f"pop_{year}_aligned.tif"),
        "ndvi": os.path.join(year_dir, f"ndvi_{year}_aligned.tif"),
        "lst": os.path.join(year_dir, f"lst_{year}_aligned.tif"),
        "dist_roads": os.path.join(ALIGNED_DIR, "static", "dist_roads_aligned.tif"),
        "dist_rail": os.path.join(ALIGNED_DIR, "static", "dist_rail_aligned.tif"),
        "dist_cbds": os.path.join(ALIGNED_DIR, "static", "dist_cbds_aligned.tif"),
        "dist_services": os.path.join(ALIGNED_DIR, "static", "dist_services_aligned.tif"),
        "slope": os.path.join(ALIGNED_DIR, "static", "slope_aligned.tif"),
    }

    arrays, profile = {}, None
    for k, path in rasters.items():
        with rasterio.open(path) as src:
            arrays[k] = src.read(1).astype(np.float32)
            if profile is None:
                profile = src.meta.copy()

    # multi-scale neighborhood densities from base urban of that year
    with rasterio.open(os.path.join(year_dir, f"urban_{year}_aligned.tif")) as u:
        urb = u.read(1)

    arrays["nbhd3"] = compute_neighborhood_density(urb, 3)
    arrays["nbhd5"] = compute_neighborhood_density(urb, 5)
    arrays["nbhd7"] = compute_neighborhood_density(urb, 7)

    return arrays, profile

def raster_to_df(factors, change):
    df = pd.DataFrame({k: v.flatten() for k, v in factors.items()})
    df["label"] = change.flatten()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def build_training_data():
    print("Building training data (2000→2010)...")
    factors2000, _ = load_factors("2000")
    with rasterio.open(os.path.join(ALIGNED_DIR, "change_2000_2010_aligned.tif")) as src:
        change = src.read(1)
    df = raster_to_df(factors2000, change)

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0].sample(n=min(len(df[df["label"] == 0]), len(pos) * 2), random_state=42)
    df_bal = pd.concat([pos, neg]).sample(frac=1, random_state=42)
    return df_bal

def train_model(df):
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    dtrain = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbose": -1
    }
    model = lgb.train(params, dtrain, num_boost_round=800)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "lgbm_model.pkl"))
    return model

def apply_model(model):
    for year in YEARS:
        print(f"Generating suitability for {year}...")
        factors, profile = load_factors(year)
        X = np.column_stack([factors[k].flatten() for k in factors])
        preds = model.predict(X)
        suitability = preds.reshape(factors["lulc"].shape)

        out_prob = os.path.join(OUTPUT_DIR, f"suitability_{year}.tif")
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(out_prob, "w", **profile) as dst:
            dst.write(suitability.astype(np.float32), 1)
        print(f"Saved: {out_prob}")

def run():
    df = build_training_data()
    model = train_model(df)
    apply_model(model)
    print("\n=== LightGBM + Multi-scale Neighborhood Suitability Completed ===")

if __name__ == "__main__":
    run()
