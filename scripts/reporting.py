import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULTS_DIR = "outputs/report"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Utility
# -------------------------
def safe_load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def safe_load_json(path):
    return json.load(open(path, "r")) if os.path.exists(path) else None

# -------------------------
# 1. Suitability (ML)
# -------------------------
def report_suitability():
    print("\n--- Suitability Report ---")
    suit_dir = os.path.join(RESULTS_DIR, "suitability")
    os.makedirs(suit_dir, exist_ok=True)

    feat_file = "outputs/suitability_ml_lgbm/feature_importances.csv"
    metrics_file = "outputs/suitability_ml_lgbm/metrics.json"

    if os.path.exists(feat_file):
        df_feat = pd.read_csv(feat_file).sort_values("importance", ascending=False).head(10)
        plt.figure(figsize=(8,5))
        sns.barplot(x="importance", y="feature", data=df_feat, color="steelblue")
        plt.title("Top-10 Suitability Features (LGBM)")
        plt.tight_layout()
        plt.savefig(os.path.join(suit_dir, "feature_importance.png"))
        plt.close()
        print("Saved feature importance plot")

    if os.path.exists(metrics_file):
        metrics = safe_load_json(metrics_file)
        with open(os.path.join(suit_dir, "metrics_summary.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("Saved metrics summary")

# -------------------------
# 2. CA Calibration + Validation
# -------------------------
def report_ca():
    print("\n--- CA Report ---")
    ca_dir = os.path.join(RESULTS_DIR, "ca")
    os.makedirs(ca_dir, exist_ok=True)

    calib_file = "outputs/calibration/bayes_joint_best.json"
    val_file = "outputs/validation/validation_results.json"

    if os.path.exists(calib_file):
        calib = safe_load_json(calib_file)
        with open(os.path.join(ca_dir, "best_calibration.json"), "w") as f:
            json.dump(calib, f, indent=2)
        print("Saved CA calibration summary")

    if os.path.exists(val_file):
        val = safe_load_json(val_file)
        with open(os.path.join(ca_dir, "validation.json"), "w") as f:
            json.dump(val, f, indent=2)

        # confusion matrix heatmap
        cm = np.array([[val["true_negative"], val["false_positive"]],
                       [val["false_negative"], val["true_positive"]]])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
        plt.title("CA Confusion Matrix (2010→2020)")
        plt.tight_layout()
        plt.savefig(os.path.join(ca_dir, "confusion_matrix.png"))
        plt.close()
        print("Saved CA validation confusion matrix")

# -------------------------
# 3. ABM Calibration + Validation
# -------------------------
def report_abm():
    print("\n--- ABM Report ---")
    abm_dir = os.path.join(RESULTS_DIR, "abm")
    os.makedirs(abm_dir, exist_ok=True)

    calib_file = "outputs/calibration_abm/abm_best.json"
    val_file = "outputs/validation/validation_abm_results.json"

    if os.path.exists(calib_file):
        calib = safe_load_json(calib_file)
        with open(os.path.join(abm_dir, "best_calibration.json"), "w") as f:
            json.dump(calib, f, indent=2)
        print("Saved ABM calibration summary")

    if os.path.exists(val_file):
        val = safe_load_json(val_file)

        # Comparison ABM vs CA baseline
        if "baseline" in val:
            df = pd.DataFrame([
                {"model":"ABM", "FoM":val["FoM"], "Precision":val["precision"], "Recall":val["recall"], "F1":val["f1"]},
                {"model":"CA-only", "FoM":val["baseline"]["FoM"], "Precision":val["baseline"]["precision"], "Recall":val["baseline"]["recall"], "F1":val["baseline"]["f1"]}
            ])
            df_melt = df.melt("model")
            plt.figure(figsize=(8,5))
            sns.barplot(x="variable", y="value", hue="model", data=df_melt)
            plt.title("ABM vs CA-only (Validation 2010→2020)")
            plt.ylabel("Score")
            plt.tight_layout()
            plt.savefig(os.path.join(abm_dir, "abm_vs_ca.png"))
            plt.close()
            print("Saved ABM vs CA comparison plot")

        with open(os.path.join(abm_dir, "validation.json"), "w") as f:
            json.dump(val, f, indent=2)

# -------------------------
# 4. Scenarios: Costs + Uncertainty
# -------------------------
def report_scenarios():
    print("\n--- Scenario Report ---")
    sc_dir = os.path.join(RESULTS_DIR, "scenarios")
    os.makedirs(sc_dir, exist_ok=True)

    cost_file = "outputs/costs/costs_summary_2050.csv"
    unc_file = "outputs/uncertainty/uncertainty_summary.csv"

    df_costs = safe_load_csv(cost_file)
    df_unc = safe_load_csv(unc_file)

    if df_costs is not None:
        # Stacked cost breakdown
        df_costs_plot = df_costs.set_index("scenario")[["infra_cost_million","service_cost_million","env_cost_million"]]
        df_costs_plot.plot(kind="bar", stacked=True, figsize=(9,6))
        plt.ylabel("Cost (Million Rs.)")
        plt.title("Cost Breakdown by Scenario (2020→2050)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(sc_dir, "cost_breakdown.png"))
        plt.close()
        print("Saved cost breakdown plot")

    if df_unc is not None:
        # Uncertainty bar with error bars
        plt.figure(figsize=(9,6))
        plt.bar(df_unc["scenario"], df_unc["mean_urban_cells"],
                yerr=[df_unc["mean_urban_cells"]-df_unc["min_urban_cells"],
                      df_unc["max_urban_cells"]-df_unc["mean_urban_cells"]],
                capsize=5, color="orange")
        plt.ylabel("Urban Cells by 2050")
        plt.title("Uncertainty in Urban Area by Scenario")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(sc_dir, "uncertainty.png"))
        plt.close()
        print("Saved uncertainty plot")

    if df_costs is not None and df_unc is not None:
        df_summary = pd.merge(df_costs, df_unc, on="scenario", how="outer")
        out_csv = os.path.join(sc_dir, "scenarios_summary.csv")
        df_summary.to_csv(out_csv, index=False)
        print(f"Saved combined scenario summary: {out_csv}")

# -------------------------
# Master run
# -------------------------
def run():
    print("\n=== Building Final Integrated Report ===")
    report_suitability()
    report_ca()
    report_abm()
    report_scenarios()
    print("\n=== Reporting complete. All outputs saved to outputs/report/ ===")

if __name__ == "__main__":
    run()
