# 🏙️ Urban Growth Modelling & Cost of Sprawl – Sri Lanka

This project presents a **comprehensive spatial modelling framework** to simulate and evaluate urban growth dynamics in Sri Lanka up to **2050**, integrating:

- 🧠 **Machine Learning (LightGBM)** for land suitability prediction,  
- 🧩 **Cellular Automata (CA)** for spatial diffusion of urban growth,  
- 👥 **Agent-Based Modelling (ABM)** for human decision-making (developers & households),  
- 🧾 **Scenario-based forecasting** for policy interventions, and  
- 💰 **Cost of Sprawl analysis** to quantify fiscal and environmental impacts.

---

## 🎯 Objectives

- Model past urbanization patterns (2000–2020) using observed data.  
- Simulate future growth under different policy scenarios (2020–2050).  
- Integrate behavioral and market-driven logic via ABM.  
- Quantify **uncertainty**, **costs**, and **fiscal impact** of urban expansion.

---

## 🧠 Methodological Framework

### 1️⃣ Data Preparation

All spatial datasets were aligned to a **common 30 m grid** to ensure pixel-level consistency.  
Inputs included:
- Land Use / Land Cover (LULC)  
- NDVI, LST, Population Density  
- Distance to Roads, Railways, CBDs  
- Slope (derived from DEM using Sobel filter)  
- Land Price per perch  
- Restricted masks (protected areas, water, etc.)

**Tools used:** `rasterio`, `numpy`, `scipy.ndimage`, `pandas`

**Output:**  
- Aligned raster layers (2000, 2010, 2020)  
- Change maps (e.g., 2000→2010, 2010→2020) identifying *new urban pixels*

> *Why alignment matters:*  
> CA and ABM simulate at cell level — even a 1-pixel offset causes spatial bias in probability and neighborhood calculations.

---

### 2️⃣ Machine Learning Suitability Modelling (LightGBM)

A **LightGBM classifier** was trained to predict where new urbanization occurred during 2000→2010.  
This model learned the spatial relationships between multiple explanatory variables and urban conversion.

**Features:**
- NDVI (greenness)  
- LST (surface temperature)  
- Population density  
- Slope & Elevation  
- Distance to roads, rail, CBDs  
- Land price  
- LULC type

**Target:**  
`1 = newly urbanized pixel`, `0 = non-urban pixel`

**Handling imbalance:** downsampled negatives to maintain learning stability.  
**Output:** continuous suitability maps (0–1) for 2000, 2010, 2020.

> *LightGBM learns non-linear interactions — for example:*  
> “Cells close to roads, with moderate land price and low NDVI, are more likely to urbanize.”

**Tools:** `lightgbm`, `scikit-learn`, `pandas`, `numpy`

---

### 3️⃣ Cellular Automata (CA) Simulation

The CA simulates how urbanization spreads spatially from existing urban centers:

\[
p = \sigma(a \cdot Suitability + b \cdot Neighborhood + c)
\]

where:  
- σ = sigmoid function (bounded 0–1)  
- *a*, *b*, *c* = coefficients  
- *T* = temperature, controls randomness (higher = smoother)  

The **Neighborhood index** captures the density of surrounding urban cells, supporting:
- **Isotropic kernels:** equal influence in all directions.  
- **Anisotropic kernels:** stronger in one axis (e.g., N–S corridor).  
- **Multi-scale neighborhoods:** 3×3, 5×5, 7×7 with weighted influence.

**Rules implemented:**
1. **Threshold rule** – cells urbanize if p ≥ threshold.  
2. **Area-match rule** – each year converts *k* cells (observed average), prioritizing those with higher p.

**Tools:** `numpy`, `scipy.ndimage.convolve`

---

### 4️⃣ CA Calibration & Validation

**Goal:** Tune CA parameters (a, b, c, T, kernel size, anisotropy) for best spatial agreement with observed change.

**Optimization method:**  
`Optuna` (Bayesian search) maximizing a custom score:

\[
Score = FoM - 0.5(Q + A)
\]

- **FoM (Figure of Merit):** overlap between observed and simulated new urban areas.  
- **Q (Quantity):** error in total amount of change.  
- **A (Allocation):** error in spatial allocation.

**Validation (2010→2020):**  
Ran the CA with best parameters, reported FoM, Q, A, precision, recall, and F1.

> *Result:*  
> The CA correctly predicted the magnitude of growth, but with moderate FoM — growth patterns matched general regions, though exact parcels varied.

---

### 5️⃣ Agent-Based Model (ABM) – Human Decision Layer

The **ABM** introduces behavioral logic by modelling two agents:

#### Developer logic:
- Favors **road proximity** and **low land price**
- Profit = 0.6 × (road proximity) + 0.4 × (1 – land price)

#### Household logic:
- Favors **high suitability**, **CBD proximity**, and **good environment**
- Utility = 0.5 × suitability + 0.3 × CBD proximity + 0.2 × environment

#### Combined ABM weight:
\[
W = p_{CA} \times Profit \times Utility
\]

Cells with high combined weight have higher conversion probability.  
ABM introduces realism — growth is influenced by accessibility and affordability, not just spatial diffusion.

**Tools:** `numpy`, `optuna`, `pandas`

---

### 6️⃣ ABM Calibration & Validation

Calibrated developer and household weights:
\[
(w_{road}, w_{land}, w_s, w_d, w_e)
\]
using **Optuna** to maximize FoM (Figure of Merit) on the 2010→2020 validation.

**Key takeaway:**
- ABM logic improved behavioral realism.  
- CA-only gave higher FoM (more spatially accurate),  
- ABM–CA produced *more plausible growth narratives* aligned with how developers and households act.

---

### 7️⃣ Scenario Simulation (2020→2050)

**Scenarios modelled:**

| Scenario | Logic |
|-----------|--------|
| **BAU (Business As Usual)** | Base suitability |
| **Compact** | +20% boost near existing urban (encourages densification) |
| **Transit-Oriented** | +30% near railways |
| **Highway Bias** | +30% near roads |
| **Farmland Protection** | Set suitability = 0 for farmland |

Each scenario was simulated **year by year (2020–2050)** using the ABM–CA model.

**Outputs per scenario:**
- Annual urban maps (2021–2050)  
- Final 2050 urban map  
- Growth summary CSV (urban cells per year)

---

### 8️⃣ Uncertainty Analysis (Monte Carlo)

**Goal:** measure how random variations affect model outcomes.

- Each scenario simulated **20 runs** with different random seeds.  
- Aggregated to:
  - **Probability Map:** fraction of runs that urbanized per pixel (0–1)  
  - **Uncertainty Map:** standard deviation (spatial variability)  
  - **Summary stats:** mean / min / max total urban area (2050)

> *Interpretation:*  
> - High probability = consistently urbanized → robust forecast.  
> - High uncertainty = unstable zones → sensitive to behavior/policy.

---

### 9️⃣ Cost of Sprawl (Fiscal Impact Analysis)

The **Cost of Sprawl model** monetizes the infrastructure and environmental impacts of new urban expansion.

#### Cost Components:

| Type | Formula | Proxy Dataset |
|-------|----------|----------------|
| **Infrastructure Cost** | ∝ distance to nearest road × road cost per km | Distance to roads |
| **Service Cost** | = new population × service cost per person | Population density |
| **Environmental Cost** | ∝ NDVI × environmental value per NDVI | NDVI raster |
| **Revenue** | = new population × average tax revenue per person | Population density |

All values are expressed in **millions of Sri Lankan Rupees (LKR)**.

**Output per scenario:**
- Infrastructure cost  
- Service delivery cost  
- Environmental cost  
- Fiscal Impact Analysis (FIA = Revenue – Service Cost)  
- Total cost summary CSV

> *Why it matters:*  
> This quantifies not just *where* cities grow, but *how expensive or sustainable* each growth pattern is.

---

## ⚙️ Running the Model

Run each module in sequence:

```bash
# 1. Train suitability model (ML)
python scripts/suitability_ml_lgbm.py

# 2. Calibrate and validate CA
python scripts/calibration_ca.py
python scripts/validation_ca.py

# 3. Calibrate and validate ABM
python scripts/calibration_abm.py
python scripts/validation_abm.py

# 4. Scenario simulation to 2050
python scripts/scenarios.py

# 5. Uncertainty (Monte Carlo)
python scripts/uncertainty.py

# 6. Cost of Sprawl estimation
python scripts/costs.py
```
---

## 📂 Folder Structure

```plaintext
Spatial-Modelling/
│
├── data/
│   ├── raw/                  → Original input rasters (NDVI, LST, LULC, etc.)
│   ├── aligned/              → Aligned & resampled layers for modelling
│   └── static/               → Distance maps, masks, land price, etc.
│
├── scripts/
│   ├── data_prep.py          → Align and preprocess raster data
│   ├── suitability_lgbm.py   → ML model for urban suitability
│   ├── ca_model.py           → Core Cellular Automata engine
│   ├── abm_model.py          → Agent-based behavioural layer
│   ├── calibration.py        → CA calibration using Optuna
│   ├── validation.py         → CA validation (FoM, Q/A)
│   ├── calibration_abm.py    → ABM parameter optimization
│   ├── validation_abm.py     → ABM validation (2010→2020)
│   ├── scenarios.py          → Scenario simulations (BAU, Compact, etc.)
│   ├── uncertainty.py        → Monte Carlo analysis (probability/uncertainty maps)
│   ├── costs.py              → Cost of sprawl estimation (future step)
│   ├── reporting.py          → Generate analysis plots and tables
│   └── reporting_pptx.py     → Auto-generate PowerPoint presentation
│
├── outputs/
│   ├── suitability/          → ML feature importance & suitability rasters
│   ├── calibration/          → Best-fit CA parameters
│   ├── validation/           → Validation metrics & confusion matrices
│   ├── calibration_abm/      → Best-fit ABM parameters
│   ├── scenarios/            → Scenario maps & summaries
│   ├── uncertainty/          → Probability & uncertainty maps
│   └── report/               → Final PowerPoint & CSV summaries
│
└── README.md
```

## 🧮 Tools & Libraries

| **Category** | **Libraries / Frameworks** |
|---------------|-----------------------------|
| GIS & Raster | Rasterio, GDAL, NumPy |
| Machine Learning | LightGBM, Scikit-learn |
| Spatial Simulation | CA, ABM, SciPy |
| Optimization | Optuna |
| Visualization | Matplotlib, Seaborn |
| Reporting | Python-PPTX, Pandas |

---

## 📊 Key Outputs

- ML-based **Suitability Maps** (0–1 probability)
- **CA & ABM simulation maps** (annual + 2050)
- **Scenario summaries** with costs and uncertainty
- **Probability & Uncertainty maps** from Monte Carlo analysis
- **Cost of Sprawl** summary table (in Million LKR)
- Automatically generated **PowerPoint Report**

---

## 🧾 Example Metrics (2010→2020 Validation)

| **Model** | **FoM** | **Precision** | **Recall** | **F1** | **Q** | **A** |
|------------|----------|---------------|-------------|--------|--------|--------|
| CA-only | 0.163 | 0.281 | 0.281 | 0.281 | 0.000008 | 0.075 |
| ABM–CA Hybrid | 0.059 | 0.112 | 0.112 | 0.112 | 0.000034 | 0.093 |

**Interpretation:**  
The CA baseline is more spatially accurate (higher FoM),  
while ABM–CA shifts growth to more behaviorally plausible zones — reflecting market and accessibility dynamics.

---

## 🧭 Future Improvements

- Integrate **dynamic population projections** and economic growth data  
- Include **climate risk** (e.g., flood, heat) into suitability  
- Add **multi-objective calibration** for better FoM–Q–A balance  
- Extend **FIA** with lifecycle and energy cost modules  
- Visualize results via **interactive dashboards** (e.g., Streamlit or Kepler.gl)

---

## ✍️ Author

**Developed by:**
1. Thulnith Sathviru
2. Visva Devmini
3. Thirasari Perera+

**Project:** Spatial Modelling – Cost of Sprawl  
**Institution:** *University of Moratuwa*  
**Year:** 2025

