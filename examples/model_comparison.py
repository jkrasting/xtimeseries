"""
Climate Model Precipitation Comparison
========================================

This example demonstrates extreme value analysis comparing observed precipitation
with CMIP6 climate model output (GFDL-CM4 and CESM2). The analysis includes
comparison of GEV parameters, return levels, and trend detection.

**Data sources:**

- Observations: NOAA Climate Data Online (CDO)
- Models: CMIP6 historical experiment via Pangeo/Google Cloud
"""

# %%
# Setup and Configuration
# -----------------------
# Import libraries and configure paths.

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xtimeseries as xts

# Handle path for both standalone execution and sphinx-gallery
try:
    _THIS_DIR = Path(__file__).parent
except NameError:
    import xtimeseries as _xts
    _THIS_DIR = Path(_xts.__file__).parent.parent.parent / "examples"

DATA_DIR = _THIS_DIR.parent / "tests" / "data"

# Overlapping period for fair comparison
START_YEAR = 1950
END_YEAR = 2014

# %%
# Loading Observations
# --------------------
# Load observed precipitation data from NOAA.

obs_file = DATA_DIR / "noaa_new_brunswick.csv"

if not obs_file.exists():
    raise FileNotFoundError(
        f"Observation data not found: {obs_file}\n"
        "Run 'python scripts/fetch_noaa_data.py' to download."
    )

df = pd.read_csv(obs_file, parse_dates=["date"], index_col="date")
prcp = df["PRCP"].dropna()
annual_max = prcp.resample("YE").max()
annual_max = annual_max[(annual_max.index.year >= START_YEAR) & (annual_max.index.year <= END_YEAR)]

obs_values = annual_max.values
obs_years = annual_max.index.year.values

print(f"Observations: {len(obs_values)} years ({obs_years[0]}-{obs_years[-1]})")
print(f"  Range: {obs_values.min():.2f} - {obs_values.max():.2f} inches")

# %%
# Loading Model Data
# ------------------
# Load CMIP6 model precipitation data.

data_dict = {"Observations": (obs_values, obs_years)}

for model_name in ["GFDL-CM4", "CESM2"]:
    model_file = DATA_DIR / f"cmip6_{model_name.lower()}_pr.nc"

    if not model_file.exists():
        print(f"Warning: {model_name} data not found, skipping")
        continue

    ds = xr.open_dataset(model_file)
    da = ds["pr"].squeeze(drop=True)

    # Convert units: kg m-2 s-1 to inches/day
    da = da * 86400 / 25.4

    annual_max_model = da.resample(time="YE").max()
    years_model = annual_max_model["time"].dt.year.values
    mask = (years_model >= START_YEAR) & (years_model <= END_YEAR)
    annual_max_model = annual_max_model.isel(time=mask)

    values_model = annual_max_model.values.flatten()
    years_model = annual_max_model["time"].dt.year.values

    valid = ~np.isnan(values_model)
    values_model = values_model[valid]
    years_model = years_model[valid]

    data_dict[model_name] = (values_model, years_model)

    print(f"{model_name}: {len(values_model)} years ({years_model[0]}-{years_model[-1]})")
    print(f"  Range: {values_model.min():.2f} - {values_model.max():.2f} inches/day")

# %%
# Time Series Comparison
# ----------------------
# Plot time series of all data sources.

fig, ax = plt.subplots(figsize=(12, 6))

colors = {"Observations": "black", "GFDL-CM4": "C0", "CESM2": "C1"}
markers = {"Observations": "o", "GFDL-CM4": "s", "CESM2": "^"}

for name, (values, years) in data_dict.items():
    ax.plot(
        years, values,
        color=colors[name], marker=markers[name],
        markersize=4, linewidth=1, alpha=0.7, label=name,
    )

ax.set_xlabel("Year")
ax.set_ylabel("Annual Maximum Precipitation (inches/day)")
ax.set_title("Annual Maximum Precipitation: Observations vs Climate Models")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# GEV Parameter Fitting
# ---------------------
# Fit stationary GEV to all datasets.

print("Stationary GEV Parameters:")
print(f"{'Dataset':<15} {'Location':>10} {'Scale':>10} {'Shape':>10}")
print("-" * 50)

gev_params = {}
for name, (values, years) in data_dict.items():
    params = xts.fit_gev(values)
    gev_params[name] = params
    print(f"{name:<15} {params['loc']:>10.2f} {params['scale']:>10.2f} {params['shape']:>10.3f}")

# %%
# Parameter Comparison
# --------------------
# Visualize GEV parameter differences.

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

datasets = list(gev_params.keys())
colors_list = ["black", "C0", "C1"][:len(datasets)]

# Location
ax = axes[0]
locs = [gev_params[d]["loc"] for d in datasets]
ax.bar(datasets, locs, color=colors_list, alpha=0.7, edgecolor="black")
ax.set_ylabel("Location (μ)")
ax.set_title("Location Parameter")
ax.tick_params(axis="x", rotation=15)

# Scale
ax = axes[1]
scales = [gev_params[d]["scale"] for d in datasets]
ax.bar(datasets, scales, color=colors_list, alpha=0.7, edgecolor="black")
ax.set_ylabel("Scale (σ)")
ax.set_title("Scale Parameter")
ax.tick_params(axis="x", rotation=15)

# Shape
ax = axes[2]
shapes = [gev_params[d]["shape"] for d in datasets]
ax.bar(datasets, shapes, color=colors_list, alpha=0.7, edgecolor="black")
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_ylabel("Shape (ξ)")
ax.set_title("Shape Parameter")
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.show()

# %%
# Return Level Comparison
# -----------------------
# Compare return level curves across datasets.

fig, ax = plt.subplots(figsize=(10, 6))

return_periods = np.array([2, 5, 10, 20, 50, 100, 200, 500])
linestyles = {"Observations": "-", "GFDL-CM4": "--", "CESM2": ":"}

for name in data_dict.keys():
    params = gev_params[name]
    rl = xts.return_level(
        return_periods,
        loc=params["loc"],
        scale=params["scale"],
        shape=params["shape"],
    )
    ax.semilogx(
        return_periods, rl,
        color=colors[name], linestyle=linestyles[name],
        linewidth=2, label=name,
    )

    # Add empirical points
    values = data_dict[name][0]
    n = len(values)
    sorted_vals = np.sort(values)[::-1]
    empirical_rp = (n + 1) / np.arange(1, n + 1)
    ax.scatter(empirical_rp, sorted_vals, color=colors[name], s=20, alpha=0.5, zorder=2)

ax.set_xlabel("Return Period (years)")
ax.set_ylabel("Return Level (inches/day)")
ax.set_title("Return Level Comparison: Observations vs Climate Models")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.show()

# %%
# Trend Detection
# ---------------
# Test for trends in all datasets.

print("Trend Detection (Likelihood Ratio Test):")
print(f"{'Dataset':<15} {'Statistic':>10} {'p-value':>10} {'Significant':>12} {'Preferred':>12}")
print("-" * 65)

lrt_results = {}
for name, (values, years) in data_dict.items():
    lrt = xts.likelihood_ratio_test(values, years, trend_in="loc")
    lrt_results[name] = lrt

    sig_str = "Yes" if lrt["significant"] else "No"
    print(f"{name:<15} {lrt['statistic']:>10.3f} {lrt['p_value']:>10.4f} {sig_str:>12} {lrt['preferred_model']:>12}")

# %%
# Q-Q Plot Comparison
# -------------------
# Compare Q-Q plots for all datasets.

fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4))
if len(datasets) == 1:
    axes = [axes]

for ax, name in zip(axes, datasets):
    values = data_dict[name][0]
    params = gev_params[name]
    xts.qq_plot(values, loc=params["loc"], scale=params["scale"], shape=params["shape"], ax=ax)
    ax.set_title(f"Q-Q Plot: {name}")

plt.tight_layout()
plt.show()

# %%
# Model Skill Assessment
# ----------------------
# Assess model skill relative to observations.

print("Model Skill Assessment:")

obs_params = gev_params["Observations"]

for model in ["GFDL-CM4", "CESM2"]:
    if model not in gev_params:
        continue

    model_params = gev_params[model]

    print(f"\n{model} vs Observations:")

    # Location bias
    loc_bias = model_params["loc"] - obs_params["loc"]
    loc_bias_pct = 100 * loc_bias / obs_params["loc"]
    print(f"  Location bias: {loc_bias:+.2f} ({loc_bias_pct:+.1f}%)")

    # Scale bias
    scale_bias = model_params["scale"] - obs_params["scale"]
    scale_bias_pct = 100 * scale_bias / obs_params["scale"]
    print(f"  Scale bias:    {scale_bias:+.2f} ({scale_bias_pct:+.1f}%)")

    # Shape difference
    shape_diff = model_params["shape"] - obs_params["shape"]
    print(f"  Shape diff:    {shape_diff:+.3f}")

    # 100-year return level comparison
    obs_rl100 = xts.return_level(100, **obs_params)
    model_rl100 = xts.return_level(100, **model_params)
    rl100_bias = model_rl100 - obs_rl100
    rl100_bias_pct = 100 * rl100_bias / obs_rl100
    print(f"  100-yr RL bias: {rl100_bias:+.2f} ({rl100_bias_pct:+.1f}%)")

    # Tail behavior
    if model_params["shape"] > obs_params["shape"] + 0.1:
        print("  Tail behavior: Model has heavier tail than observations")
    elif model_params["shape"] < obs_params["shape"] - 0.1:
        print("  Tail behavior: Model has lighter tail than observations")
    else:
        print("  Tail behavior: Similar to observations")

print("\nAnalysis complete!")
