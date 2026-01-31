#!/usr/bin/env python
"""
Climate Model Precipitation Comparison for New Brunswick, NJ.

This script demonstrates extreme value analysis comparing observed precipitation
with CMIP6 climate model output (GFDL-CM4 and CESM2). The analysis includes:

- Loading and aligning observed and model precipitation data
- Extracting annual maximum precipitation from all sources
- Comparing stationary GEV parameters (location, scale, shape)
- Comparing return level curves
- Testing for trends in each dataset
- Assessing model skill (bias, tail behavior)

Data sources:
- Observations: NOAA Climate Data Online (CDO)
- Models: CMIP6 historical experiment via Pangeo/Google Cloud

Usage:
    python examples/model_comparison.py

Requires data files:
- tests/data/noaa_new_brunswick.csv
- tests/data/cmip6_gfdl-cm4_pr.nc
- tests/data/cmip6_cesm2_pr.nc

Run fetch scripts first if data files are missing:
- python scripts/fetch_noaa_data.py
- python scripts/fetch_cmip6_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xtimeseries as xts

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "tests" / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "_static"
SAVE_FIGURES = True

# Overlapping period for fair comparison
START_YEAR = 1950
END_YEAR = 2014  # CMIP6 historical typically ends here


def load_observations() -> tuple[np.ndarray, np.ndarray]:
    """Load observed precipitation data."""
    obs_file = DATA_DIR / "noaa_new_brunswick.csv"

    if not obs_file.exists():
        raise FileNotFoundError(
            f"Observation data not found: {obs_file}\n"
            "Run 'python scripts/fetch_noaa_data.py' to download."
        )

    df = pd.read_csv(obs_file, parse_dates=["date"], index_col="date")

    # Get precipitation and extract annual maximum
    # Data is already in inches (standard NOAA units)
    prcp = df["PRCP"].dropna()

    annual_max = prcp.resample("YE").max()

    # Filter to overlapping period
    annual_max = annual_max[(annual_max.index.year >= START_YEAR) & (annual_max.index.year <= END_YEAR)]

    values = annual_max.values
    years = annual_max.index.year.values

    print(f"Observations: {len(values)} years ({years[0]}-{years[-1]})")
    print(f"  Range: {values.min():.2f} - {values.max():.2f} inches")

    return values, years


def load_model_data(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load CMIP6 model precipitation data."""
    model_file = DATA_DIR / f"cmip6_{model_name.lower()}_pr.nc"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model data not found: {model_file}\n"
            "Run 'python scripts/fetch_cmip6_data.py' to download."
        )

    ds = xr.open_dataset(model_file)
    da = ds["pr"]

    # Squeeze out singleton dimensions (member_id, dcpp_init_year)
    da = da.squeeze(drop=True)

    # Convert units: kg m-2 s-1 to inches/day
    # 1 kg/m2 = 1 mm of water, 86400 seconds/day, 25.4 mm/inch
    da = da * 86400 / 25.4

    # Extract annual maximum
    annual_max = da.resample(time="YE").max()

    # Filter to overlapping period
    years = annual_max["time"].dt.year.values
    mask = (years >= START_YEAR) & (years <= END_YEAR)
    annual_max = annual_max.isel(time=mask)

    values = annual_max.values.flatten()
    years = annual_max["time"].dt.year.values

    # Remove NaN
    valid = ~np.isnan(values)
    values = values[valid]
    years = years[valid]

    print(f"{model_name}: {len(values)} years ({years[0]}-{years[-1]})")
    print(f"  Range: {values.min():.2f} - {values.max():.2f} inches/day")

    return values, years


def plot_timeseries(data_dict: dict):
    """Plot time series comparison of all data sources."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"Observations": "black", "GFDL-CM4": "C0", "CESM2": "C1"}
    markers = {"Observations": "o", "GFDL-CM4": "s", "CESM2": "^"}

    for name, (values, years) in data_dict.items():
        ax.plot(
            years,
            values,
            color=colors[name],
            marker=markers[name],
            markersize=4,
            linewidth=1,
            alpha=0.7,
            label=name,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Maximum Precipitation (inches/day)")
    ax.set_title("Annual Maximum Precipitation: Observations vs Climate Models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / "model_comparison_timeseries.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'model_comparison_timeseries.png'}")

    return fig, ax


def fit_gev_all(data_dict: dict) -> dict:
    """Fit stationary GEV to all datasets."""
    print("\n" + "=" * 60)
    print("Stationary GEV Parameters")
    print("=" * 60)

    results = {}

    print(f"{'Dataset':<15} {'Location':>10} {'Scale':>10} {'Shape':>10}")
    print("-" * 50)

    for name, (values, years) in data_dict.items():
        params = xts.fit_gev(values)
        results[name] = params

        print(f"{name:<15} {params['loc']:>10.2f} {params['scale']:>10.2f} {params['shape']:>10.3f}")

    return results


def plot_parameters(gev_params: dict):
    """Plot GEV parameter comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    datasets = list(gev_params.keys())
    colors = ["black", "C0", "C1"]

    # Location
    ax = axes[0]
    locs = [gev_params[d]["loc"] for d in datasets]
    ax.bar(datasets, locs, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Location (μ)")
    ax.set_title("Location Parameter")
    ax.tick_params(axis="x", rotation=15)

    # Scale
    ax = axes[1]
    scales = [gev_params[d]["scale"] for d in datasets]
    ax.bar(datasets, scales, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Scale (σ)")
    ax.set_title("Scale Parameter")
    ax.tick_params(axis="x", rotation=15)

    # Shape
    ax = axes[2]
    shapes = [gev_params[d]["shape"] for d in datasets]
    ax.bar(datasets, shapes, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Shape (ξ)")
    ax.set_title("Shape Parameter")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "model_comparison_parameters.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'model_comparison_parameters.png'}")

    return fig, axes


def plot_return_levels(data_dict: dict, gev_params: dict):
    """Plot return level curves for all datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    return_periods = np.array([2, 5, 10, 20, 50, 100, 200, 500])
    colors = {"Observations": "black", "GFDL-CM4": "C0", "CESM2": "C1"}
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
            return_periods,
            rl,
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=2,
            label=name,
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

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "model_comparison_return_levels.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'model_comparison_return_levels.png'}")

    return fig, ax


def test_trends_all(data_dict: dict) -> dict:
    """Test for trends in all datasets."""
    print("\n" + "=" * 60)
    print("Trend Detection (Likelihood Ratio Test)")
    print("=" * 60)

    results = {}

    print(f"{'Dataset':<15} {'Statistic':>10} {'p-value':>10} {'Significant':>12} {'Preferred':>12}")
    print("-" * 65)

    for name, (values, years) in data_dict.items():
        lrt = xts.likelihood_ratio_test(values, years, trend_in="loc")
        results[name] = lrt

        sig_str = "Yes" if lrt["significant"] else "No"
        print(f"{name:<15} {lrt['statistic']:>10.3f} {lrt['p_value']:>10.4f} {sig_str:>12} {lrt['preferred_model']:>12}")

    return results


def fit_nonstationary_all(data_dict: dict, lrt_results: dict) -> dict:
    """Fit non-stationary GEV where trends are significant."""
    print("\n" + "=" * 60)
    print("Non-Stationary GEV Parameters (where significant)")
    print("=" * 60)

    ns_params = {}

    for name, (values, years) in data_dict.items():
        if lrt_results[name]["significant"] or lrt_results[name]["preferred_model"] == "nonstationary":
            params = xts.fit_nonstationary_gev(values, years, trend_in="loc")
            ns_params[name] = params

            trend_per_decade = params["loc1"] * 10
            print(f"\n{name}:")
            print(f"  Location intercept: {params['loc0']:.2f}")
            print(f"  Location trend:     {params['loc1']:.4f}/year = {trend_per_decade:+.2f}/decade")
            print(f"  Scale:              {np.exp(params['scale0']):.2f}")
            print(f"  Shape:              {params['shape']:.3f}")
        else:
            print(f"\n{name}: No significant trend, stationary model preferred")

    return ns_params


def plot_trends(data_dict: dict, ns_params: dict, lrt_results: dict):
    """Plot trend comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = list(data_dict.keys())
    colors = {"Observations": "black", "GFDL-CM4": "C0", "CESM2": "C1"}

    trends = []
    errs = []

    for name in datasets:
        if name in ns_params:
            trend = ns_params[name]["loc1"] * 10  # per decade
            trends.append(trend)
            # Approximate error (would need bootstrap for proper CI)
            errs.append(abs(trend) * 0.5)  # Placeholder
        else:
            # Use linear regression as fallback
            values, years = data_dict[name]
            slope, _ = np.polyfit(years, values, 1)
            trends.append(slope * 10)
            errs.append(abs(slope * 10) * 0.5)

    x = np.arange(len(datasets))
    bars = ax.bar(x, trends, color=[colors[d] for d in datasets], alpha=0.7, edgecolor="black")

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Trend (units per decade)")
    ax.set_title("Precipitation Trend Comparison")

    # Add significance markers
    for i, name in enumerate(datasets):
        if lrt_results[name]["significant"]:
            ax.annotate("*", xy=(i, trends[i] + 0.1), ha="center", fontsize=16, fontweight="bold")

    ax.annotate(
        "* = statistically significant (p < 0.05)",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
    )

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "model_comparison_trends.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'model_comparison_trends.png'}")

    return fig, ax


def plot_qq_comparison(data_dict: dict, gev_params: dict):
    """Plot Q-Q comparison for all datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    datasets = list(data_dict.keys())

    for ax, name in zip(axes, datasets):
        values = data_dict[name][0]
        params = gev_params[name]

        xts.qq_plot(values, loc=params["loc"], scale=params["scale"], shape=params["shape"], ax=ax)
        ax.set_title(f"Q-Q Plot: {name}")

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "model_comparison_qq.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'model_comparison_qq.png'}")

    return fig, axes


def assess_model_skill(data_dict: dict, gev_params: dict):
    """Assess model skill relative to observations."""
    print("\n" + "=" * 60)
    print("Model Skill Assessment")
    print("=" * 60)

    obs_params = gev_params["Observations"]
    obs_values = data_dict["Observations"][0]

    for model in ["GFDL-CM4", "CESM2"]:
        model_params = gev_params[model]
        model_values = data_dict[model][0]

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

        # Tail behavior assessment
        if model_params["shape"] > obs_params["shape"] + 0.1:
            print("  Tail behavior: Model has heavier tail than observations")
        elif model_params["shape"] < obs_params["shape"] - 0.1:
            print("  Tail behavior: Model has lighter tail than observations")
        else:
            print("  Tail behavior: Similar to observations")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Climate Model Precipitation Comparison")
    print(f"Analysis period: {START_YEAR}-{END_YEAR}")
    print("=" * 60)

    # Load all data
    print("\nLoading data...")

    try:
        obs_values, obs_years = load_observations()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        gfdl_values, gfdl_years = load_model_data("GFDL-CM4")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        gfdl_values, gfdl_years = None, None

    try:
        cesm_values, cesm_years = load_model_data("CESM2")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        cesm_values, cesm_years = None, None

    # Build data dictionary
    data_dict = {"Observations": (obs_values, obs_years)}
    if gfdl_values is not None:
        data_dict["GFDL-CM4"] = (gfdl_values, gfdl_years)
    if cesm_values is not None:
        data_dict["CESM2"] = (cesm_values, cesm_years)

    if len(data_dict) < 2:
        print("\nNeed at least observations and one model for comparison.")
        print("Run the data fetch scripts first.")
        return

    # Plot time series
    plot_timeseries(data_dict)

    # Fit stationary GEV
    gev_params = fit_gev_all(data_dict)
    plot_parameters(gev_params)

    # Plot return levels
    plot_return_levels(data_dict, gev_params)

    # Test for trends
    lrt_results = test_trends_all(data_dict)

    # Fit non-stationary where significant
    ns_params = fit_nonstationary_all(data_dict, lrt_results)

    # Plot trends
    plot_trends(data_dict, ns_params, lrt_results)

    # Q-Q plots
    plot_qq_comparison(data_dict, gev_params)

    # Model skill assessment
    assess_model_skill(data_dict, gev_params)

    plt.show()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
