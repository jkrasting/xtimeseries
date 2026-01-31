#!/usr/bin/env python
"""
New Brunswick, NJ Precipitation Extremes Analysis.

This script demonstrates extreme value analysis of observed daily precipitation
from the NOAA Cooperative Observer station at New Brunswick, NJ. The analysis
includes:

- Extraction of annual maximum precipitation
- Stationary GEV fitting with diagnostic plots
- Trend detection using likelihood ratio test
- Non-stationary GEV fitting (if trend is significant)
- Time-varying return levels
- Effective return period calculations
- Bootstrap confidence intervals for trend uncertainty

Data source: NOAA Climate Data Online (CDO)
Station: GHCND:USC00286055 (New Brunswick 3 SE, NJ)

Usage:
    python examples/newbrunswick_precipitation.py

Requires data file: tests/data/noaa_new_brunswick.csv
Run scripts/fetch_noaa_data.py first if data file is missing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xtimeseries as xts

# ============================================================================
# Configuration
# ============================================================================

DATA_FILE = Path(__file__).parent.parent / "tests" / "data" / "noaa_new_brunswick.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "_static"
SAVE_FIGURES = True


def load_data() -> pd.DataFrame:
    """Load and prepare NOAA precipitation data."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}\n"
            "Run 'python scripts/fetch_noaa_data.py' to download the data."
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["date"], index_col="date")
    print(f"Loaded {len(df)} daily records")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Variables: {', '.join(df.columns)}")

    return df


def extract_annual_maxima(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract annual maximum precipitation."""
    # Get precipitation column
    if "PRCP" in df.columns:
        prcp = df["PRCP"]
    else:
        raise KeyError("Precipitation column 'PRCP' not found in data")

    # Resample to annual maximum
    annual_max = prcp.resample("YE").max()

    # Remove years with missing data
    annual_max = annual_max.dropna()

    # Extract values and years
    values = annual_max.values
    years = annual_max.index.year.values

    print(f"\nAnnual maxima extracted:")
    print(f"  Years: {years[0]} to {years[-1]} ({len(years)} years)")
    print(f"  Min: {values.min():.2f} inches")
    print(f"  Max: {values.max():.2f} inches")
    print(f"  Mean: {values.mean():.2f} inches")

    return values, years


def plot_annual_maxima(values: np.ndarray, years: np.ndarray, trend_line: np.ndarray = None):
    """Plot annual maximum precipitation time series."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(years, values, alpha=0.7, s=40, label="Annual maximum", zorder=3)

    if trend_line is not None:
        ax.plot(years, trend_line, "r-", linewidth=2, label="Trend", zorder=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Maximum Precipitation (inches)")
    ax.set_title("Annual Maximum Daily Precipitation - New Brunswick, NJ")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / "newbrunswick_annual_max.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'newbrunswick_annual_max.png'}")

    return fig, ax


def fit_stationary_gev(values: np.ndarray) -> dict:
    """Fit stationary GEV distribution."""
    print("\n" + "=" * 60)
    print("Stationary GEV Fit")
    print("=" * 60)

    params = xts.fit_gev(values)

    print(f"Location (μ): {params['loc']:.2f} inches")
    print(f"Scale (σ):    {params['scale']:.2f} inches")
    print(f"Shape (ξ):    {params['shape']:.3f}")

    # Interpret shape parameter
    if params["shape"] > 0.05:
        print("  → Fréchet type (heavy tail, unbounded)")
    elif params["shape"] < -0.05:
        print("  → Weibull type (bounded upper tail)")
    else:
        print("  → Approximately Gumbel type (exponential tail)")

    return params


def plot_diagnostics(values: np.ndarray, params: dict):
    """Create 4-panel diagnostic plot."""
    fig, axes = xts.diagnostic_plots(
        values,
        loc=params["loc"],
        scale=params["scale"],
        shape=params["shape"],
        figsize=(10, 10),
    )

    fig.suptitle("GEV Diagnostic Plots - New Brunswick Precipitation", fontsize=14, y=1.02)

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "newbrunswick_diagnostics.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'newbrunswick_diagnostics.png'}")

    return fig, axes


def test_for_trend(values: np.ndarray, years: np.ndarray) -> dict:
    """Test for significant trend using likelihood ratio test."""
    print("\n" + "=" * 60)
    print("Likelihood Ratio Test for Trend")
    print("=" * 60)

    result = xts.likelihood_ratio_test(values, years, trend_in="loc")

    print(f"Test statistic (D):       {result['statistic']:.3f}")
    print(f"Degrees of freedom:       {result['df']}")
    print(f"p-value:                  {result['p_value']:.4f}")
    print(f"Significant at α=0.05:    {'Yes' if result['significant'] else 'No'}")
    print(f"AIC (stationary):         {result['aic_stationary']:.1f}")
    print(f"AIC (non-stationary):     {result['aic_nonstationary']:.1f}")
    print(f"Preferred model (AIC):    {result['preferred_model']}")

    return result


def plot_trend_test(values: np.ndarray, years: np.ndarray, lrt_result: dict):
    """Plot likelihood ratio test visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Time series with simple linear trend
    ax1 = axes[0]
    ax1.scatter(years, values, alpha=0.7, s=40, label="Annual maximum")

    # Add linear regression line
    slope, intercept = np.polyfit(years, values, 1)
    trend_line = intercept + slope * years
    ax1.plot(years, trend_line, "r--", linewidth=2, label=f"Linear trend: {slope:.3f} inches/year")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual Maximum (inches)")
    ax1.set_title("Annual Maxima with Linear Trend")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Bar chart comparing models
    ax2 = axes[1]
    models = ["Stationary", "Non-stationary"]
    aic_values = [lrt_result["aic_stationary"], lrt_result["aic_nonstationary"]]
    colors = ["C0" if lrt_result["preferred_model"] == "stationary" else "C1",
              "C1" if lrt_result["preferred_model"] == "nonstationary" else "C0"]

    bars = ax2.bar(models, aic_values, color=colors, alpha=0.7, edgecolor="black")

    # Mark preferred model
    preferred_idx = 0 if lrt_result["preferred_model"] == "stationary" else 1
    bars[preferred_idx].set_edgecolor("green")
    bars[preferred_idx].set_linewidth(3)

    ax2.set_ylabel("AIC (lower is better)")
    ax2.set_title(f"Model Comparison (p = {lrt_result['p_value']:.4f})")

    # Add text annotation
    result_text = "Significant trend" if lrt_result["significant"] else "No significant trend"
    ax2.annotate(
        result_text,
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="green" if lrt_result["significant"] else "gray",
    )

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "newbrunswick_trend_test.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'newbrunswick_trend_test.png'}")

    return fig, axes


def fit_nonstationary_gev(values: np.ndarray, years: np.ndarray) -> dict:
    """Fit non-stationary GEV with trend in location."""
    print("\n" + "=" * 60)
    print("Non-Stationary GEV Fit")
    print("=" * 60)

    params = xts.fit_nonstationary_gev(values, years, trend_in="loc")

    print(f"Location intercept (μ₀): {params['loc0']:.2f} inches")
    print(f"Location trend (μ₁):     {params['loc1']:.4f} inches/year")
    print(f"Scale (exp(σ₀)):         {np.exp(params['scale0']):.2f} inches")
    print(f"Shape (ξ):               {params['shape']:.3f}")

    # Interpret trend
    trend_per_decade = params["loc1"] * 10
    total_change = params["loc1"] * (years[-1] - years[0])
    print(f"\nTrend interpretation:")
    print(f"  Change per decade:     {trend_per_decade:+.3f} inches/decade")
    print(f"  Total change ({years[0]}-{years[-1]}): {total_change:+.2f} inches")

    return params


def plot_nonstationary_fit(values: np.ndarray, years: np.ndarray, params: dict):
    """Plot non-stationary GEV fit."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Observations
    ax.scatter(years, values, alpha=0.7, s=40, label="Annual maximum", zorder=3)

    # Time-varying location parameter
    loc_t = params["loc0"] + params["loc1"] * years
    ax.plot(years, loc_t, "r-", linewidth=2, label="Location μ(t)", zorder=2)

    # Quantile bands (5th and 95th percentiles)
    scale_t = np.exp(params["scale0"] + params["scale1"] * years)
    from scipy import stats

    for q, label, ls in [(0.05, "5th percentile", "--"), (0.95, "95th percentile", "--")]:
        quantile = stats.genextreme.ppf(q, c=-params["shape"], loc=loc_t, scale=scale_t)
        ax.plot(years, quantile, ls, color="blue", alpha=0.5, label=label, zorder=1)

    ax.fill_between(
        years,
        stats.genextreme.ppf(0.05, c=-params["shape"], loc=loc_t, scale=scale_t),
        stats.genextreme.ppf(0.95, c=-params["shape"], loc=loc_t, scale=scale_t),
        alpha=0.2,
        color="blue",
        label="5-95% range",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Maximum Precipitation (inches)")
    ax.set_title("Non-Stationary GEV Fit - New Brunswick Precipitation")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "newbrunswick_nonstationary.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'newbrunswick_nonstationary.png'}")

    return fig, ax


def plot_return_levels(years: np.ndarray, params: dict):
    """Plot return level curves for different reference years."""
    fig, ax = plt.subplots(figsize=(10, 6))

    return_periods = np.array([2, 5, 10, 20, 50, 100, 200])

    # Reference years to compare
    ref_years = [years[0], years[len(years) // 2], years[-1]]
    colors = ["blue", "green", "red"]
    labels = ["Start of record", "Mid-record", "End of record"]

    for year, color, label in zip(ref_years, colors, labels):
        rl = xts.nonstationary_return_level(return_periods, params, year)
        ax.semilogx(return_periods, rl, "-o", color=color, linewidth=2, markersize=6, label=f"{year} ({label})")

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level (inches)")
    ax.set_title("Return Level Curves by Reference Year")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "newbrunswick_return_levels.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'newbrunswick_return_levels.png'}")

    return fig, ax


def compute_effective_return_periods(years: np.ndarray, params: dict):
    """Calculate effective return periods showing how extremes are changing."""
    print("\n" + "=" * 60)
    print("Effective Return Periods")
    print("=" * 60)

    eff = xts.effective_return_level(
        params,
        reference_value=years[0],
        future_value=years[-1],
        return_periods=[10, 20, 50, 100],
    )

    print(f"Reference year: {years[0]}")
    print(f"Current year:   {years[-1]}")
    print("\nHow historical events are changing:")
    print(f"{'Historical Event':>20} {'Now Becomes':>20}")
    print("-" * 45)

    for i, rp in enumerate(eff["return_periods"]):
        eff_period = eff["effective_period"][i]
        print(f"{rp:>3}-year event -> {eff_period:>8.1f}-year event")

    return eff


def plot_effective_return_periods(eff: dict):
    """Plot effective return period changes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Return level change
    ax1 = axes[0]
    x = np.arange(len(eff["return_periods"]))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, eff["reference_levels"], width, label=f"{int(eff['reference_value'])}", color="C0")
    bars2 = ax1.bar(x + width / 2, eff["future_levels"], width, label=f"{int(eff['future_value'])}", color="C1")

    ax1.set_xlabel("Return Period (years)")
    ax1.set_ylabel("Return Level (inches)")
    ax1.set_title("Return Levels: Then vs Now")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(rp)) for rp in eff["return_periods"]])
    ax1.legend(title="Year")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Effective return period
    ax2 = axes[1]
    ax2.bar(x, eff["effective_period"], color="C2", alpha=0.7, edgecolor="black")

    # Add reference line at original return periods
    for i, rp in enumerate(eff["return_periods"]):
        ax2.hlines(rp, i - 0.4, i + 0.4, colors="red", linestyles="--", linewidth=2)

    ax2.set_xlabel("Original Return Period (years)")
    ax2.set_ylabel("Effective Return Period (years)")
    ax2.set_title("What Historical Events Are Now")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(rp)) for rp in eff["return_periods"]])
    ax2.grid(True, alpha=0.3, axis="y")

    # Add annotation
    ax2.annotate(
        "Red lines = original period\nBars = current effective period",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
    )

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "newbrunswick_effective.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'newbrunswick_effective.png'}")

    return fig, axes


def bootstrap_trend_uncertainty(values: np.ndarray, years: np.ndarray, n_bootstrap: int = 1000):
    """Estimate uncertainty in trend parameter using bootstrap."""
    print("\n" + "=" * 60)
    print("Bootstrap Confidence Intervals for Trend")
    print("=" * 60)

    rng = np.random.default_rng(42)
    trends = []

    print(f"Running {n_bootstrap} bootstrap replicates...")

    for i in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(len(values), size=len(values), replace=True)
        boot_values = values[idx]
        boot_years = years[idx]

        try:
            boot_params = xts.fit_nonstationary_gev(boot_values, boot_years, trend_in="loc")
            trends.append(boot_params["loc1"])
        except Exception:
            pass  # Skip failed fits

    trends = np.array(trends)

    # Calculate statistics
    trend_mean = np.mean(trends)
    trend_std = np.std(trends)
    ci_lower = np.percentile(trends, 2.5)
    ci_upper = np.percentile(trends, 97.5)

    print(f"Point estimate:     {trends[0]:.5f} inches/year")
    print(f"Bootstrap mean:     {trend_mean:.5f} inches/year")
    print(f"Bootstrap std:      {trend_std:.5f} inches/year")
    print(f"95% CI:             [{ci_lower:.5f}, {ci_upper:.5f}]")

    # Is zero in the confidence interval?
    if ci_lower > 0:
        print("\nConclusion: Trend is significantly positive (zero not in 95% CI)")
        significant = True
    elif ci_upper < 0:
        print("\nConclusion: Trend is significantly negative (zero not in 95% CI)")
        significant = True
    else:
        print("\nConclusion: Trend is NOT statistically significant (zero in 95% CI)")
        significant = False

    return {
        "trends": trends,
        "mean": trend_mean,
        "std": trend_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
    }


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("New Brunswick Precipitation Extreme Value Analysis")
    print("=" * 60)

    # Load data
    df = load_data()

    # Extract annual maxima
    values, years = extract_annual_maxima(df)

    # Plot annual maxima time series
    plot_annual_maxima(values, years)

    # Fit stationary GEV
    params_stat = fit_stationary_gev(values)

    # Generate diagnostic plots
    plot_diagnostics(values, params_stat)

    # Test for trend
    lrt_result = test_for_trend(values, years)
    plot_trend_test(values, years, lrt_result)

    # If significant trend, fit non-stationary model
    if lrt_result["significant"] or lrt_result["preferred_model"] == "nonstationary":
        params_ns = fit_nonstationary_gev(values, years)
        plot_nonstationary_fit(values, years, params_ns)
        plot_return_levels(years, params_ns)
        eff = compute_effective_return_periods(years, params_ns)
        plot_effective_return_periods(eff)
        bootstrap_trend_uncertainty(values, years, n_bootstrap=500)
    else:
        print("\nNo significant trend detected. Using stationary model.")
        print("Skipping non-stationary analysis.")

    # Calculate return levels for stationary model
    print("\n" + "=" * 60)
    print("Return Levels (Stationary Model)")
    print("=" * 60)

    for rp in [10, 50, 100]:
        rl = xts.return_level(rp, loc=params_stat["loc"], scale=params_stat["scale"], shape=params_stat["shape"])
        print(f"{rp:3d}-year return level: {rl:.2f} inches")

    plt.show()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
