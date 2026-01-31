#!/usr/bin/env python
"""
Atlantic City Storm Surge Extreme Value Analysis.

This script demonstrates extreme value analysis of storm surge extremes
from the NOAA tide gauge at Atlantic City, NJ. The 110+ year record provides
an exceptional dataset for understanding changes in storm intensity.

Analysis includes:
- Loading and processing tide gauge data
- Calculating storm surge (observed - predicted tide)
- Extracting annual maximum surge
- Stationary GEV fitting with diagnostics
- Trend detection in storm surge extremes
- Non-stationary analysis (if trend significant)
- Time-varying return levels
- Distinguishing storm trends from mean sea level rise

Scientific context:
By analyzing surge (observed water level - astronomical tide prediction),
we isolate the meteorological forcing component. This allows us to detect
whether storm intensity is changing over time, independent of the background
mean sea level rise signal.

Data source: NOAA Tides & Currents CO-OPS API
Station: 8534720 (Atlantic City, NJ)

Usage:
    python examples/atlantic_city_sealevel.py

Requires: tests/data/noaa_atlantic_city_tides.csv
Run 'python scripts/fetch_noaa_tides.py' first if data file is missing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xtimeseries as xts

# ============================================================================
# Configuration
# ============================================================================

DATA_FILE = Path(__file__).parent.parent / "tests" / "data" / "noaa_atlantic_city_tides.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "_static"
SAVE_FIGURES = True

# Minimum years required for analysis
MIN_YEARS = 30


def load_data() -> pd.DataFrame:
    """Load tide gauge data."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}\n"
            "Run 'python scripts/fetch_noaa_tides.py' to download the data."
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["time"], index_col="time")

    print(f"Loaded {len(df)} hourly records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {', '.join(df.columns)}")

    return df


def prepare_surge_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare storm surge data from observed and predicted water levels."""
    print("\n" + "=" * 60)
    print("Preparing Storm Surge Data")
    print("=" * 60)

    # Check if surge already calculated
    if "surge" in df.columns:
        surge = df["surge"]
    elif "observed" in df.columns and "predicted" in df.columns:
        surge = df["observed"] - df["predicted"]
    else:
        raise KeyError("Need 'surge' or both 'observed' and 'predicted' columns")

    # Remove NaN
    surge = surge.dropna()

    print(f"Hourly surge records: {len(surge)}")
    print(f"Surge statistics (m):")
    print(f"  Min:  {surge.min():.3f}")
    print(f"  Max:  {surge.max():.3f}")
    print(f"  Mean: {surge.mean():.3f}")
    print(f"  Std:  {surge.std():.3f}")

    return surge.to_frame("surge")


def extract_daily_max_surge(surge_df: pd.DataFrame) -> pd.Series:
    """Extract daily maximum surge."""
    daily_max = surge_df["surge"].resample("D").max()
    daily_max = daily_max.dropna()

    print(f"\nDaily maximum surge records: {len(daily_max)}")

    return daily_max


def extract_annual_max_surge(daily_max: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Extract annual maximum surge."""
    annual_max = daily_max.resample("YE").max()

    # Remove years with insufficient data (< 300 days)
    day_counts = daily_max.resample("YE").count()
    valid_years = day_counts >= 300
    annual_max = annual_max[valid_years]

    # Remove NaN
    annual_max = annual_max.dropna()

    values = annual_max.values
    years = annual_max.index.year.values

    print(f"\nAnnual maximum surge:")
    print(f"  Years: {years[0]} to {years[-1]} ({len(years)} years)")
    print(f"  Min:  {values.min():.3f} m")
    print(f"  Max:  {values.max():.3f} m")
    print(f"  Mean: {values.mean():.3f} m")

    return values, years


def plot_surge_record(values: np.ndarray, years: np.ndarray, df: pd.DataFrame):
    """Plot the full storm surge record."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # Top: Annual maxima time series
    ax1 = axes[0]
    ax1.plot(years, values, "o-", markersize=4, linewidth=1, alpha=0.7, label="Annual max surge")

    # Add trend line
    slope, intercept = np.polyfit(years, values, 1)
    trend_line = intercept + slope * years
    ax1.plot(years, trend_line, "r--", linewidth=2, label=f"Trend: {slope * 1000:.2f} mm/year")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual Maximum Surge (m)")
    ax1.set_title("Storm Surge Extremes at Atlantic City, NJ (110+ year record)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Histogram
    ax2 = axes[1]
    ax2.hist(values, bins=20, density=True, alpha=0.7, color="C0", edgecolor="black")
    ax2.set_xlabel("Annual Maximum Surge (m)")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Annual Maximum Surge")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / "atlantic_city_surge_record.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'atlantic_city_surge_record.png'}")

    return fig, axes


def plot_decomposition(df: pd.DataFrame, sample_days: int = 14):
    """Plot water level decomposition showing observed, tide, and surge."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Find a period with interesting surge (e.g., a storm event)
    if "surge" in df.columns:
        surge = df["surge"].dropna()
    else:
        surge = df["observed"] - df["predicted"]
        surge = surge.dropna()

    # Find the date of maximum surge
    max_surge_date = surge.idxmax()

    # Extract sample period around maximum
    start = max_surge_date - pd.Timedelta(days=sample_days // 2)
    end = max_surge_date + pd.Timedelta(days=sample_days // 2)

    sample = df.loc[start:end].copy()

    if len(sample) < 24:  # Need at least 1 day of data
        print("Insufficient data for decomposition plot")
        return None, None

    # Top: Observed water level
    ax = axes[0]
    if "observed" in sample.columns:
        ax.plot(sample.index, sample["observed"], "b-", linewidth=1, label="Observed")
    ax.set_ylabel("Water Level (m, MSL)")
    ax.set_title(f"Water Level Components (around {max_surge_date.strftime('%Y-%m-%d')})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Middle: Predicted tide
    ax = axes[1]
    if "predicted" in sample.columns:
        ax.plot(sample.index, sample["predicted"], "g-", linewidth=1, label="Predicted tide")
    ax.set_ylabel("Tide (m, MSL)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Bottom: Storm surge
    ax = axes[2]
    if "surge" in sample.columns:
        surge_sample = sample["surge"]
    else:
        surge_sample = sample["observed"] - sample["predicted"]
    ax.fill_between(sample.index, 0, surge_sample, alpha=0.5, color="C1", label="Surge")
    ax.plot(sample.index, surge_sample, "C1-", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Surge (m)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "atlantic_city_surge_decomposition.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'atlantic_city_surge_decomposition.png'}")

    return fig, axes


def fit_stationary_gev(values: np.ndarray) -> dict:
    """Fit stationary GEV distribution."""
    print("\n" + "=" * 60)
    print("Stationary GEV Fit")
    print("=" * 60)

    params = xts.fit_gev(values)

    print(f"Location (μ): {params['loc']:.4f} m = {params['loc'] * 1000:.1f} mm")
    print(f"Scale (σ):    {params['scale']:.4f} m = {params['scale'] * 1000:.1f} mm")
    print(f"Shape (ξ):    {params['shape']:.3f}")

    # Interpret shape parameter
    if params["shape"] > 0.05:
        print("  → Fréchet type (heavy tail, unbounded)")
        print("    Storm surge has heavy upper tail - rare extreme events can be very large")
    elif params["shape"] < -0.05:
        print("  → Weibull type (bounded upper tail)")
    else:
        print("  → Approximately Gumbel type")

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

    fig.suptitle("GEV Diagnostic Plots - Atlantic City Storm Surge", fontsize=14, y=1.02)

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "atlantic_city_surge_diagnostics.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'atlantic_city_surge_diagnostics.png'}")

    return fig, axes


def test_for_trend(values: np.ndarray, years: np.ndarray) -> dict:
    """Test for significant trend using likelihood ratio test."""
    print("\n" + "=" * 60)
    print("Trend Detection in Storm Surge Extremes")
    print("=" * 60)

    print("\nThis test examines whether storm surge extremes are changing")
    print("independent of mean sea level rise (which affects 'observed' but not 'surge').\n")

    result = xts.likelihood_ratio_test(values, years, trend_in="loc")

    print(f"Test statistic (D):       {result['statistic']:.3f}")
    print(f"Degrees of freedom:       {result['df']}")
    print(f"p-value:                  {result['p_value']:.4f}")
    print(f"Significant at α=0.05:    {'Yes' if result['significant'] else 'No'}")
    print(f"AIC (stationary):         {result['aic_stationary']:.1f}")
    print(f"AIC (non-stationary):     {result['aic_nonstationary']:.1f}")
    print(f"Preferred model (AIC):    {result['preferred_model']}")

    if result["significant"]:
        print("\n→ Evidence that storm surge extremes are changing over time!")
        print("  This suggests storm intensity may be changing, not just sea level.")
    else:
        print("\n→ No significant trend detected in storm surge extremes.")
        print("  Storm intensity appears relatively stable over this period.")

    return result


def fit_nonstationary_gev(values: np.ndarray, years: np.ndarray) -> dict:
    """Fit non-stationary GEV with trend in location."""
    print("\n" + "=" * 60)
    print("Non-Stationary GEV Fit")
    print("=" * 60)

    params = xts.fit_nonstationary_gev(values, years, trend_in="loc")

    print(f"Location intercept (μ₀): {params['loc0']:.4f} m")
    print(f"Location trend (μ₁):     {params['loc1']:.6f} m/year")
    print(f"Scale (exp(σ₀)):         {np.exp(params['scale0']):.4f} m")
    print(f"Shape (ξ):               {params['shape']:.3f}")

    # Convert to mm/year and mm/decade
    trend_mm_year = params["loc1"] * 1000
    trend_mm_decade = trend_mm_year * 10

    print(f"\nTrend interpretation:")
    print(f"  Change per year:       {trend_mm_year:+.2f} mm/year")
    print(f"  Change per decade:     {trend_mm_decade:+.1f} mm/decade")

    total_change = params["loc1"] * (years[-1] - years[0])
    total_change_mm = total_change * 1000
    print(f"  Total change ({years[0]}-{years[-1]}): {total_change_mm:+.1f} mm")

    return params


def plot_nonstationary_fit(values: np.ndarray, years: np.ndarray, params: dict):
    """Plot non-stationary GEV fit."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Observations
    ax.scatter(years, values * 1000, alpha=0.7, s=40, label="Annual max surge", zorder=3)

    # Time-varying location parameter
    loc_t = params["loc0"] + params["loc1"] * years
    ax.plot(years, loc_t * 1000, "r-", linewidth=2, label="Location μ(t)", zorder=2)

    # Quantile bands
    scale_t = np.exp(params["scale0"] + params["scale1"] * years)
    from scipy import stats

    for q, ls, label in [(0.05, "--", "5th pctile"), (0.95, "--", "95th pctile")]:
        quantile = stats.genextreme.ppf(q, c=-params["shape"], loc=loc_t, scale=scale_t)
        ax.plot(years, quantile * 1000, ls, color="blue", alpha=0.5, label=label, zorder=1)

    ax.fill_between(
        years,
        stats.genextreme.ppf(0.05, c=-params["shape"], loc=loc_t, scale=scale_t) * 1000,
        stats.genextreme.ppf(0.95, c=-params["shape"], loc=loc_t, scale=scale_t) * 1000,
        alpha=0.2,
        color="blue",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Maximum Storm Surge (mm)")
    ax.set_title("Non-Stationary GEV Fit - Atlantic City Storm Surge")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "atlantic_city_surge_trend.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'atlantic_city_surge_trend.png'}")

    return fig, ax


def plot_return_levels(years: np.ndarray, params_stat: dict, params_ns: dict = None):
    """Plot return level curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    return_periods = np.array([2, 5, 10, 20, 50, 100, 200])

    # Stationary return levels
    rl_stat = xts.return_level(
        return_periods,
        loc=params_stat["loc"],
        scale=params_stat["scale"],
        shape=params_stat["shape"],
    )
    ax.semilogx(return_periods, rl_stat * 1000, "k-", linewidth=2, label="Stationary")

    # Non-stationary return levels at different years
    if params_ns is not None:
        ref_years = [years[0], years[-1]]
        colors = ["blue", "red"]
        labels = [f"{years[0]} (historical)", f"{years[-1]} (recent)"]

        for year, color, label in zip(ref_years, colors, labels):
            rl_ns = xts.nonstationary_return_level(return_periods, params_ns, year)
            ax.semilogx(return_periods, rl_ns * 1000, "--", color=color, linewidth=2, label=label)

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level (mm)")
    ax.set_title("Storm Surge Return Levels - Atlantic City, NJ")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "atlantic_city_surge_return_levels.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'atlantic_city_surge_return_levels.png'}")

    return fig, ax


def compute_effective_return_periods(years: np.ndarray, params: dict):
    """Calculate effective return periods."""
    print("\n" + "=" * 60)
    print("Effective Return Periods (How Storm Risk is Changing)")
    print("=" * 60)

    eff = xts.effective_return_level(
        params,
        reference_value=years[0],
        future_value=years[-1],
        return_periods=[10, 20, 50, 100],
    )

    print(f"\nComparing {years[0]} to {years[-1]}:")
    print(f"{'Historical Event':>20} {'Now Becomes':>20}")
    print("-" * 45)

    for i, rp in enumerate(eff["return_periods"]):
        eff_period = eff["effective_period"][i]
        ref_level_mm = eff["reference_levels"][i] * 1000
        print(f"  {rp:>3}-year ({ref_level_mm:.0f} mm) -> {eff_period:>6.1f}-year event")

    print(f"\nChange in return levels:")
    print(f"{'RP (yr)':>10} {years[0]:>10} {years[-1]:>10} {'Change':>10}")
    print("-" * 45)
    for i, rp in enumerate(eff["return_periods"]):
        ref = eff["reference_levels"][i] * 1000
        fut = eff["future_levels"][i] * 1000
        change = eff["change"][i] * 1000
        print(f"{rp:>10} {ref:>10.1f} {fut:>10.1f} {change:>+10.1f} mm")

    return eff


def plot_effective_return_periods(eff: dict):
    """Plot effective return period changes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Return level comparison
    ax1 = axes[0]
    x = np.arange(len(eff["return_periods"]))
    width = 0.35

    ref_mm = eff["reference_levels"] * 1000
    fut_mm = eff["future_levels"] * 1000

    bars1 = ax1.bar(x - width / 2, ref_mm, width, label=f"{int(eff['reference_value'])}", color="C0")
    bars2 = ax1.bar(x + width / 2, fut_mm, width, label=f"{int(eff['future_value'])}", color="C1")

    ax1.set_xlabel("Return Period (years)")
    ax1.set_ylabel("Return Level (mm)")
    ax1.set_title("Storm Surge Return Levels: Then vs Now")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(rp)) for rp in eff["return_periods"]])
    ax1.legend(title="Year")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Effective return period
    ax2 = axes[1]
    ax2.bar(x, eff["effective_period"], color="C2", alpha=0.7, edgecolor="black")

    # Add reference lines
    for i, rp in enumerate(eff["return_periods"]):
        ax2.hlines(rp, i - 0.4, i + 0.4, colors="red", linestyles="--", linewidth=2)

    ax2.set_xlabel("Original Return Period (years)")
    ax2.set_ylabel("Effective Return Period Now (years)")
    ax2.set_title("How Historical Storm Surges Are Changing")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(rp)) for rp in eff["return_periods"]])
    ax2.grid(True, alpha=0.3, axis="y")

    ax2.annotate(
        "Red lines = original period",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
    )

    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / "atlantic_city_surge_effective.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / 'atlantic_city_surge_effective.png'}")

    return fig, axes


def print_summary(values: np.ndarray, years: np.ndarray, params_stat: dict, lrt_result: dict):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("Summary: Atlantic City Storm Surge Analysis")
    print("=" * 60)

    print(f"\nRecord length: {len(years)} years ({years[0]}-{years[-1]})")
    print(f"This is one of the longest tide gauge records in the United States.")

    print(f"\nKey findings:")

    # Stationary parameters
    print(f"  GEV shape parameter: {params_stat['shape']:.3f}")
    if params_stat["shape"] > 0:
        print("    → Heavy-tailed distribution (positive shape)")
        print("    → Rare extreme storms can produce very large surges")

    # Return levels
    for rp in [10, 50, 100]:
        rl = xts.return_level(rp, **params_stat)
        print(f"  {rp:3d}-year return level: {rl * 1000:.0f} mm ({rl:.3f} m)")

    # Trend
    print(f"\n  Trend test p-value: {lrt_result['p_value']:.4f}")
    if lrt_result["significant"]:
        print("    → Significant trend detected in storm surge extremes")
        print("    → Storm intensity appears to be changing over time")
    else:
        print("    → No significant trend detected")
        print("    → Storm intensity appears relatively stable")

    print("\nScientific note:")
    print("  By analyzing surge (observed - predicted tide), we isolate")
    print("  meteorological forcing from astronomical tides and mean sea level.")
    print("  A trend in surge suggests changing storm climatology, separate")
    print("  from the well-documented rise in mean sea level.")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Atlantic City Storm Surge Extreme Value Analysis")
    print("=" * 60)

    # Load data
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Prepare surge data
    surge_df = prepare_surge_data(df)

    # Extract daily and annual maxima
    daily_max = extract_daily_max_surge(surge_df)
    values, years = extract_annual_max_surge(daily_max)

    if len(years) < MIN_YEARS:
        print(f"\nInsufficient data: need at least {MIN_YEARS} years, have {len(years)}")
        return

    # Plot the full record
    plot_surge_record(values, years, df)

    # Plot decomposition (observed = tide + surge)
    plot_decomposition(df)

    # Fit stationary GEV
    params_stat = fit_stationary_gev(values)

    # Diagnostic plots
    plot_diagnostics(values, params_stat)

    # Test for trend
    lrt_result = test_for_trend(values, years)

    # If trend significant, fit non-stationary
    params_ns = None
    if lrt_result["significant"] or lrt_result["preferred_model"] == "nonstationary":
        params_ns = fit_nonstationary_gev(values, years)
        plot_nonstationary_fit(values, years, params_ns)
        eff = compute_effective_return_periods(years, params_ns)
        plot_effective_return_periods(eff)

    # Return level plot
    plot_return_levels(years, params_stat, params_ns)

    # Summary
    print_summary(values, years, params_stat, lrt_result)

    plt.show()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
