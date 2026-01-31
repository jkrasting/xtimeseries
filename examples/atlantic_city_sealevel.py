"""
Storm Surge Analysis at Atlantic City
======================================

This example demonstrates extreme value analysis of storm surge extremes
from the NOAA tide gauge at Atlantic City, NJ. The 110+ year record provides
an exceptional dataset for understanding changes in storm intensity.

By analyzing surge (observed water level - astronomical tide prediction),
we isolate the meteorological forcing component. This allows us to detect
whether storm intensity is changing over time, independent of the background
mean sea level rise signal.

**Data source:** NOAA Tides & Currents CO-OPS API, Station 8534720 (Atlantic City, NJ)
"""

# %%
# Setup and Configuration
# -----------------------
# First, we import the necessary libraries and configure paths.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import xtimeseries as xts

# Handle path for both standalone execution and sphinx-gallery
try:
    _THIS_DIR = Path(__file__).parent
except NameError:
    import xtimeseries as _xts
    _THIS_DIR = Path(_xts.__file__).parent.parent.parent / "examples"

DATA_FILE = _THIS_DIR.parent / "tests" / "data" / "noaa_atlantic_city_daily.csv"

# %%
# Loading the Data
# ----------------
# Load the tide gauge data from the pre-downloaded NOAA dataset.

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}\n"
        "Run 'python scripts/fetch_noaa_tides.py' to download the data."
    )

df = pd.read_csv(DATA_FILE, parse_dates=["time"], index_col="time")

print(f"Loaded {len(df)} daily records")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Columns: {', '.join(df.columns)}")

# %%
# Preparing Storm Surge Data
# --------------------------
# Calculate storm surge as the difference between observed and predicted tide.

if "surge" in df.columns:
    surge = df["surge"]
elif "observed" in df.columns and "predicted" in df.columns:
    surge = df["observed"] - df["predicted"]
else:
    raise KeyError("Need 'surge' or both 'observed' and 'predicted' columns")

surge = surge.dropna()

print(f"Daily max surge records: {len(surge)}")
print(f"Surge statistics (m):")
print(f"  Min:  {surge.min():.3f}")
print(f"  Max:  {surge.max():.3f}")
print(f"  Mean: {surge.mean():.3f}")
print(f"  Std:  {surge.std():.3f}")

# %%
# Extract Annual Maximum Surge
# ----------------------------
# Extract the annual maximum surge values.

daily_max = surge
annual_max = daily_max.resample("YE").max()

# Remove years with insufficient data (< 300 days)
day_counts = daily_max.resample("YE").count()
valid_years = day_counts >= 300
annual_max = annual_max[valid_years].dropna()

values = annual_max.values
years = annual_max.index.year.values

print(f"\nAnnual maximum surge:")
print(f"  Years: {years[0]} to {years[-1]} ({len(years)} years)")
print(f"  Min:  {values.min():.3f} m")
print(f"  Max:  {values.max():.3f} m")
print(f"  Mean: {values.mean():.3f} m")

# %%
# Visualizing the Surge Record
# -----------------------------
# Plot the annual maximum surge time series and its distribution.

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
plt.show()

# %%
# Stationary GEV Fitting
# ----------------------
# Fit a stationary GEV distribution to the annual maximum surge.

params = xts.fit_gev(values)

print(f"GEV Parameters:")
print(f"  Location (μ): {params['loc']:.4f} m = {params['loc'] * 1000:.1f} mm")
print(f"  Scale (σ):    {params['scale']:.4f} m = {params['scale'] * 1000:.1f} mm")
print(f"  Shape (ξ):    {params['shape']:.3f}")

if params["shape"] > 0.05:
    print("  → Fréchet type (heavy tail, unbounded)")
    print("    Storm surge has heavy upper tail - rare extreme events can be very large")
elif params["shape"] < -0.05:
    print("  → Weibull type (bounded upper tail)")
else:
    print("  → Approximately Gumbel type")

# %%
# GEV Diagnostic Plots
# --------------------
# Create 4-panel diagnostic plot to assess fit quality.

fig, axes = xts.diagnostic_plots(
    values,
    loc=params["loc"],
    scale=params["scale"],
    shape=params["shape"],
    figsize=(10, 10),
)

fig.suptitle("GEV Diagnostic Plots - Atlantic City Storm Surge", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Trend Detection
# ---------------
# Test for significant trend using likelihood ratio test.

print("Testing for trend in storm surge extremes...")
print("This examines whether storm surge extremes are changing")
print("independent of mean sea level rise.\n")

lrt_result = xts.likelihood_ratio_test(values, years, trend_in="loc")

print(f"Test statistic (D):       {lrt_result['statistic']:.3f}")
print(f"Degrees of freedom:       {lrt_result['df']}")
print(f"p-value:                  {lrt_result['p_value']:.4f}")
print(f"Significant at α=0.05:    {'Yes' if lrt_result['significant'] else 'No'}")
print(f"AIC (stationary):         {lrt_result['aic_stationary']:.1f}")
print(f"AIC (non-stationary):     {lrt_result['aic_nonstationary']:.1f}")
print(f"Preferred model (AIC):    {lrt_result['preferred_model']}")

# %%
# Non-Stationary Analysis
# -----------------------
# If trend is significant, fit non-stationary GEV and show time-varying results.

if lrt_result["significant"] or lrt_result["preferred_model"] == "nonstationary":
    print("\n→ Evidence that storm surge extremes are changing over time!")

    # Fit non-stationary GEV
    params_ns = xts.fit_nonstationary_gev(values, years, trend_in="loc")

    print(f"\nNon-Stationary GEV Parameters:")
    print(f"  Location intercept (μ₀): {params_ns['loc0']:.4f} m")
    print(f"  Location trend (μ₁):     {params_ns['loc1']:.6f} m/year")
    print(f"  Scale (exp(σ₀)):         {np.exp(params_ns['scale0']):.4f} m")
    print(f"  Shape (ξ):               {params_ns['shape']:.3f}")

    trend_mm_year = params_ns["loc1"] * 1000
    trend_mm_decade = trend_mm_year * 10
    print(f"\nTrend interpretation:")
    print(f"  Change per year:       {trend_mm_year:+.2f} mm/year")
    print(f"  Change per decade:     {trend_mm_decade:+.1f} mm/decade")

    # Plot non-stationary fit
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(years, values * 1000, alpha=0.7, s=40, label="Annual max surge", zorder=3)

    loc_t = params_ns["loc0"] + params_ns["loc1"] * years
    ax.plot(years, loc_t * 1000, "r-", linewidth=2, label="Location μ(t)", zorder=2)

    scale_t = np.exp(params_ns["scale0"] + params_ns["scale1"] * years)

    for q, ls, label in [(0.05, "--", "5th pctile"), (0.95, "--", "95th pctile")]:
        quantile = stats.genextreme.ppf(q, c=-params_ns["shape"], loc=loc_t, scale=scale_t)
        ax.plot(years, quantile * 1000, ls, color="blue", alpha=0.5, label=label, zorder=1)

    ax.fill_between(
        years,
        stats.genextreme.ppf(0.05, c=-params_ns["shape"], loc=loc_t, scale=scale_t) * 1000,
        stats.genextreme.ppf(0.95, c=-params_ns["shape"], loc=loc_t, scale=scale_t) * 1000,
        alpha=0.2, color="blue",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Maximum Storm Surge (mm)")
    ax.set_title("Non-Stationary GEV Fit - Atlantic City Storm Surge")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("\n→ No significant trend detected in storm surge extremes.")
    print("  Storm intensity appears relatively stable over this period.")
    params_ns = None

# %%
# Return Level Curves
# -------------------
# Compare return levels between stationary and non-stationary models.

fig, ax = plt.subplots(figsize=(10, 6))

return_periods = np.array([2, 5, 10, 20, 50, 100, 200])

# Stationary return levels
rl_stat = xts.return_level(
    return_periods,
    loc=params["loc"],
    scale=params["scale"],
    shape=params["shape"],
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
plt.show()

# %%
# Summary
# -------
# Print a summary of the analysis findings.

print("\n" + "=" * 60)
print("Summary: Atlantic City Storm Surge Analysis")
print("=" * 60)

print(f"\nRecord length: {len(years)} years ({years[0]}-{years[-1]})")
print(f"This is one of the longest tide gauge records in the United States.")

print(f"\nKey findings:")
print(f"  GEV shape parameter: {params['shape']:.3f}")
if params["shape"] > 0:
    print("    → Heavy-tailed distribution (positive shape)")
    print("    → Rare extreme storms can produce very large surges")

for rp in [10, 50, 100]:
    rl = xts.return_level(rp, **params)
    print(f"  {rp:3d}-year return level: {rl * 1000:.0f} mm ({rl:.3f} m)")

print(f"\n  Trend test p-value: {lrt_result['p_value']:.4f}")
if lrt_result["significant"]:
    print("    → Significant trend detected in storm surge extremes")
else:
    print("    → No significant trend detected")

print("\nAnalysis complete!")
