"""
Precipitation Extremes at New Brunswick, NJ
============================================

This example demonstrates extreme value analysis of observed daily precipitation
from the NOAA Cooperative Observer station at New Brunswick, NJ. The analysis
includes GEV fitting, trend detection, and non-stationary modeling.

**Data source:** NOAA Climate Data Online (CDO), Station GHCND:USC00286055
"""

# %%
# Setup and Configuration
# -----------------------
# Import libraries and configure paths.

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

DATA_FILE = _THIS_DIR.parent / "tests" / "data" / "noaa_new_brunswick.csv"

# %%
# Loading the Data
# ----------------
# Load precipitation data from the NOAA station.

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}\n"
        "Run 'python scripts/fetch_noaa_data.py' to download the data."
    )

df = pd.read_csv(DATA_FILE, parse_dates=["date"], index_col="date")
print(f"Loaded {len(df)} daily records")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Variables: {', '.join(df.columns)}")

# %%
# Extract Annual Maxima
# ---------------------
# Extract the annual maximum precipitation values.

if "PRCP" not in df.columns:
    raise KeyError("Precipitation column 'PRCP' not found in data")

prcp = df["PRCP"]
annual_max = prcp.resample("YE").max().dropna()

values = annual_max.values
years = annual_max.index.year.values

print(f"\nAnnual maxima extracted:")
print(f"  Years: {years[0]} to {years[-1]} ({len(years)} years)")
print(f"  Min: {values.min():.2f} inches")
print(f"  Max: {values.max():.2f} inches")
print(f"  Mean: {values.mean():.2f} inches")

# %%
# Visualizing Annual Maxima
# -------------------------
# Plot the annual maximum precipitation time series.

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(years, values, alpha=0.7, s=40, label="Annual maximum", zorder=3)

# Add linear trend line
slope, intercept = np.polyfit(years, values, 1)
trend_line = intercept + slope * years
ax.plot(years, trend_line, "r-", linewidth=2, label=f"Trend: {slope:.4f} in/year", zorder=2)

ax.set_xlabel("Year")
ax.set_ylabel("Annual Maximum Precipitation (inches)")
ax.set_title("Annual Maximum Daily Precipitation - New Brunswick, NJ")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Stationary GEV Fit
# ------------------
# Fit a stationary GEV distribution to the data.

params = xts.fit_gev(values)

print(f"GEV Parameters:")
print(f"  Location (μ): {params['loc']:.2f} inches")
print(f"  Scale (σ):    {params['scale']:.2f} inches")
print(f"  Shape (ξ):    {params['shape']:.3f}")

if params["shape"] > 0.05:
    print("  → Fréchet type (heavy tail, unbounded)")
elif params["shape"] < -0.05:
    print("  → Weibull type (bounded upper tail)")
else:
    print("  → Approximately Gumbel type (exponential tail)")

# %%
# Diagnostic Plots
# ----------------
# Create 4-panel diagnostic plot.

fig, axes = xts.diagnostic_plots(
    values,
    loc=params["loc"],
    scale=params["scale"],
    shape=params["shape"],
    figsize=(10, 10),
)

fig.suptitle("GEV Diagnostic Plots - New Brunswick Precipitation", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Trend Detection
# ---------------
# Test for significant trend using likelihood ratio test.

lrt_result = xts.likelihood_ratio_test(values, years, trend_in="loc")

print(f"Likelihood Ratio Test:")
print(f"  Test statistic (D):       {lrt_result['statistic']:.3f}")
print(f"  Degrees of freedom:       {lrt_result['df']}")
print(f"  p-value:                  {lrt_result['p_value']:.4f}")
print(f"  Significant at α=0.05:    {'Yes' if lrt_result['significant'] else 'No'}")
print(f"  AIC (stationary):         {lrt_result['aic_stationary']:.1f}")
print(f"  AIC (non-stationary):     {lrt_result['aic_nonstationary']:.1f}")
print(f"  Preferred model (AIC):    {lrt_result['preferred_model']}")

# %%
# Trend Test Visualization
# ------------------------
# Visualize the likelihood ratio test results.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Time series with linear trend
ax1 = axes[0]
ax1.scatter(years, values, alpha=0.7, s=40, label="Annual maximum")
ax1.plot(years, trend_line, "r--", linewidth=2, label=f"Linear trend: {slope:.4f} in/year")
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

preferred_idx = 0 if lrt_result["preferred_model"] == "stationary" else 1
bars[preferred_idx].set_edgecolor("green")
bars[preferred_idx].set_linewidth(3)

ax2.set_ylabel("AIC (lower is better)")
ax2.set_title(f"Model Comparison (p = {lrt_result['p_value']:.4f})")

result_text = "Significant trend" if lrt_result["significant"] else "No significant trend"
ax2.annotate(
    result_text, xy=(0.5, 0.95), xycoords="axes fraction",
    ha="center", fontsize=12, fontweight="bold",
    color="green" if lrt_result["significant"] else "gray",
)

plt.tight_layout()
plt.show()

# %%
# Non-Stationary Analysis (if significant)
# -----------------------------------------
# If trend is significant, fit non-stationary GEV.

if lrt_result["significant"] or lrt_result["preferred_model"] == "nonstationary":
    params_ns = xts.fit_nonstationary_gev(values, years, trend_in="loc")

    print(f"Non-Stationary GEV Parameters:")
    print(f"  Location intercept (μ₀): {params_ns['loc0']:.2f} inches")
    print(f"  Location trend (μ₁):     {params_ns['loc1']:.4f} inches/year")
    print(f"  Scale (exp(σ₀)):         {np.exp(params_ns['scale0']):.2f} inches")
    print(f"  Shape (ξ):               {params_ns['shape']:.3f}")

    trend_per_decade = params_ns["loc1"] * 10
    total_change = params_ns["loc1"] * (years[-1] - years[0])
    print(f"\nTrend interpretation:")
    print(f"  Change per decade:     {trend_per_decade:+.3f} inches/decade")
    print(f"  Total change ({years[0]}-{years[-1]}): {total_change:+.2f} inches")

    # Plot non-stationary fit
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(years, values, alpha=0.7, s=40, label="Annual maximum", zorder=3)

    loc_t = params_ns["loc0"] + params_ns["loc1"] * years
    ax.plot(years, loc_t, "r-", linewidth=2, label="Location μ(t)", zorder=2)

    scale_t = np.exp(params_ns["scale0"] + params_ns["scale1"] * years)

    q05 = stats.genextreme.ppf(0.05, c=-params_ns["shape"], loc=loc_t, scale=scale_t)
    q95 = stats.genextreme.ppf(0.95, c=-params_ns["shape"], loc=loc_t, scale=scale_t)

    ax.plot(years, q05, "--", color="blue", alpha=0.5, label="5th percentile", zorder=1)
    ax.plot(years, q95, "--", color="blue", alpha=0.5, label="95th percentile", zorder=1)
    ax.fill_between(years, q05, q95, alpha=0.2, color="blue")

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Maximum Precipitation (inches)")
    ax.set_title("Non-Stationary GEV Fit - New Brunswick Precipitation")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("\nNo significant trend detected. Using stationary model.")
    params_ns = None

# %%
# Return Levels
# -------------
# Calculate and display return levels.

print("\nReturn Levels (Stationary Model):")
print(f"{'Return Period':>15} {'Return Level':>15}")
print("-" * 35)

for rp in [10, 25, 50, 100]:
    rl = xts.return_level(rp, loc=params["loc"], scale=params["scale"], shape=params["shape"])
    print(f"{rp:>15} {rl:>15.2f} inches")

print("\nAnalysis complete!")
