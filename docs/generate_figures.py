#!/usr/bin/env python
"""
Generate static figures for xtimeseries documentation.

This script generates all PNG figures needed by the Sphinx documentation.
Run from the docs/ directory:

    python generate_figures.py

Figures are saved to docs/_static/
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

# Try to import xtimeseries
try:
    import xtimeseries as xts
except ImportError:
    print("Warning: xtimeseries not installed, using scipy directly for some figures")
    xts = None

from scipy import stats

# Output directory
OUTPUT_DIR = Path(__file__).parent / "_static"
OUTPUT_DIR.mkdir(exist_ok=True)

# Figure style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})


def generate_gev_shapes():
    """Generate figure showing GEV PDF for different shape parameters."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-2, 8, 500)
    loc, scale = 0, 1

    shapes = [
        (0.3, "Fréchet (ξ = 0.3)", "tab:red"),
        (0.0, "Gumbel (ξ = 0)", "tab:green"),
        (-0.3, "Weibull (ξ = -0.3)", "tab:blue"),
    ]

    for shape, label, color in shapes:
        # scipy uses c = -xi
        c = -shape
        pdf = stats.genextreme.pdf(x, c, loc=loc, scale=scale)
        ax.plot(x, pdf, label=label, color=color, linewidth=2)

    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.set_title("GEV Distribution for Different Shape Parameters")
    ax.legend(loc="upper right")
    ax.set_xlim(-2, 8)
    ax.set_ylim(0, 0.45)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "gev_shapes.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: gev_shapes.png")


def generate_gpd_shapes():
    """Generate figure showing GPD PDF for different shape parameters."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(0.01, 6, 500)
    scale = 1

    shapes = [
        (0.3, "Pareto tail (ξ = 0.3)", "tab:red"),
        (0.0, "Exponential (ξ = 0)", "tab:green"),
        (-0.3, "Bounded (ξ = -0.3)", "tab:blue"),
    ]

    for shape, label, color in shapes:
        # scipy.stats.genpareto uses c = xi (same convention!)
        if shape < 0:
            # For negative shape, x has upper bound at -scale/shape
            x_bounded = x[x < -scale / shape]
            pdf = stats.genpareto.pdf(x_bounded, shape, loc=0, scale=scale)
            ax.plot(x_bounded, pdf, label=label, color=color, linewidth=2)
        else:
            pdf = stats.genpareto.pdf(x, shape, loc=0, scale=scale)
            ax.plot(x, pdf, label=label, color=color, linewidth=2)

    ax.set_xlabel("x (exceedance)")
    ax.set_ylabel("Probability Density")
    ax.set_title("GPD for Different Shape Parameters")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "gpd_shapes.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: gpd_shapes.png")


def generate_return_level_curve():
    """Generate a return level vs return period plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Parameters
    loc, scale, shape = 30, 5, 0.1

    # Return periods
    T = np.array([1.5, 2, 5, 10, 20, 50, 100, 200, 500])

    # Calculate return levels
    # y_p = -ln(1 - 1/T)
    y_p = -np.log(1 - 1 / T)

    if shape != 0:
        rl = loc + (scale / shape) * (y_p ** (-shape) - 1)
    else:
        rl = loc - scale * np.log(-np.log(1 - 1 / T))

    # Approximate confidence band (for illustration)
    se = scale * 0.15 * np.log(T)  # Rough approximation
    lower = rl - 1.96 * se
    upper = rl + 1.96 * se

    ax.semilogx(T, rl, "b-", linewidth=2, label="Return level")
    ax.fill_between(T, lower, upper, alpha=0.2, color="blue", label="95% CI")

    # Add some "observed" points
    rng = np.random.default_rng(42)
    n = 50
    obs = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=n, random_state=rng)
    obs.sort()
    plotting_pos = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    T_obs = 1 / (1 - plotting_pos)
    ax.scatter(T_obs, obs, s=30, c="black", alpha=0.6, zorder=5, label="Observations")

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level")
    ax.set_title("Return Level Plot")
    ax.legend(loc="lower right")
    ax.set_xlim(1, 600)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "return_level_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: return_level_curve.png")


def generate_probability_plot_example():
    """Generate example probability plot."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Generate data and fit
    rng = np.random.default_rng(42)
    loc, scale, shape = 30, 5, 0.1
    data = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=50, random_state=rng)

    # Fit
    c, fit_loc, fit_scale = stats.genextreme.fit(data)
    fit_shape = -c

    # Empirical probabilities (Gringorten plotting position)
    data_sorted = np.sort(data)
    n = len(data_sorted)
    empirical_prob = (np.arange(1, n + 1) - 0.44) / (n + 0.12)

    # Theoretical probabilities
    theoretical_prob = stats.genextreme.cdf(data_sorted, -fit_shape, loc=fit_loc, scale=fit_scale)

    ax.scatter(theoretical_prob, empirical_prob, s=40, alpha=0.7, edgecolors="black", linewidths=0.5)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="1:1 line")

    ax.set_xlabel("Theoretical Probability")
    ax.set_ylabel("Empirical Probability")
    ax.set_title("GEV Probability Plot")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "probability_plot_example.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: probability_plot_example.png")


def generate_qq_plot_example():
    """Generate example Q-Q plot."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Generate data and fit
    rng = np.random.default_rng(42)
    loc, scale, shape = 30, 5, 0.1
    data = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=50, random_state=rng)

    # Fit
    c, fit_loc, fit_scale = stats.genextreme.fit(data)
    fit_shape = -c

    # Empirical and theoretical quantiles
    data_sorted = np.sort(data)
    n = len(data_sorted)
    probs = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    theoretical_quantiles = stats.genextreme.ppf(probs, -fit_shape, loc=fit_loc, scale=fit_scale)

    ax.scatter(theoretical_quantiles, data_sorted, s=40, alpha=0.7, edgecolors="black", linewidths=0.5)
    min_val = min(theoretical_quantiles.min(), data_sorted.min())
    max_val = max(theoretical_quantiles.max(), data_sorted.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="1:1 line")

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title("GEV Q-Q Plot")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "qq_plot_example.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: qq_plot_example.png")


def generate_return_level_plot_example():
    """Generate example return level plot with confidence bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Generate data and fit
    rng = np.random.default_rng(42)
    loc, scale, shape = 30, 5, 0.1
    data = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=50, random_state=rng)

    # Fit
    c, fit_loc, fit_scale = stats.genextreme.fit(data)
    fit_shape = -c

    # Return periods for curve
    T = np.logspace(np.log10(1.1), np.log10(500), 100)
    y_p = -np.log(1 - 1 / T)

    if fit_shape != 0:
        rl = fit_loc + (fit_scale / fit_shape) * (y_p ** (-fit_shape) - 1)
    else:
        rl = fit_loc - fit_scale * np.log(y_p)

    # Bootstrap CI (simplified)
    se = fit_scale * 0.12 * np.log(T)
    lower = rl - 1.96 * se
    upper = rl + 1.96 * se

    # Plot
    ax.semilogx(T, rl, "b-", linewidth=2, label="Fitted GEV")
    ax.fill_between(T, lower, upper, alpha=0.2, color="blue", label="95% CI")

    # Observations
    data_sorted = np.sort(data)
    n = len(data_sorted)
    plotting_pos = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    T_obs = 1 / (1 - plotting_pos)
    ax.scatter(T_obs, data_sorted, s=40, c="black", alpha=0.7, zorder=5, label="Observations")

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level")
    ax.set_title("Return Level Plot with 95% Confidence Bands")
    ax.legend(loc="lower right")
    ax.set_xlim(1, 600)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "return_level_plot_example.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: return_level_plot_example.png")


def generate_temperature_annual_max():
    """Generate temperature annual maxima time series figure."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Generate synthetic data with trend
    rng = np.random.default_rng(42)
    n_years = 50
    years = np.arange(1970, 1970 + n_years)

    # True parameters with trend
    loc0, loc1, scale, shape = 35, 0.02, 3, -0.1
    loc_t = loc0 + loc1 * (years - years.mean())

    # Generate data
    data = []
    for loc in loc_t:
        val = stats.genextreme.rvs(-shape, loc=loc, scale=scale, random_state=rng)
        data.append(val)
    data = np.array(data)

    # Plot
    ax.plot(years, data, "o-", markersize=5, linewidth=1, alpha=0.8)
    ax.plot(years, loc_t, "r--", linewidth=2, label="Trend")

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Maximum Temperature (°C)")
    ax.set_title("Annual Maximum Temperature Time Series")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "temperature_annual_max.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: temperature_annual_max.png")


def generate_temperature_fit():
    """Generate 4-panel GEV diagnostic plot for temperature."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Generate data
    rng = np.random.default_rng(42)
    loc, scale, shape = 35, 3, -0.1
    data = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=50, random_state=rng)

    # Fit
    c, fit_loc, fit_scale = stats.genextreme.fit(data)
    fit_shape = -c

    # Panel 1: Probability plot
    ax = axes[0, 0]
    data_sorted = np.sort(data)
    n = len(data_sorted)
    emp_prob = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    theo_prob = stats.genextreme.cdf(data_sorted, -fit_shape, loc=fit_loc, scale=fit_scale)
    ax.scatter(theo_prob, emp_prob, s=30, alpha=0.7)
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("Theoretical")
    ax.set_ylabel("Empirical")
    ax.set_title("(a) Probability Plot")
    ax.set_aspect("equal")

    # Panel 2: Q-Q plot
    ax = axes[0, 1]
    theo_quantiles = stats.genextreme.ppf(emp_prob, -fit_shape, loc=fit_loc, scale=fit_scale)
    ax.scatter(theo_quantiles, data_sorted, s=30, alpha=0.7)
    lim = [min(theo_quantiles.min(), data_sorted.min()), max(theo_quantiles.max(), data_sorted.max())]
    ax.plot(lim, lim, "r--")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title("(b) Q-Q Plot")

    # Panel 3: Return level plot
    ax = axes[1, 0]
    T = np.logspace(np.log10(1.1), np.log10(200), 100)
    y_p = -np.log(1 - 1 / T)
    rl = fit_loc + (fit_scale / fit_shape) * (y_p ** (-fit_shape) - 1) if fit_shape != 0 else fit_loc - fit_scale * np.log(y_p)
    ax.semilogx(T, rl, "b-", linewidth=2)
    T_obs = 1 / (1 - emp_prob)
    ax.scatter(T_obs, data_sorted, s=30, c="black", alpha=0.7)
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level (°C)")
    ax.set_title("(c) Return Level Plot")
    ax.grid(True, alpha=0.3)

    # Panel 4: Density plot
    ax = axes[1, 1]
    ax.hist(data, bins=15, density=True, alpha=0.6, edgecolor="black")
    x = np.linspace(data.min() - 2, data.max() + 2, 200)
    pdf = stats.genextreme.pdf(x, -fit_shape, loc=fit_loc, scale=fit_scale)
    ax.plot(x, pdf, "r-", linewidth=2, label="Fitted GEV")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.set_title("(d) Density Plot")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "temperature_fit.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: temperature_fit.png")


def generate_precipitation_idf():
    """Generate IDF curve example."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Synthetic IDF data (typical values)
    durations = np.array([1, 2, 6, 12, 24, 48])  # hours
    return_periods = [2, 5, 10, 25, 50, 100]

    # IDF formula: I = a / (d + b)^c
    # Different a for different return periods
    a_values = {2: 30, 5: 40, 10: 50, 25: 60, 50: 70, 100: 80}
    b, c = 5, 0.7

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(return_periods)))

    for i, T in enumerate(return_periods):
        a = a_values[T]
        intensity = a / (durations + b) ** c
        ax.loglog(durations, intensity, "o-", color=colors[i], label=f"{T}-year", linewidth=2, markersize=6)

    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Intensity (mm/hour)")
    ax.set_title("Intensity-Duration-Frequency Curves")
    ax.legend(title="Return Period", loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.8, 60)
    ax.set_ylim(0.5, 30)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "precipitation_idf.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: precipitation_idf.png")


def generate_trend_detection():
    """Generate trend detection figure."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Generate data with trend
    rng = np.random.default_rng(42)
    n_years = 70
    years = np.arange(1950, 1950 + n_years)

    loc0, loc1, scale, shape = 30, 0.03, 3, -0.1
    loc_t = loc0 + loc1 * (years - years.mean())

    data = []
    for loc in loc_t:
        val = stats.genextreme.rvs(-shape, loc=loc, scale=scale, random_state=rng)
        data.append(val)
    data = np.array(data)

    # Plot data
    ax.scatter(years, data, s=40, alpha=0.7, edgecolors="black", linewidths=0.5, label="Annual maxima")

    # Fitted trend (using simple linear regression for visualization)
    slope, intercept = np.polyfit(years, data, 1)
    trend_line = intercept + slope * years
    ax.plot(years, trend_line, "r-", linewidth=2, label=f"Trend: {slope:.3f}°C/year")

    # True trend
    ax.plot(years, loc_t, "g--", linewidth=2, alpha=0.7, label="True μ(t)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Maximum (°C)")
    ax.set_title("Detecting Trends in Annual Maximum Temperature")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "trend_detection.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: trend_detection.png")


def generate_gridded_trend_map():
    """Generate gridded trend map figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create synthetic gridded trend data
    rng = np.random.default_rng(42)
    nlat, nlon = 8, 12

    # Create a spatial pattern of trends (some positive, some negative)
    lat = np.linspace(-40, 40, nlat)
    lon = np.linspace(-60, 60, nlon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Base trend pattern (latitude-dependent with some noise)
    trends = 0.02 * (lat_grid / 40) + 0.01 * np.sin(lon_grid * np.pi / 60) + rng.normal(0, 0.005, (nlat, nlon))

    # Plot
    im = ax.pcolormesh(lon, lat, trends, cmap="RdBu_r", vmin=-0.04, vmax=0.04, shading="auto")
    cbar = plt.colorbar(im, ax=ax, label="Location Trend (units/year)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Pattern of Trend in Annual Maximum")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "gridded_trend_map.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: gridded_trend_map.png")


def generate_temperature_return_level():
    """Generate temperature return level plot for examples."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Generate temperature-like data
    rng = np.random.default_rng(42)
    loc, scale, shape = 35, 3, -0.1
    data = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=50, random_state=rng)

    # Fit
    c, fit_loc, fit_scale = stats.genextreme.fit(data)
    fit_shape = -c

    # Return level curve
    T = np.logspace(np.log10(1.1), np.log10(200), 100)
    y_p = -np.log(1 - 1 / T)
    rl = fit_loc + (fit_scale / fit_shape) * (y_p ** (-fit_shape) - 1)

    # Confidence bands
    se = fit_scale * 0.12 * np.log(T)
    lower = rl - 1.96 * se
    upper = rl + 1.96 * se

    ax.semilogx(T, rl, "b-", linewidth=2, label="Fitted GEV")
    ax.fill_between(T, lower, upper, alpha=0.2, color="blue", label="95% CI")

    # Observations
    data_sorted = np.sort(data)
    n = len(data_sorted)
    emp_prob = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    T_obs = 1 / (1 - emp_prob)
    ax.scatter(T_obs, data_sorted, s=40, c="black", alpha=0.7, zorder=5, label="Observations")

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level (°C)")
    ax.set_title("Temperature Return Levels")
    ax.legend(loc="lower right")
    ax.set_xlim(1, 250)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "temperature_return_level.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: temperature_return_level.png")


def generate_nonstationary_visualization():
    """Generate 2-panel non-stationary GEV visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate data with trend
    rng = np.random.default_rng(42)
    n_years = 70
    years = np.arange(1950, 1950 + n_years)

    loc0, loc1, scale, shape = 30, 0.03, 3, -0.1
    loc_t = loc0 + loc1 * (years - years.mean())

    data = []
    for loc in loc_t:
        val = stats.genextreme.rvs(-shape, loc=loc, scale=scale, random_state=rng)
        data.append(val)
    data = np.array(data)

    # Left panel: Data with fitted trend
    ax1 = axes[0]
    ax1.scatter(years, data, alpha=0.7, s=40, label="Observations")

    # Fitted location trend
    fitted_loc = loc0 + loc1 * (years - years.mean())
    ax1.plot(years, fitted_loc, "r-", linewidth=2, label="Fitted μ(t)")

    # Quantile bands
    for q, ls in [(0.05, "--"), (0.95, "--")]:
        quantile = stats.genextreme.ppf(q, -shape, loc=fitted_loc, scale=scale)
        label = "5th/95th percentile" if q == 0.05 else None
        ax1.plot(years, quantile, ls, color="blue", alpha=0.5, label=label)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual Maximum (°C)")
    ax1.set_title("Non-Stationary GEV Fit")
    ax1.legend(loc="upper left")

    # Right panel: Return level curves for different years
    ax2 = axes[1]
    T_range = np.array([2, 5, 10, 20, 50, 100, 200])
    y_p = -np.log(1 - 1 / T_range)

    for year, color in [(1950, "blue"), (1985, "green"), (2019, "red")]:
        loc_year = loc0 + loc1 * (year - years.mean())
        rl = loc_year + (scale / shape) * (y_p ** (-shape) - 1)
        ax2.semilogx(T_range, rl, "-o", color=color, label=f"{year}", markersize=6)

    ax2.set_xlabel("Return Period (years)")
    ax2.set_ylabel("Return Level (°C)")
    ax2.set_title("Return Level Curves by Year")
    ax2.legend(title="Reference Year")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "nonstationary_visualization.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: nonstationary_visualization.png")


def generate_theory_return_level():
    """Generate return level plot for theory section."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Generate data and fit
    rng = np.random.default_rng(123)
    loc, scale, shape = 30, 5, 0.1
    data = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=50, random_state=rng)

    # Fit
    c, fit_loc, fit_scale = stats.genextreme.fit(data)
    fit_shape = -c

    # Return level curve
    T = np.logspace(np.log10(1.1), np.log10(500), 100)
    y_p = -np.log(1 - 1 / T)

    if fit_shape != 0:
        rl = fit_loc + (fit_scale / fit_shape) * (y_p ** (-fit_shape) - 1)
    else:
        rl = fit_loc - fit_scale * np.log(y_p)

    # Confidence bands
    se = fit_scale * 0.15 * np.log(T)
    lower = rl - 1.96 * se
    upper = rl + 1.96 * se

    ax.semilogx(T, rl, "b-", linewidth=2, label="Fitted GEV")
    ax.fill_between(T, lower, upper, alpha=0.2, color="blue", label="95% CI")

    # Observations
    data_sorted = np.sort(data)
    n = len(data_sorted)
    emp_prob = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    T_obs = 1 / (1 - emp_prob)
    ax.scatter(T_obs, data_sorted, s=40, c="black", alpha=0.7, zorder=5, label="Observations")

    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level")
    ax.set_title("Return Level Plot with 95% Confidence Bands")
    ax.legend(loc="lower right")
    ax.set_xlim(1, 600)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "theory_return_level.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: theory_return_level.png")


def main():
    """Generate all figures."""
    print(f"Generating figures in {OUTPUT_DIR}")
    print("=" * 50)

    generate_gev_shapes()
    generate_gpd_shapes()
    generate_return_level_curve()
    generate_probability_plot_example()
    generate_qq_plot_example()
    generate_return_level_plot_example()
    generate_temperature_annual_max()
    generate_temperature_fit()
    generate_precipitation_idf()
    generate_trend_detection()
    generate_gridded_trend_map()
    generate_temperature_return_level()
    generate_nonstationary_visualization()
    generate_theory_return_level()

    print("=" * 50)
    print(f"Generated {len(list(OUTPUT_DIR.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
