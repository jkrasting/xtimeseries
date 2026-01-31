"""
Temperature Extremes Analysis
=============================

This example demonstrates analyzing temperature extremes:

- Generating synthetic daily temperature data with trends
- Extracting block maxima (annual maximum temperatures)
- Fitting GEV distribution to annual maxima
- Calculating return levels for heat extremes
- Non-stationary analysis to detect warming trends
"""

import numpy as np
import xarray as xr
import xtimeseries as xts

# ============================================================================
# 1. Generate synthetic daily temperature data
# ============================================================================
print("=" * 60)
print("1. Generate Synthetic Daily Temperature")
print("=" * 60)

# Generate 50 years of daily temperature data
temp = xts.generate_temperature_like(
    n_years=50,
    mean_annual=15.0,        # 15°C annual mean
    seasonal_amplitude=15.0,  # 15°C seasonal swing
    daily_std=3.0,           # 3°C daily variability
    trend_per_decade=0.3,    # 0.3°C warming per decade
    ar1_coef=0.8,            # Daily autocorrelation
    seed=42,
)

print(f"Generated {len(temp)} days of temperature data")
print(f"Date range: {temp.time.values[0]} to {temp.time.values[-1]}")
print(f"Temperature range: {temp.min().values:.1f}°C to {temp.max().values:.1f}°C")

# ============================================================================
# 2. Extract annual maxima
# ============================================================================
print("\n" + "=" * 60)
print("2. Extract Annual Maximum Temperatures")
print("=" * 60)

annual_max = xts.block_maxima(temp, freq="YE")

print(f"Number of years: {len(annual_max)}")
print(f"Mean annual maximum: {annual_max.mean().values:.2f}°C")
print(f"Std of annual maxima: {annual_max.std().values:.2f}°C")

# ============================================================================
# 3. Fit GEV distribution
# ============================================================================
print("\n" + "=" * 60)
print("3. Fit GEV to Annual Maxima")
print("=" * 60)

params = xts.fit_gev(annual_max.values)

print(f"GEV Parameters:")
print(f"  Location (mu):    {params['loc']:.3f}°C")
print(f"  Scale (sigma):    {params['scale']:.3f}°C")
print(f"  Shape (xi):       {params['shape']:.3f}")

# Interpret shape parameter
if params["shape"] > 0.1:
    print("  -> Heavy tail (Frechet type): extreme heat waves possible")
elif params["shape"] < -0.1:
    print("  -> Bounded tail (Weibull type): physical upper limit")
else:
    print("  -> Light tail (Gumbel type): exponential-like extremes")

# ============================================================================
# 4. Calculate return levels
# ============================================================================
print("\n" + "=" * 60)
print("4. Heat Extreme Return Levels")
print("=" * 60)

return_periods = [2, 5, 10, 20, 50, 100]
return_levels = xts.return_level(return_periods, **params)

print(f"{'Event':>20} {'Return Level':>12}")
print("-" * 35)
for rp, rl in zip(return_periods, return_levels):
    print(f"{rp:>3}-year max temp: {rl:>12.2f}°C")

# ============================================================================
# 5. Analyze trend (non-stationary analysis)
# ============================================================================
print("\n" + "=" * 60)
print("5. Non-Stationary Analysis (Trend Detection)")
print("=" * 60)

years = annual_max.time.dt.year.values

# Test for significant trend
lrt_result = xts.likelihood_ratio_test(annual_max.values, years)

print(f"Likelihood Ratio Test:")
print(f"  Test statistic: {lrt_result['statistic']:.3f}")
print(f"  p-value: {lrt_result['p_value']:.4f}")
print(f"  Significant trend: {'Yes' if lrt_result['significant'] else 'No'}")

# Fit non-stationary model
if lrt_result["p_value"] < 0.1:  # Even marginal significance
    ns_params = xts.fit_nonstationary_gev(annual_max.values, years)

    print(f"\nNon-stationary GEV Parameters:")
    print(f"  Location intercept: {ns_params['loc0']:.3f}°C")
    print(f"  Location trend:     {ns_params['loc1']:.4f}°C/year")
    print(f"  Scale:              {np.exp(ns_params['scale0']):.3f}°C")

    # Trend per decade
    trend_per_decade = ns_params["loc1"] * 10
    print(f"\n  Warming trend: {trend_per_decade:.2f}°C per decade")

    # Compare return levels: past vs future
    early_year = years[0]
    late_year = years[-1]

    rl_early = xts.nonstationary_return_level(100, ns_params, early_year)
    rl_late = xts.nonstationary_return_level(100, ns_params, late_year)

    print(f"\n100-year return level:")
    print(f"  Year {early_year}: {rl_early:.2f}°C")
    print(f"  Year {late_year}: {rl_late:.2f}°C")
    print(f"  Change: {rl_late - rl_early:.2f}°C")

# ============================================================================
# 6. Cold extremes (annual minima)
# ============================================================================
print("\n" + "=" * 60)
print("6. Cold Extreme Analysis (Annual Minima)")
print("=" * 60)

annual_min = xts.block_minima(temp, freq="YE")

# For minima, we negate and treat as maxima
negated_min = -annual_min.values
params_min = xts.fit_gev(negated_min)

# Convert back to original scale
print(f"Cold Extreme Return Levels:")
for rp in [10, 50, 100]:
    rl_negated = xts.return_level(rp, **params_min)
    rl_actual = -rl_negated
    print(f"  {rp}-year min temp: {rl_actual:.2f}°C")

print("\nDone!")
