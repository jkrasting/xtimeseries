"""
Non-Stationary Trend Analysis
==============================

This example demonstrates non-stationary extreme value analysis:

- Generating data with known time trends
- Fitting non-stationary GEV with time-varying location
- Likelihood ratio test for trend significance
- Calculating time-varying return levels
- Understanding effective return periods
"""

import numpy as np
import xtimeseries as xts

# ============================================================================
# 1. Generate non-stationary data with known trend
# ============================================================================
print("=" * 60)
print("1. Generate Non-Stationary Data")
print("=" * 60)

# True parameters
TRUE_LOC_INTERCEPT = 25.0   # Location at year 0
TRUE_LOC_SLOPE = 0.06       # 0.6°C per decade trend
TRUE_SCALE = 4.0
TRUE_SHAPE = -0.1           # Bounded upper tail

result = xts.generate_nonstationary_series(
    n=80,                    # 80 years
    loc_intercept=TRUE_LOC_INTERCEPT,
    loc_slope=TRUE_LOC_SLOPE,
    scale=TRUE_SCALE,
    shape=TRUE_SHAPE,
    seed=42,
)

data = result["data"]
years = result["time"]

print(f"Generated {len(data)} years of annual maxima")
print(f"True trend: {TRUE_LOC_SLOPE} per year = {TRUE_LOC_SLOPE * 10:.2f} per decade")
print(f"\nLocation parameter:")
print(f"  Year 0:  {result['true_loc'][0]:.2f}")
print(f"  Year 79: {result['true_loc'][-1]:.2f}")
print(f"  Change:  {result['true_loc'][-1] - result['true_loc'][0]:.2f}")

# ============================================================================
# 2. Stationary vs Non-Stationary Model Comparison
# ============================================================================
print("\n" + "=" * 60)
print("2. Compare Stationary vs Non-Stationary Models")
print("=" * 60)

# Fit stationary model
params_stat = xts.fit_gev(data)
print("Stationary GEV fit:")
print(f"  Location: {params_stat['loc']:.3f}")
print(f"  Scale:    {params_stat['scale']:.3f}")
print(f"  Shape:    {params_stat['shape']:.3f}")

# Fit non-stationary model
params_ns = xts.fit_nonstationary_gev(data, years, trend_in="loc")
print("\nNon-stationary GEV fit:")
print(f"  Location intercept: {params_ns['loc0']:.3f} (true: {TRUE_LOC_INTERCEPT})")
print(f"  Location slope:     {params_ns['loc1']:.4f} (true: {TRUE_LOC_SLOPE})")
print(f"  Scale:              {np.exp(params_ns['scale0']):.3f} (true: {TRUE_SCALE})")
print(f"  Shape:              {params_ns['shape']:.3f} (true: {TRUE_SHAPE})")

# Model selection
print(f"\nModel selection:")
print(f"  AIC stationary:     {params_ns['aic']:.1f}")  # From NS fit
print(f"  AIC non-stationary: {params_ns['aic']:.1f}")
print(f"  BIC non-stationary: {params_ns['bic']:.1f}")

# ============================================================================
# 3. Likelihood Ratio Test
# ============================================================================
print("\n" + "=" * 60)
print("3. Likelihood Ratio Test for Trend")
print("=" * 60)

lrt = xts.likelihood_ratio_test(data, years, trend_in="loc")

print(f"Test statistic (D): {lrt['statistic']:.3f}")
print(f"Degrees of freedom: {lrt['df']}")
print(f"p-value:            {lrt['p_value']:.4f}")
print(f"Significant at 5%:  {'Yes' if lrt['significant'] else 'No'}")
print(f"Preferred model:    {lrt['preferred_model']}")

# ============================================================================
# 4. Time-Varying Return Levels
# ============================================================================
print("\n" + "=" * 60)
print("4. Time-Varying Return Levels")
print("=" * 60)

# Return levels at different time points
time_points = [0, 40, 79]
return_periods = [10, 50, 100]

print(f"{'Year':>6} ", end="")
for rp in return_periods:
    print(f"{rp}-yr RL  ", end="")
print()
print("-" * 35)

for t in time_points:
    print(f"{years[t]:>6} ", end="")
    for rp in return_periods:
        rl = xts.nonstationary_return_level(rp, params_ns, years[t])
        print(f"{rl:>8.2f} ", end="")
    print()

# ============================================================================
# 5. Effective Return Periods
# ============================================================================
print("\n" + "=" * 60)
print("5. Effective Return Periods (How Events Become More Frequent)")
print("=" * 60)

eff = xts.effective_return_level(
    params_ns,
    reference_value=years[0],    # Start of record
    future_value=years[-1],      # End of record
    return_periods=[10, 20, 50, 100],
)

print("How historical events become more frequent:")
print(f"{'Historical Event':>20} {'Now Becomes':>20}")
print("-" * 45)

for i, rp in enumerate(eff["return_periods"]):
    eff_period = eff["effective_period"][i]
    print(f"{rp:>3}-year event -> {eff_period:>8.1f}-year event")

print("\nChange in return levels:")
print(f"{'Return Period':>15} {'Reference':>10} {'Future':>10} {'Change':>10}")
print("-" * 50)
for i, rp in enumerate(eff["return_periods"]):
    ref_rl = eff["reference_levels"][i]
    fut_rl = eff["future_levels"][i]
    change = eff["change"][i]
    print(f"{rp:>15} {ref_rl:>10.2f} {fut_rl:>10.2f} {change:>+10.2f}")

# ============================================================================
# 6. Bootstrap Confidence Intervals for Trend
# ============================================================================
print("\n" + "=" * 60)
print("6. Uncertainty in Trend Estimate")
print("=" * 60)

# Bootstrap the non-stationary fit
# (simplified - full implementation would bootstrap the trend coefficient)
from xtimeseries.confidence import bootstrap_parameters

boot_params = bootstrap_parameters(data, n_bootstrap=500, random_state=42)

print("Stationary parameter uncertainty (for reference):")
print(f"  Location: {boot_params['loc']:.3f} ({boot_params['loc_ci'][0]:.3f}, {boot_params['loc_ci'][1]:.3f})")
print(f"  Scale:    {boot_params['scale']:.3f} ({boot_params['scale_ci'][0]:.3f}, {boot_params['scale_ci'][1]:.3f})")
print(f"  Shape:    {boot_params['shape']:.3f} ({boot_params['shape_ci'][0]:.3f}, {boot_params['shape_ci'][1]:.3f})")

print("\nNote: Full trend uncertainty requires custom bootstrapping of the")
print("      non-stationary model (refitting to each bootstrap sample).")

print("\nDone!")
