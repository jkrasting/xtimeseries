#!/usr/bin/env python
"""
Basic usage example for xtimeseries.

This script demonstrates core functionality:
- Generating synthetic data
- Fitting GEV distribution
- Calculating return levels
- Bootstrap confidence intervals
"""

import numpy as np
import xtimeseries as xts

# ============================================================================
# 1. Generate synthetic data with known parameters
# ============================================================================
print("=" * 60)
print("1. Generate Synthetic Data")
print("=" * 60)

# True parameters for validation
TRUE_LOC = 30.0
TRUE_SCALE = 5.0
TRUE_SHAPE = 0.1  # Frechet type (heavy tail)

# Generate 100 years of annual maxima
data = xts.generate_gev_series(
    n=100,
    loc=TRUE_LOC,
    scale=TRUE_SCALE,
    shape=TRUE_SHAPE,
    seed=42,
)

print(f"Generated {len(data)} annual maxima")
print(f"True parameters: loc={TRUE_LOC}, scale={TRUE_SCALE}, shape={TRUE_SHAPE}")
print(f"Sample mean: {np.mean(data):.2f}, std: {np.std(data):.2f}")

# ============================================================================
# 2. Fit GEV distribution
# ============================================================================
print("\n" + "=" * 60)
print("2. Fit GEV Distribution")
print("=" * 60)

params = xts.fit_gev(data)

print(f"Fitted parameters:")
print(f"  Location: {params['loc']:.3f} (true: {TRUE_LOC})")
print(f"  Scale:    {params['scale']:.3f} (true: {TRUE_SCALE})")
print(f"  Shape:    {params['shape']:.3f} (true: {TRUE_SHAPE})")

# ============================================================================
# 3. Calculate return levels
# ============================================================================
print("\n" + "=" * 60)
print("3. Calculate Return Levels")
print("=" * 60)

return_periods = [10, 20, 50, 100, 200, 500]
return_levels = xts.return_level(return_periods, **params)

# Calculate true return levels for comparison
true_rls = xts.generate_gev_return_levels(TRUE_LOC, TRUE_SCALE, TRUE_SHAPE)

print(f"{'Return Period':>15} {'Fitted RL':>12} {'True RL':>12} {'Error %':>10}")
print("-" * 55)
for rp, rl in zip(return_periods, return_levels):
    # Find corresponding true value
    idx = np.where(true_rls["return_periods"] == rp)[0]
    if len(idx) > 0:
        true_rl = true_rls["return_levels"][idx[0]]
        error = 100 * abs(rl - true_rl) / true_rl
        print(f"{rp:>15} {rl:>12.2f} {true_rl:>12.2f} {error:>10.2f}%")
    else:
        print(f"{rp:>15} {rl:>12.2f}")

# ============================================================================
# 4. Bootstrap confidence intervals
# ============================================================================
print("\n" + "=" * 60)
print("4. Bootstrap Confidence Intervals")
print("=" * 60)

ci_result = xts.bootstrap_ci(
    data,
    return_periods=[10, 50, 100],
    n_bootstrap=500,
    random_state=42,
)

print(f"{'Return Period':>15} {'Return Level':>12} {'95% CI Lower':>12} {'95% CI Upper':>12}")
print("-" * 55)
for i, rp in enumerate(ci_result["return_periods"]):
    rl = ci_result["return_levels"][i]
    lower = ci_result["lower"][i]
    upper = ci_result["upper"][i]
    print(f"{rp:>15} {rl:>12.2f} {lower:>12.2f} {upper:>12.2f}")

# ============================================================================
# 5. Check if true values are within confidence intervals
# ============================================================================
print("\n" + "=" * 60)
print("5. Validation: True Values vs Confidence Intervals")
print("=" * 60)

for i, rp in enumerate(ci_result["return_periods"]):
    idx = np.where(true_rls["return_periods"] == rp)[0]
    if len(idx) > 0:
        true_rl = true_rls["return_levels"][idx[0]]
        lower = ci_result["lower"][i]
        upper = ci_result["upper"][i]
        contained = lower <= true_rl <= upper
        status = "PASS" if contained else "MISS"
        print(f"{rp}-year: True={true_rl:.2f}, CI=[{lower:.2f}, {upper:.2f}] - {status}")

print("\nDone!")
