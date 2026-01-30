"""
xtimeseries: Extreme value analysis toolkit for climate data.

This package provides tools for analyzing extreme events in climate model
output and observational data, including:

- Distribution fitting (GEV, GPD) with correct sign conventions
- Block maxima extraction with cftime calendar support
- Return period and return level calculations
- Non-stationary extreme value analysis
- Peaks-over-threshold analysis
- Bootstrap confidence intervals
- xarray integration for gridded data

Import convention:
    import xtimeseries as xts
"""

__version__ = "0.1.0"

# Core distribution functions
from .distributions import (
    fit_gev,
    fit_gpd,
    gev_cdf,
    gev_pdf,
    gev_ppf,
    gpd_cdf,
    gpd_pdf,
    gpd_ppf,
)

# Block maxima extraction
from .block_maxima import (
    block_maxima,
    block_minima,
)

# Return period calculations
from .return_periods import (
    return_level,
    return_period,
    return_level_gpd,
    return_period_gpd,
)

# Confidence intervals
from .confidence import (
    bootstrap_ci,
    bootstrap_return_levels,
)

# Non-stationary analysis
from .nonstationary import (
    fit_nonstationary_gev,
    nonstationary_return_level,
    likelihood_ratio_test,
    effective_return_level,
)

# Peaks over threshold
from .pot import (
    peaks_over_threshold,
    decluster,
    mean_residual_life,
    threshold_stability,
    select_threshold,
)

# Synthetic data generation
from .synthetic import (
    generate_gev_series,
    generate_gpd_series,
    generate_nonstationary_series,
    generate_temperature_like,
    generate_precipitation_like,
    generate_gev_return_levels,
    generate_test_dataset,
)

# xarray integration
from ._xarray import (
    xr_fit_gev,
    xr_return_level,
    xr_block_maxima,
    xr_fit_nonstationary_gev,
)

# Diagnostics
from .diagnostics import (
    probability_plot,
    return_level_plot,
    qq_plot,
    diagnostic_plots,
)

__all__ = [
    # Version
    "__version__",
    # Distributions
    "fit_gev",
    "fit_gpd",
    "gev_cdf",
    "gev_pdf",
    "gev_ppf",
    "gpd_cdf",
    "gpd_pdf",
    "gpd_ppf",
    # Block maxima
    "block_maxima",
    "block_minima",
    # Return periods
    "return_level",
    "return_period",
    "return_level_gpd",
    "return_period_gpd",
    # Confidence intervals
    "bootstrap_ci",
    "bootstrap_return_levels",
    # Non-stationary
    "fit_nonstationary_gev",
    "nonstationary_return_level",
    "likelihood_ratio_test",
    "effective_return_level",
    # POT
    "peaks_over_threshold",
    "decluster",
    "mean_residual_life",
    "threshold_stability",
    "select_threshold",
    # Synthetic data
    "generate_gev_series",
    "generate_gpd_series",
    "generate_nonstationary_series",
    "generate_temperature_like",
    "generate_precipitation_like",
    "generate_gev_return_levels",
    "generate_test_dataset",
    # xarray
    "xr_fit_gev",
    "xr_return_level",
    "xr_block_maxima",
    "xr_fit_nonstationary_gev",
    # Diagnostics
    "probability_plot",
    "return_level_plot",
    "qq_plot",
    "diagnostic_plots",
]
