"""
Synthetic data generation for testing extreme value analysis.

This module provides functions for generating synthetic time series
with known statistical properties, which is essential for validating
extreme value analysis algorithms.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
import xarray as xr
from typing import Literal


def generate_gev_series(
    n: int,
    loc: float,
    scale: float,
    shape: float,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate synthetic GEV samples with known parameters.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    loc : float
        Location parameter (mu).
    scale : float
        Scale parameter (sigma). Must be positive.
    shape : float
        Shape parameter (xi). Uses climate convention.
    seed : int or Generator, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray
        Array of GEV random variates.

    Examples
    --------
    >>> from xtimeseries import generate_gev_series, fit_gev
    >>> data = generate_gev_series(100, loc=30, scale=5, shape=0.1, seed=42)
    >>> params = fit_gev(data)
    >>> print(f"Recovered shape: {params['shape']:.3f} (true: 0.1)")

    See Also
    --------
    generate_gpd_series : Generate GPD samples.
    generate_nonstationary_series : Generate GEV with trend.
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")

    rng = _get_rng(seed)

    # scipy uses c = -shape
    return stats.genextreme.rvs(c=-shape, loc=loc, scale=scale, size=n, random_state=rng)


def generate_gpd_series(
    n: int,
    scale: float,
    shape: float,
    threshold: float = 0.0,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate synthetic GPD samples (exceedances).

    Parameters
    ----------
    n : int
        Number of exceedances to generate.
    scale : float
        Scale parameter (sigma).
    shape : float
        Shape parameter (xi).
    threshold : float, default 0.0
        Threshold value to add to exceedances.
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    ndarray
        Array of values above threshold.

    Examples
    --------
    >>> from xtimeseries import generate_gpd_series, fit_gpd
    >>> exceedances = generate_gpd_series(200, scale=5, shape=0.2, seed=42)
    >>> params = fit_gpd(exceedances)
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")

    rng = _get_rng(seed)

    exceedances = stats.genpareto.rvs(c=shape, scale=scale, size=n, random_state=rng)
    return exceedances + threshold


def generate_nonstationary_series(
    n: int,
    loc_intercept: float,
    loc_slope: float,
    scale: float,
    shape: float,
    scale_slope: float = 0.0,
    seed: int | np.random.Generator | None = None,
) -> dict:
    """
    Generate GEV samples with time-varying location (and optionally scale).

    Parameters
    ----------
    n : int
        Number of years/samples.
    loc_intercept : float
        Location at time 0 (mu_0).
    loc_slope : float
        Trend in location per time step (mu_1).
    scale : float
        Scale parameter (sigma) or log-scale intercept.
    shape : float
        Shape parameter (xi), held constant.
    scale_slope : float, default 0.0
        Trend in log-scale per time step.
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with:
        - 'data': generated values
        - 'time': time indices (0 to n-1)
        - 'true_loc': time-varying location
        - 'true_scale': time-varying scale
        - 'true_params': dict of true parameter values

    Examples
    --------
    >>> from xtimeseries import generate_nonstationary_series, fit_nonstationary_gev
    >>> result = generate_nonstationary_series(
    ...     n=100, loc_intercept=30, loc_slope=0.05, scale=5, shape=-0.1, seed=42
    ... )
    >>> # Fit and recover trend
    >>> params = fit_nonstationary_gev(result['data'], result['time'])
    >>> print(f"Recovered trend: {params['loc1']:.4f} (true: 0.05)")

    See Also
    --------
    fit_nonstationary_gev : Fit non-stationary GEV.
    """
    rng = _get_rng(seed)

    time = np.arange(n)

    # Time-varying parameters
    loc_t = loc_intercept + loc_slope * time
    scale_t = np.exp(np.log(scale) + scale_slope * time)

    # Generate data
    data = np.array([
        stats.genextreme.rvs(c=-shape, loc=loc, scale=sc, random_state=rng)
        for loc, sc in zip(loc_t, scale_t)
    ])

    return {
        "data": data,
        "time": time,
        "true_loc": loc_t,
        "true_scale": scale_t,
        "true_params": {
            "loc_intercept": loc_intercept,
            "loc_slope": loc_slope,
            "scale": scale,
            "scale_slope": scale_slope,
            "shape": shape,
        },
    }


def generate_temperature_like(
    n_years: int,
    mean_annual: float = 15.0,
    seasonal_amplitude: float = 15.0,
    daily_std: float = 3.0,
    trend_per_decade: float = 0.0,
    ar1_coef: float = 0.8,
    calendar: Literal["standard", "noleap", "360_day"] = "standard",
    start_year: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> xr.DataArray:
    """
    Generate realistic synthetic daily temperature time series.

    Creates a time series with:
    - Mean seasonal cycle
    - Long-term trend
    - Autocorrelated daily variability

    Parameters
    ----------
    n_years : int
        Number of years to generate.
    mean_annual : float, default 15.0
        Mean annual temperature (degrees C).
    seasonal_amplitude : float, default 15.0
        Amplitude of seasonal cycle (degrees C).
    daily_std : float, default 3.0
        Standard deviation of daily variability.
    trend_per_decade : float, default 0.0
        Linear trend per decade (degrees C).
    ar1_coef : float, default 0.8
        AR(1) autocorrelation coefficient for daily variability.
    calendar : {'standard', 'noleap', '360_day'}, default 'standard'
        Calendar type for time coordinate.
    start_year : int, default 2000
        Starting year.
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    xarray.DataArray
        Daily temperature time series with time coordinate.

    Examples
    --------
    >>> from xtimeseries import generate_temperature_like, block_maxima, fit_gev
    >>> temp = generate_temperature_like(50, trend_per_decade=0.3, seed=42)
    >>> annual_max = block_maxima(temp)
    >>> params = fit_gev(annual_max.values)

    See Also
    --------
    generate_precipitation_like : Generate precipitation data.
    """
    rng = _get_rng(seed)

    # Determine number of days based on calendar
    if calendar == "360_day":
        days_per_year = 360
    elif calendar == "noleap":
        days_per_year = 365
    else:
        days_per_year = 365  # Approximate for standard

    n_days = n_years * days_per_year

    # Create time coordinate
    if calendar == "standard":
        times = xr.date_range(
            f"{start_year}-01-01",
            periods=n_days,
            freq="D",
        )
    else:
        times = xr.cftime_range(
            f"{start_year:04d}-01-01",
            periods=n_days,
            freq="D",
            calendar=calendar,
        )

    # Day of year and fractional year
    if calendar == "standard":
        doy = np.array([t.timetuple().tm_yday for t in times.to_pydatetime()])
        years = np.array([t.year + t.timetuple().tm_yday / days_per_year for t in times.to_pydatetime()])
    else:
        doy = np.array([t.dayofyr for t in times])
        years = np.array([t.year + t.dayofyr / days_per_year for t in times])

    # Seasonal cycle (peak around day 200 for northern hemisphere summer)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * (doy - 105) / days_per_year)

    # Long-term trend
    trend = trend_per_decade * (years - start_year) / 10.0

    # AR(1) residuals
    innovation_std = daily_std * np.sqrt(1 - ar1_coef**2)
    residuals = np.zeros(n_days)
    residuals[0] = rng.normal(0, daily_std)
    for t in range(1, n_days):
        residuals[t] = ar1_coef * residuals[t - 1] + innovation_std * rng.normal()

    # Combine components
    temperature = mean_annual + seasonal + trend + residuals

    # Create DataArray
    da = xr.DataArray(
        temperature,
        dims=["time"],
        coords={"time": times},
        name="temperature",
        attrs={
            "units": "degC",
            "long_name": "Synthetic daily temperature",
            "true_mean_annual": mean_annual,
            "true_seasonal_amplitude": seasonal_amplitude,
            "true_daily_std": daily_std,
            "true_trend_per_decade": trend_per_decade,
            "true_ar1_coef": ar1_coef,
        },
    )

    return da


def generate_precipitation_like(
    n_years: int,
    wet_prob_dry: float = 0.3,
    wet_prob_wet: float = 0.6,
    gamma_shape: float = 0.7,
    gamma_scale: float = 5.0,
    seasonal_wet_amplitude: float = 0.1,
    heavy_tail_prob: float = 0.05,
    heavy_tail_multiplier: float = 3.0,
    calendar: Literal["standard", "noleap", "360_day"] = "standard",
    start_year: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> xr.DataArray:
    """
    Generate realistic synthetic daily precipitation time series.

    Creates precipitation with:
    - Markov chain for wet/dry occurrence
    - Gamma distribution for wet-day amounts
    - Heavy tail modification for extreme events

    Parameters
    ----------
    n_years : int
        Number of years.
    wet_prob_dry : float, default 0.3
        Probability of wet day given previous dry day.
    wet_prob_wet : float, default 0.6
        Probability of wet day given previous wet day.
    gamma_shape : float, default 0.7
        Shape parameter for gamma distribution of amounts.
    gamma_scale : float, default 5.0
        Scale parameter for gamma distribution (mm).
    seasonal_wet_amplitude : float, default 0.1
        Amplitude of seasonal variation in wet probability.
    heavy_tail_prob : float, default 0.05
        Probability of drawing from heavy-tailed distribution.
    heavy_tail_multiplier : float, default 3.0
        Multiplier for heavy tail scale.
    calendar : {'standard', 'noleap', '360_day'}, default 'standard'
        Calendar type.
    start_year : int, default 2000
        Starting year.
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    xarray.DataArray
        Daily precipitation time series (mm).

    Examples
    --------
    >>> from xtimeseries import generate_precipitation_like, block_maxima
    >>> precip = generate_precipitation_like(50, seed=42)
    >>> annual_max = block_maxima(precip)
    >>> print(f"Mean annual max: {annual_max.mean().values:.1f} mm")
    """
    rng = _get_rng(seed)

    # Determine days per year
    if calendar == "360_day":
        days_per_year = 360
    elif calendar == "noleap":
        days_per_year = 365
    else:
        days_per_year = 365

    n_days = n_years * days_per_year

    # Create time coordinate
    if calendar == "standard":
        times = xr.date_range(f"{start_year}-01-01", periods=n_days, freq="D")
        doy = np.array([t.timetuple().tm_yday for t in times.to_pydatetime()])
    else:
        times = xr.cftime_range(
            f"{start_year:04d}-01-01", periods=n_days, freq="D", calendar=calendar
        )
        doy = np.array([t.dayofyr for t in times])

    # Seasonal modulation of wet probability
    seasonal_mod = seasonal_wet_amplitude * np.sin(2 * np.pi * (doy - 91) / days_per_year)

    # Initialize arrays
    is_wet = np.zeros(n_days, dtype=bool)
    precip = np.zeros(n_days)

    # First day
    is_wet[0] = rng.random() < (wet_prob_dry + wet_prob_wet) / 2

    # Markov chain for wet/dry occurrence
    for t in range(1, n_days):
        if is_wet[t - 1]:
            p_wet = np.clip(wet_prob_wet + seasonal_mod[t], 0, 1)
        else:
            p_wet = np.clip(wet_prob_dry + seasonal_mod[t], 0, 1)
        is_wet[t] = rng.random() < p_wet

    # Generate amounts for wet days
    n_wet = is_wet.sum()

    # Mixture: mostly gamma, some heavy-tailed (GPD)
    use_heavy = rng.random(n_wet) < heavy_tail_prob

    # Gamma for normal rainfall
    amounts = stats.gamma.rvs(a=gamma_shape, scale=gamma_scale, size=n_wet, random_state=rng)

    # Replace some with heavy-tailed values
    n_heavy = use_heavy.sum()
    if n_heavy > 0:
        heavy_amounts = stats.genpareto.rvs(
            c=0.3, scale=gamma_scale * heavy_tail_multiplier, size=n_heavy, random_state=rng
        )
        amounts[use_heavy] = heavy_amounts + gamma_scale

    precip[is_wet] = amounts

    # Create DataArray
    da = xr.DataArray(
        precip,
        dims=["time"],
        coords={"time": times},
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Synthetic daily precipitation",
            "true_wet_prob_dry": wet_prob_dry,
            "true_wet_prob_wet": wet_prob_wet,
            "true_gamma_shape": gamma_shape,
            "true_gamma_scale": gamma_scale,
        },
    )

    return da


def generate_gev_return_levels(
    loc: float,
    scale: float,
    shape: float,
    return_periods: ArrayLike | None = None,
) -> dict:
    """
    Calculate analytical (true) return levels for known GEV parameters.

    Useful for validating return level estimation algorithms.

    Parameters
    ----------
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    shape : float
        Shape parameter (climate convention).
    return_periods : array-like, optional
        Return periods. Default is [2, 5, 10, 20, 50, 100, 200, 500, 1000].

    Returns
    -------
    dict
        Dictionary with 'return_periods' and 'return_levels' arrays.

    Examples
    --------
    >>> from xtimeseries import generate_gev_return_levels
    >>> true_rls = generate_gev_return_levels(loc=30, scale=5, shape=0.1)
    >>> print(f"True 100-year level: {true_rls['return_levels'][5]:.2f}")
    """
    if return_periods is None:
        return_periods = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    else:
        return_periods = np.asarray(return_periods)

    p = 1 - 1 / return_periods
    return_levels = stats.genextreme.ppf(p, c=-shape, loc=loc, scale=scale)

    return {
        "return_periods": return_periods,
        "return_levels": return_levels,
        "params": {"loc": loc, "scale": scale, "shape": shape},
    }


def generate_test_dataset(
    n_years: int = 50,
    nlat: int = 5,
    nlon: int = 5,
    base_loc: float = 25.0,
    loc_lat_gradient: float = 0.5,
    scale: float = 5.0,
    shape: float = -0.1,
    seed: int | np.random.Generator | None = None,
) -> xr.Dataset:
    """
    Generate 3D xarray test dataset with spatial gradients.

    Parameters
    ----------
    n_years : int, default 50
        Number of years.
    nlat, nlon : int, default 5
        Grid dimensions.
    base_loc : float, default 25.0
        Base location parameter.
    loc_lat_gradient : float, default 0.5
        Gradient in location with latitude.
    scale : float, default 5.0
        Scale parameter (constant).
    shape : float, default -0.1
        Shape parameter (constant).
    seed : int or Generator, optional
        Random seed.

    Returns
    -------
    xarray.Dataset
        Dataset with 'annual_max' and 'true_location' variables.
    """
    rng = _get_rng(seed)

    # Create coordinates
    times = xr.date_range("2000-01-01", periods=n_years, freq="YE")
    lats = np.linspace(20, 50, nlat)
    lons = np.linspace(-120, -70, nlon)

    # Spatial variation in location
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    loc_spatial = base_loc + loc_lat_gradient * (lat_grid - lats.mean())

    # Generate data
    data = np.zeros((n_years, nlat, nlon))
    for i in range(nlat):
        for j in range(nlon):
            data[:, i, j] = stats.genextreme.rvs(
                c=-shape,
                loc=loc_spatial[i, j],
                scale=scale,
                size=n_years,
                random_state=rng,
            )

    ds = xr.Dataset(
        {
            "annual_max": (["time", "lat", "lon"], data),
            "true_location": (["lat", "lon"], loc_spatial),
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "true_scale": scale,
            "true_shape": shape,
            "description": "Synthetic test data with known GEV parameters",
        },
    )

    return ds


def _get_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Get random number generator from seed."""
    if seed is None:
        return np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        return seed
    else:
        return np.random.default_rng(seed)
