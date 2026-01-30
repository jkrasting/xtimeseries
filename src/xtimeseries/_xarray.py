"""
xarray integration for extreme value analysis.

This module provides vectorized operations on xarray DataArrays using
apply_ufunc, enabling efficient extreme value analysis on gridded data.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats
from typing import Literal


def xr_fit_gev(
    da: xr.DataArray,
    dim: str = "time",
) -> xr.Dataset:
    """
    Fit GEV distribution to DataArray along a dimension.

    Parameters
    ----------
    da : xarray.DataArray
        Input data (e.g., annual maxima).
    dim : str, default 'time'
        Dimension along which to fit.

    Returns
    -------
    xarray.Dataset
        Dataset with 'loc', 'scale', 'shape' variables containing
        fitted parameters at each grid point.

    Notes
    -----
    Uses xr.apply_ufunc for vectorized fitting. Grid points with
    fewer than 10 valid values return NaN.

    The shape parameter uses the climate convention (not scipy's c = -xi).

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create sample data
    >>> times = xr.date_range('1970', periods=50, freq='YE')
    >>> data = xr.DataArray(
    ...     np.random.gumbel(loc=30, scale=5, size=(50, 10, 10)),
    ...     dims=['time', 'lat', 'lon'],
    ...     coords={'time': times, 'lat': np.arange(10), 'lon': np.arange(10)}
    ... )
    >>> params = xr_fit_gev(data)
    >>> print(params)
    <xarray.Dataset>
    Dimensions:  (lat: 10, lon: 10)
    Data variables:
        loc      (lat, lon) float64 ...
        scale    (lat, lon) float64 ...
        shape    (lat, lon) float64 ...

    See Also
    --------
    fit_gev : Fit GEV to 1D array.
    xr_return_level : Calculate return levels for DataArray.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray")

    def _fit_gev_1d(arr):
        """Fit GEV to 1D array, handling NaN."""
        arr = arr[~np.isnan(arr)]
        if len(arr) < 10:
            return np.array([np.nan, np.nan, np.nan])
        try:
            c, loc, scale = stats.genextreme.fit(arr)
            return np.array([loc, scale, -c])  # Convert c to shape
        except Exception:
            return np.array([np.nan, np.nan, np.nan])

    result = xr.apply_ufunc(
        _fit_gev_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["param"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"param": 3}},
    )

    result = result.assign_coords(param=["loc", "scale", "shape"])

    # Convert to Dataset
    ds = result.to_dataset(dim="param")

    # Add attributes
    ds["loc"].attrs["long_name"] = "GEV location parameter"
    ds["scale"].attrs["long_name"] = "GEV scale parameter"
    ds["shape"].attrs["long_name"] = "GEV shape parameter (climate convention)"

    return ds


def xr_return_level(
    da: xr.DataArray,
    return_periods: list[float] | np.ndarray,
    dim: str = "time",
) -> xr.DataArray:
    """
    Calculate return levels for DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Input data (block maxima).
    return_periods : list or array
        Return periods in years.
    dim : str, default 'time'
        Dimension along which to fit and compute return levels.

    Returns
    -------
    xarray.DataArray
        Return levels with 'return_period' dimension added.

    Examples
    --------
    >>> rls = xr_return_level(annual_max, [10, 50, 100])
    >>> # Get 100-year return level
    >>> rl_100 = rls.sel(return_period=100)
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray")

    return_periods = np.asarray(return_periods)
    p = 1 - 1 / return_periods

    def _compute_return_levels(arr):
        """Compute return levels for 1D array."""
        arr = arr[~np.isnan(arr)]
        if len(arr) < 10:
            return np.full(len(return_periods), np.nan)
        try:
            c, loc, scale = stats.genextreme.fit(arr)
            return stats.genextreme.ppf(p, c=c, loc=loc, scale=scale)
        except Exception:
            return np.full(len(return_periods), np.nan)

    result = xr.apply_ufunc(
        _compute_return_levels,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["return_period"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"return_period": len(return_periods)}},
    )

    result = result.assign_coords(return_period=return_periods)
    result.attrs["long_name"] = "GEV return levels"
    result.attrs["units"] = da.attrs.get("units", "")

    return result


def xr_block_maxima(
    da: xr.DataArray,
    dim: str = "time",
    freq: str = "YE",
    min_periods: int | None = None,
) -> xr.DataArray:
    """
    Extract block maxima from DataArray.

    This is a convenience wrapper around block_maxima for xarray data.

    Parameters
    ----------
    da : xarray.DataArray
        Input time series.
    dim : str, default 'time'
        Time dimension name.
    freq : str, default 'YE'
        Resampling frequency.
    min_periods : int, optional
        Minimum observations per block.

    Returns
    -------
    xarray.DataArray
        Block maxima.

    See Also
    --------
    block_maxima : Core block maxima function.
    """
    from .block_maxima import block_maxima
    return block_maxima(da, dim=dim, freq=freq, min_periods=min_periods)


def xr_fit_nonstationary_gev(
    da: xr.DataArray,
    covariate: xr.DataArray | np.ndarray,
    dim: str = "time",
    trend_in: Literal["loc", "scale", "both"] = "loc",
) -> xr.Dataset:
    """
    Fit non-stationary GEV to gridded data.

    Parameters
    ----------
    da : xarray.DataArray
        Input data (block maxima).
    covariate : DataArray or array
        Covariate for trend (e.g., time in years). Must align with dim.
    dim : str, default 'time'
        Dimension along which to fit.
    trend_in : {'loc', 'scale', 'both'}, default 'loc'
        Which parameters have trend.

    Returns
    -------
    xarray.Dataset
        Dataset with fitted parameters including trend coefficients.

    Examples
    --------
    >>> # Fit with linear trend in location
    >>> years = annual_max.time.dt.year.values
    >>> params = xr_fit_nonstationary_gev(annual_max, years)
    >>> print(params['loc1'])  # Location trend slope
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray")

    # Convert covariate to array
    if isinstance(covariate, xr.DataArray):
        covariate = covariate.values
    covariate = np.asarray(covariate)

    # Standardize covariate
    cov_mean = np.mean(covariate)
    cov_std = np.std(covariate)
    if cov_std < 1e-10:
        cov_std = 1.0
    cov_scaled = (covariate - cov_mean) / cov_std

    def _fit_ns_gev_1d(arr):
        """Fit non-stationary GEV to 1D array."""
        from scipy import optimize

        arr = np.asarray(arr)
        valid = ~np.isnan(arr)
        if valid.sum() < 10:
            return np.array([np.nan] * 5)

        arr_valid = arr[valid]
        cov_valid = cov_scaled[valid]

        # Initial estimates
        try:
            c_init, loc_init, scale_init = stats.genextreme.fit(arr_valid)
        except Exception:
            return np.array([np.nan] * 5)

        def neg_log_lik(params):
            loc0, loc1, log_scale, shape = params
            loc_t = loc0 + loc1 * cov_valid
            scale_t = np.exp(log_scale)
            try:
                ll = np.sum(stats.genextreme.logpdf(arr_valid, c=-shape, loc=loc_t, scale=scale_t))
                return -ll if np.isfinite(ll) else 1e10
            except Exception:
                return 1e10

        x0 = [loc_init, 0.0, np.log(scale_init), -c_init]
        bounds = [(None, None), (None, None), (-5, 5), (-1, 1)]

        try:
            result = optimize.minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
            loc0, loc1, log_scale, shape = result.x
            # Convert back to original covariate scale
            loc1_orig = loc1 / cov_std
            loc0_orig = loc0 - loc1 * cov_mean / cov_std
            return np.array([loc0_orig, loc1_orig, np.exp(log_scale), shape, result.fun])
        except Exception:
            return np.array([np.nan] * 5)

    result = xr.apply_ufunc(
        _fit_ns_gev_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["param"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"param": 5}},
    )

    result = result.assign_coords(param=["loc0", "loc1", "scale", "shape", "nllh"])

    ds = result.to_dataset(dim="param")

    ds["loc0"].attrs["long_name"] = "Location intercept"
    ds["loc1"].attrs["long_name"] = "Location slope per covariate unit"
    ds["scale"].attrs["long_name"] = "Scale parameter"
    ds["shape"].attrs["long_name"] = "Shape parameter"
    ds["nllh"].attrs["long_name"] = "Negative log-likelihood"

    return ds


def xr_bootstrap_ci(
    da: xr.DataArray,
    return_periods: list[float] | np.ndarray,
    dim: str = "time",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int | None = None,
) -> xr.Dataset:
    """
    Compute bootstrap confidence intervals for return levels on gridded data.

    Parameters
    ----------
    da : xarray.DataArray
        Input data.
    return_periods : list or array
        Return periods.
    dim : str, default 'time'
        Fitting dimension.
    n_bootstrap : int, default 500
        Number of bootstrap samples.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    xarray.Dataset
        Dataset with 'return_level', 'lower', 'upper', 'se' variables.

    Notes
    -----
    This can be slow for large grids. Consider using a subset or
    parallelization via dask.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray")

    return_periods = np.asarray(return_periods)
    p = 1 - 1 / return_periods

    rng = np.random.default_rng(seed)

    def _bootstrap_ci_1d(arr):
        """Bootstrap CI for 1D array."""
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 10:
            nan_arr = np.full((4, len(return_periods)), np.nan)
            return nan_arr

        # Point estimate
        try:
            c, loc, scale = stats.genextreme.fit(arr)
            point_est = stats.genextreme.ppf(p, c=c, loc=loc, scale=scale)
        except Exception:
            return np.full((4, len(return_periods)), np.nan)

        # Bootstrap
        boot_rls = np.zeros((n_bootstrap, len(return_periods)))
        for i in range(n_bootstrap):
            boot_sample = rng.choice(arr, size=n, replace=True)
            try:
                c_b, loc_b, scale_b = stats.genextreme.fit(boot_sample)
                boot_rls[i, :] = stats.genextreme.ppf(p, c=c_b, loc=loc_b, scale=scale_b)
            except Exception:
                boot_rls[i, :] = np.nan

        # Remove failed fits
        valid = ~np.any(np.isnan(boot_rls), axis=1)
        boot_rls = boot_rls[valid, :]

        if len(boot_rls) < n_bootstrap * 0.5:
            return np.full((4, len(return_periods)), np.nan)

        lower = np.percentile(boot_rls, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_rls, 100 * (1 - alpha / 2), axis=0)
        se = np.std(boot_rls, axis=0)

        return np.stack([point_est, lower, upper, se])

    result = xr.apply_ufunc(
        _bootstrap_ci_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["stat", "return_period"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"stat": 4, "return_period": len(return_periods)}},
    )

    result = result.assign_coords(
        stat=["return_level", "lower", "upper", "se"],
        return_period=return_periods,
    )

    ds = result.to_dataset(dim="stat")

    ds["return_level"].attrs["long_name"] = "GEV return levels"
    ds["lower"].attrs["long_name"] = f"Lower {100*(1-alpha):.0f}% CI bound"
    ds["upper"].attrs["long_name"] = f"Upper {100*(1-alpha):.0f}% CI bound"
    ds["se"].attrs["long_name"] = "Bootstrap standard error"

    return ds
