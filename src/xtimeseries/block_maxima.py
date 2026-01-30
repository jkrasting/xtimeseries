"""
Block maxima extraction for extreme value analysis.

This module provides functions for extracting block maxima (and minima)
from time series data, with full support for xarray DataArrays and
cftime calendars including noleap and year 0001 start dates.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from typing import Literal, Union
import warnings


def block_maxima(
    data: Union[xr.DataArray, ArrayLike],
    dim: str = "time",
    freq: str = "YE",
    min_periods: int | None = None,
    skipna: bool = True,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Extract block maxima from a time series.

    This function extracts the maximum value within each block (e.g., year,
    season, month) for use in GEV fitting. It supports both numpy arrays
    and xarray DataArrays with standard or cftime calendars.

    Parameters
    ----------
    data : xarray.DataArray or array-like
        Input time series. If DataArray, must have a time dimension.
        If array-like, treated as annual values (one per block).
    dim : str, default 'time'
        Name of the time dimension (for DataArray input).
    freq : str, default 'YE'
        Resampling frequency. Common values:
        - 'YE' or 'YS': Annual (year end or year start)
        - 'QE' or 'QS': Quarterly
        - 'ME' or 'MS': Monthly
        - 'D': Daily (for sub-daily data)
    min_periods : int, optional
        Minimum number of observations required per block. If None,
        defaults to 1. Blocks with fewer observations return NaN.
    skipna : bool, default True
        If True, skip NaN values when computing maxima.

    Returns
    -------
    xarray.DataArray or ndarray
        Block maxima values. For DataArray input, returns DataArray with
        resampled time coordinate. For array input, returns the input
        unchanged (assumed to already be block maxima).

    Notes
    -----
    For climate data with non-standard calendars (noleap, 360_day), this
    function uses xarray's built-in resampling which properly handles
    cftime objects.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create daily temperature data
    >>> times = xr.date_range('2000-01-01', periods=365*10, freq='D')
    >>> temp = xr.DataArray(
    ...     20 + 10*np.sin(2*np.pi*np.arange(365*10)/365) + np.random.randn(365*10)*3,
    ...     dims=['time'],
    ...     coords={'time': times}
    ... )
    >>> annual_max = block_maxima(temp)
    >>> print(len(annual_max))
    10

    >>> # With noleap calendar
    >>> times_noleap = xr.cftime_range('0001-01-01', periods=365*50, freq='D', calendar='noleap')
    >>> temp_noleap = xr.DataArray(np.random.randn(365*50), dims=['time'], coords={'time': times_noleap})
    >>> annual_max = block_maxima(temp_noleap)

    See Also
    --------
    block_minima : Extract block minima.
    """
    # Handle numpy arrays
    if not isinstance(data, xr.DataArray):
        data = np.asarray(data)
        if skipna:
            return data  # Assume already block maxima
        return data

    # Validate time dimension exists
    if dim not in data.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray. Available: {list(data.dims)}")

    # Set default min_periods
    if min_periods is None:
        min_periods = 1

    # Use xarray resampling (handles cftime automatically)
    resampled = data.resample({dim: freq})

    if skipna:
        result = resampled.max(skipna=True)
    else:
        result = resampled.max(skipna=False)

    # Copy attributes
    result.attrs = data.attrs.copy()
    result.attrs["block_maxima_freq"] = freq

    return result


def block_minima(
    data: Union[xr.DataArray, ArrayLike],
    dim: str = "time",
    freq: str = "YE",
    min_periods: int | None = None,
    skipna: bool = True,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Extract block minima from a time series.

    This function extracts the minimum value within each block (e.g., year,
    season, month). Useful for analyzing cold extremes or drought.

    Parameters
    ----------
    data : xarray.DataArray or array-like
        Input time series.
    dim : str, default 'time'
        Name of the time dimension.
    freq : str, default 'YE'
        Resampling frequency.
    min_periods : int, optional
        Minimum observations per block.
    skipna : bool, default True
        Skip NaN values when computing minima.

    Returns
    -------
    xarray.DataArray or ndarray
        Block minima values.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> times = xr.date_range('2000-01-01', periods=365*10, freq='D')
    >>> temp = xr.DataArray(
    ...     10 + 15*np.sin(2*np.pi*np.arange(365*10)/365) + np.random.randn(365*10)*3,
    ...     dims=['time'],
    ...     coords={'time': times}
    ... )
    >>> annual_min = block_minima(temp)

    See Also
    --------
    block_maxima : Extract block maxima.

    Notes
    -----
    For analyzing cold extremes with GEV, you may want to negate the minima
    and treat them as maxima: ``-block_minima(data)``. This allows using
    standard GEV fitting and return level calculations.
    """
    # Handle numpy arrays
    if not isinstance(data, xr.DataArray):
        data = np.asarray(data)
        return data  # Assume already block minima

    # Validate time dimension exists
    if dim not in data.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray. Available: {list(data.dims)}")

    # Set default min_periods
    if min_periods is None:
        min_periods = 1

    # Use xarray resampling
    resampled = data.resample({dim: freq})

    if skipna:
        result = resampled.min(skipna=True)
    else:
        result = resampled.min(skipna=False)

    # Copy attributes
    result.attrs = data.attrs.copy()
    result.attrs["block_minima_freq"] = freq

    return result


def seasonal_block_maxima(
    data: xr.DataArray,
    dim: str = "time",
    season: Literal["DJF", "MAM", "JJA", "SON"] | None = None,
    min_periods: int | None = None,
) -> xr.DataArray:
    """
    Extract seasonal block maxima.

    Parameters
    ----------
    data : xarray.DataArray
        Input time series with time dimension.
    dim : str, default 'time'
        Name of the time dimension.
    season : {'DJF', 'MAM', 'JJA', 'SON'}, optional
        Season to extract. If None, returns maxima for all seasons.
    min_periods : int, optional
        Minimum observations per season.

    Returns
    -------
    xarray.DataArray
        Seasonal maxima.

    Examples
    --------
    >>> # Get summer (JJA) maxima
    >>> summer_max = seasonal_block_maxima(temp, season='JJA')

    >>> # Get all seasonal maxima
    >>> all_seasonal = seasonal_block_maxima(temp)
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray")

    if min_periods is None:
        min_periods = 1

    # Group by season
    seasonal = data.groupby(f"{dim}.season")

    if season is not None:
        # Extract specific season first
        season_data = seasonal.groups.get(season)
        if season_data is None:
            raise ValueError(f"Season '{season}' not found in data")
        data_season = data.isel({dim: list(seasonal.groups[season])})
        # Then resample annually
        result = data_season.resample({dim: "YE"}).max(skipna=True)
        result.attrs["season"] = season
    else:
        # Get max for each season-year combination
        result = data.resample({dim: "QE-NOV"}).max(skipna=True)

    result.attrs = data.attrs.copy()
    result.attrs["block_type"] = "seasonal_maxima"

    return result


def count_per_block(
    data: xr.DataArray,
    dim: str = "time",
    freq: str = "YE",
) -> xr.DataArray:
    """
    Count valid (non-NaN) observations per block.

    Useful for quality control and determining min_periods.

    Parameters
    ----------
    data : xarray.DataArray
        Input time series.
    dim : str, default 'time'
        Time dimension name.
    freq : str, default 'YE'
        Resampling frequency.

    Returns
    -------
    xarray.DataArray
        Count of valid observations per block.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray")

    return data.resample({dim: freq}).count(dim=dim)
