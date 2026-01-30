"""
Peaks-over-threshold (POT) analysis.

This module provides functions for threshold selection and extraction
of exceedances for GPD analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
from scipy import stats
from typing import Union
import warnings


def peaks_over_threshold(
    data: Union[xr.DataArray, ArrayLike],
    threshold: float,
    decluster: bool = True,
    run_length: int = 3,
    dim: str = "time",
) -> dict:
    """
    Extract peaks over threshold from a time series.

    Parameters
    ----------
    data : DataArray or array-like
        Input time series.
    threshold : float
        Threshold value.
    decluster : bool, default True
        If True, apply declustering to ensure independence.
    run_length : int, default 3
        Minimum separation (in time steps) between independent exceedances.
    dim : str, default 'time'
        Time dimension name (for DataArray).

    Returns
    -------
    dict
        Dictionary containing:
        - 'exceedances': values above threshold (minus threshold)
        - 'values': actual values above threshold
        - 'indices': indices of exceedances in original array
        - 'times': times of exceedances (if DataArray)
        - 'threshold': threshold value
        - 'n_exceedances': number of exceedances
        - 'n_total': total number of observations
        - 'rate': exceedance rate (exceedances per time step)

    Examples
    --------
    >>> import numpy as np
    >>> from xtimeseries import peaks_over_threshold, fit_gpd
    >>> np.random.seed(42)
    >>> data = np.random.exponential(scale=10, size=1000)
    >>> pot = peaks_over_threshold(data, threshold=20)
    >>> params = fit_gpd(pot['exceedances'])

    See Also
    --------
    mean_residual_life : Tool for threshold selection.
    threshold_stability : Parameter stability plot.
    decluster : Declustering function.

    Notes
    -----
    For GPD fitting, use the 'exceedances' key which contains values
    shifted by the threshold: (X - u) where X > u.
    """
    # Handle DataArray
    if isinstance(data, xr.DataArray):
        times = data.coords.get(dim)
        data_arr = data.values.flatten()
    else:
        times = None
        data_arr = np.asarray(data).flatten()

    # Remove NaN
    valid_mask = ~np.isnan(data_arr)
    data_clean = data_arr[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Find exceedances
    exceed_mask = data_clean > threshold
    exceed_indices_clean = np.where(exceed_mask)[0]
    exceed_values = data_clean[exceed_mask]

    if len(exceed_values) == 0:
        return {
            "exceedances": np.array([]),
            "values": np.array([]),
            "indices": np.array([], dtype=int),
            "times": None,
            "threshold": threshold,
            "n_exceedances": 0,
            "n_total": len(data_clean),
            "rate": 0.0,
        }

    # Map back to original indices
    exceed_indices = valid_indices[exceed_indices_clean]

    if decluster:
        # Apply declustering
        cluster_result = _decluster_indices(
            exceed_indices, exceed_values, run_length
        )
        exceed_indices = cluster_result["indices"]
        exceed_values = cluster_result["values"]

    exceedances = exceed_values - threshold

    # Get times if available
    exceed_times = None
    if times is not None:
        exceed_times = times.values[exceed_indices]

    return {
        "exceedances": exceedances,
        "values": exceed_values,
        "indices": exceed_indices,
        "times": exceed_times,
        "threshold": threshold,
        "n_exceedances": len(exceedances),
        "n_total": len(data_clean),
        "rate": len(exceedances) / len(data_clean),
    }


def decluster(
    indices: ArrayLike,
    values: ArrayLike,
    run_length: int = 3,
) -> dict:
    """
    Decluster exceedances to ensure independence.

    Declustering identifies clusters of consecutive or nearby exceedances
    and retains only the maximum within each cluster.

    Parameters
    ----------
    indices : array-like
        Indices of exceedances in original time series.
    values : array-like
        Values of exceedances.
    run_length : int, default 3
        Minimum separation between independent events.

    Returns
    -------
    dict
        Dictionary with 'indices' and 'values' of declustered exceedances.

    Examples
    --------
    >>> indices = np.array([10, 11, 12, 50, 100, 101])
    >>> values = np.array([25, 28, 26, 30, 22, 24])
    >>> result = decluster(indices, values, run_length=3)
    >>> print(result['indices'])  # [11, 50, 101] - cluster maxima
    """
    return _decluster_indices(np.asarray(indices), np.asarray(values), run_length)


def _decluster_indices(
    indices: np.ndarray,
    values: np.ndarray,
    run_length: int,
) -> dict:
    """Internal declustering implementation."""
    if len(indices) == 0:
        return {"indices": np.array([], dtype=int), "values": np.array([])}

    if len(indices) == 1:
        return {"indices": indices, "values": values}

    # Sort by index
    sort_idx = np.argsort(indices)
    indices = indices[sort_idx]
    values = values[sort_idx]

    # Find cluster boundaries
    gaps = np.diff(indices)
    cluster_breaks = np.where(gaps > run_length)[0] + 1
    cluster_starts = np.concatenate([[0], cluster_breaks])
    cluster_ends = np.concatenate([cluster_breaks, [len(indices)]])

    # Take maximum within each cluster
    result_indices = []
    result_values = []

    for start, end in zip(cluster_starts, cluster_ends):
        cluster_values = values[start:end]
        max_pos = np.argmax(cluster_values)
        result_indices.append(indices[start + max_pos])
        result_values.append(cluster_values[max_pos])

    return {
        "indices": np.array(result_indices),
        "values": np.array(result_values),
    }


def mean_residual_life(
    data: ArrayLike,
    thresholds: ArrayLike | None = None,
    n_thresholds: int = 50,
) -> dict:
    """
    Compute mean residual life plot data for threshold selection.

    The mean residual life is the mean of exceedances above a threshold.
    For GPD, this should be linear in the threshold. Deviations from
    linearity suggest the threshold is too low.

    Parameters
    ----------
    data : array-like
        Input data.
    thresholds : array-like, optional
        Thresholds to evaluate. If None, uses quantiles from 50th to 99th.
    n_thresholds : int, default 50
        Number of thresholds to evaluate if thresholds is None.

    Returns
    -------
    dict
        Dictionary containing:
        - 'thresholds': evaluated threshold values
        - 'mean_excess': mean excess above each threshold
        - 'n_exceed': number of exceedances at each threshold
        - 'se': standard error of mean excess

    Notes
    -----
    For a GPD with shape xi and scale sigma:

    .. math::

        E[X - u | X > u] = \\frac{\\sigma + \\xi u}{1 - \\xi}

    This is linear in u. Choose the lowest threshold above which
    the mean residual life is approximately linear.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.exponential(scale=10, size=1000)
    >>> mrl = mean_residual_life(data)
    >>> # Plot: plt.errorbar(mrl['thresholds'], mrl['mean_excess'], yerr=mrl['se'])

    See Also
    --------
    threshold_stability : Alternative threshold selection tool.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if thresholds is None:
        # Use quantiles from 50th to 99th percentile
        quantiles = np.linspace(0.5, 0.99, n_thresholds)
        thresholds = np.percentile(data, 100 * quantiles)

    thresholds = np.asarray(thresholds)

    mean_excess = np.zeros(len(thresholds))
    n_exceed = np.zeros(len(thresholds), dtype=int)
    se = np.zeros(len(thresholds))

    for i, u in enumerate(thresholds):
        exceedances = data[data > u] - u
        n = len(exceedances)
        n_exceed[i] = n

        if n > 1:
            mean_excess[i] = np.mean(exceedances)
            se[i] = np.std(exceedances, ddof=1) / np.sqrt(n)
        else:
            mean_excess[i] = np.nan
            se[i] = np.nan

    return {
        "thresholds": thresholds,
        "mean_excess": mean_excess,
        "n_exceed": n_exceed,
        "se": se,
    }


def threshold_stability(
    data: ArrayLike,
    thresholds: ArrayLike | None = None,
    n_thresholds: int = 30,
) -> dict:
    """
    Compute parameter stability plot data for threshold selection.

    Fits GPD at multiple thresholds and examines parameter stability.
    The threshold should be chosen where parameters stabilize.

    Parameters
    ----------
    data : array-like
        Input data.
    thresholds : array-like, optional
        Thresholds to evaluate.
    n_thresholds : int, default 30
        Number of thresholds if not specified.

    Returns
    -------
    dict
        Dictionary containing:
        - 'thresholds': evaluated thresholds
        - 'shape': fitted shape parameters
        - 'modified_scale': sigma - xi * u (should be constant)
        - 'n_exceed': number of exceedances

    Notes
    -----
    The modified scale (sigma - xi * u) should be constant above
    the appropriate threshold. The shape parameter should also
    stabilize.

    Examples
    --------
    >>> stab = threshold_stability(data)
    >>> # Plot shape vs threshold to find where it stabilizes
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if thresholds is None:
        quantiles = np.linspace(0.7, 0.98, n_thresholds)
        thresholds = np.percentile(data, 100 * quantiles)

    thresholds = np.asarray(thresholds)

    shape = np.zeros(len(thresholds))
    scale = np.zeros(len(thresholds))
    modified_scale = np.zeros(len(thresholds))
    n_exceed = np.zeros(len(thresholds), dtype=int)

    for i, u in enumerate(thresholds):
        exceedances = data[data > u] - u
        n = len(exceedances)
        n_exceed[i] = n

        if n < 10:
            shape[i] = np.nan
            scale[i] = np.nan
            modified_scale[i] = np.nan
            continue

        try:
            xi, _, sigma = stats.genpareto.fit(exceedances, floc=0)
            shape[i] = xi
            scale[i] = sigma
            modified_scale[i] = sigma - xi * u
        except Exception:
            shape[i] = np.nan
            scale[i] = np.nan
            modified_scale[i] = np.nan

    return {
        "thresholds": thresholds,
        "shape": shape,
        "scale": scale,
        "modified_scale": modified_scale,
        "n_exceed": n_exceed,
    }


def select_threshold(
    data: ArrayLike,
    method: str = "quantile",
    quantile: float = 0.95,
    n_per_year: float | None = None,
    observations_per_year: float = 365.25,
) -> float:
    """
    Select threshold for POT analysis.

    Parameters
    ----------
    data : array-like
        Input data.
    method : {'quantile', 'rate'}, default 'quantile'
        Selection method:
        - 'quantile': use specified quantile
        - 'rate': select to achieve target exceedance rate
    quantile : float, default 0.95
        Quantile for threshold (if method='quantile').
    n_per_year : float, optional
        Target number of exceedances per year (if method='rate').
    observations_per_year : float, default 365.25
        Number of observations per year (for rate calculation).

    Returns
    -------
    float
        Selected threshold value.

    Examples
    --------
    >>> # Use 95th percentile
    >>> threshold = select_threshold(data, method='quantile', quantile=0.95)

    >>> # Target ~10 events per year
    >>> threshold = select_threshold(data, method='rate', n_per_year=10)
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if method == "quantile":
        return np.percentile(data, 100 * quantile)

    elif method == "rate":
        if n_per_year is None:
            raise ValueError("n_per_year required for rate method")

        n_total = len(data)
        n_years = n_total / observations_per_year
        target_exceedances = n_per_year * n_years

        # Quantile that gives target number of exceedances
        target_quantile = 1 - target_exceedances / n_total
        target_quantile = np.clip(target_quantile, 0.5, 0.999)

        return np.percentile(data, 100 * target_quantile)

    else:
        raise ValueError(f"Unknown method: {method}")
