"""
Confidence interval estimation for extreme value analysis.

This module provides bootstrap methods for computing confidence intervals
on return levels and other extreme value statistics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from typing import Callable


def bootstrap_ci(
    data: ArrayLike,
    return_periods: ArrayLike,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    method: str = "percentile",
    random_state: int | np.random.Generator | None = None,
) -> dict:
    """
    Compute bootstrap confidence intervals for GEV return levels.

    Parameters
    ----------
    data : array-like
        Block maxima data (e.g., annual maxima).
    return_periods : array-like
        Return periods (in years) for which to compute CIs.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    alpha : float, default 0.05
        Significance level. CI will be (alpha/2, 1-alpha/2) percentiles.
    method : {'percentile', 'basic', 'bca'}, default 'percentile'
        Bootstrap CI method:
        - 'percentile': simple percentile method
        - 'basic': basic bootstrap (2*estimate - percentiles)
        - 'bca': bias-corrected and accelerated (most accurate)
    random_state : int or Generator, optional
        Random state for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'return_periods': input return periods
        - 'return_levels': point estimates
        - 'lower': lower CI bounds
        - 'upper': upper CI bounds
        - 'se': standard errors (std of bootstrap samples)

    Examples
    --------
    >>> import numpy as np
    >>> from xtimeseries import bootstrap_ci
    >>> np.random.seed(42)
    >>> data = np.random.gumbel(loc=30, scale=5, size=50)
    >>> result = bootstrap_ci(data, [10, 50, 100], n_bootstrap=500)
    >>> print(f"100-year level: {result['return_levels'][2]:.2f}")
    >>> print(f"95% CI: [{result['lower'][2]:.2f}, {result['upper'][2]:.2f}]")

    See Also
    --------
    bootstrap_return_levels : Get full bootstrap distribution.
    return_level_with_ci : Delta method confidence intervals.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return_periods = np.asarray(return_periods)
    n = len(data)

    if n < 10:
        raise ValueError("Need at least 10 data points for reliable bootstrap")

    # Set up random generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Fit original data
    c, loc, scale = stats.genextreme.fit(data)
    shape = -c  # Convert to climate convention

    # Point estimates
    p = 1 - 1 / return_periods
    point_estimates = stats.genextreme.ppf(p, c=c, loc=loc, scale=scale)

    # Bootstrap resampling
    boot_return_levels = np.zeros((n_bootstrap, len(return_periods)))

    for i in range(n_bootstrap):
        # Resample with replacement
        boot_sample = rng.choice(data, size=n, replace=True)

        # Fit GEV to bootstrap sample
        try:
            c_boot, loc_boot, scale_boot = stats.genextreme.fit(boot_sample)
            boot_return_levels[i, :] = stats.genextreme.ppf(
                p, c=c_boot, loc=loc_boot, scale=scale_boot
            )
        except Exception:
            # If fitting fails, use NaN
            boot_return_levels[i, :] = np.nan

    # Remove failed fits
    valid = ~np.any(np.isnan(boot_return_levels), axis=1)
    boot_return_levels = boot_return_levels[valid, :]

    if len(boot_return_levels) < n_bootstrap * 0.9:
        import warnings
        warnings.warn(
            f"Many bootstrap fits failed ({n_bootstrap - len(boot_return_levels)}/{n_bootstrap})"
        )

    # Compute confidence intervals
    if method == "percentile":
        lower = np.percentile(boot_return_levels, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_return_levels, 100 * (1 - alpha / 2), axis=0)
    elif method == "basic":
        lower_pct = np.percentile(boot_return_levels, 100 * (1 - alpha / 2), axis=0)
        upper_pct = np.percentile(boot_return_levels, 100 * alpha / 2, axis=0)
        lower = 2 * point_estimates - lower_pct
        upper = 2 * point_estimates - upper_pct
    elif method == "bca":
        lower, upper = _bca_interval(
            data, boot_return_levels, point_estimates, alpha, rng
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Standard errors
    se = np.std(boot_return_levels, axis=0)

    return {
        "return_periods": return_periods,
        "return_levels": point_estimates,
        "lower": lower,
        "upper": upper,
        "se": se,
        "n_bootstrap": len(boot_return_levels),
    }


def _bca_interval(
    data: np.ndarray,
    boot_samples: np.ndarray,
    theta_hat: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute BCa (bias-corrected and accelerated) confidence interval.

    Parameters
    ----------
    data : ndarray
        Original data.
    boot_samples : ndarray
        Bootstrap samples of statistic (n_boot x n_params).
    theta_hat : ndarray
        Point estimate of statistic.
    alpha : float
        Significance level.
    rng : Generator
        Random number generator.

    Returns
    -------
    lower, upper : ndarray
        BCa confidence interval bounds.
    """
    from scipy.stats import norm

    n_boot, n_params = boot_samples.shape
    n = len(data)

    lower = np.zeros(n_params)
    upper = np.zeros(n_params)

    for j in range(n_params):
        boot_j = boot_samples[:, j]
        theta_j = theta_hat[j]

        # Bias correction factor
        z0 = norm.ppf(np.mean(boot_j < theta_j))

        # Acceleration factor via jackknife
        theta_jack = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            try:
                c_j, loc_j, scale_j = stats.genextreme.fit(jack_sample)
                p = 1 - 1 / (j + 2)  # Approximate return period
                theta_jack[i] = stats.genextreme.ppf(p, c=c_j, loc=loc_j, scale=scale_j)
            except Exception:
                theta_jack[i] = np.nan

        theta_jack = theta_jack[~np.isnan(theta_jack)]
        theta_mean = np.mean(theta_jack)
        a = np.sum((theta_mean - theta_jack) ** 3) / (
            6 * np.sum((theta_mean - theta_jack) ** 2) ** 1.5 + 1e-10
        )

        # Adjusted percentiles
        z_alpha_lower = norm.ppf(alpha / 2)
        z_alpha_upper = norm.ppf(1 - alpha / 2)

        alpha1 = norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        alpha2 = norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        # Clip to valid range
        alpha1 = np.clip(alpha1, 0.001, 0.999)
        alpha2 = np.clip(alpha2, 0.001, 0.999)

        lower[j] = np.percentile(boot_j, 100 * alpha1)
        upper[j] = np.percentile(boot_j, 100 * alpha2)

    return lower, upper


def bootstrap_return_levels(
    data: ArrayLike,
    return_periods: ArrayLike,
    n_bootstrap: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Get full bootstrap distribution of return levels.

    Parameters
    ----------
    data : array-like
        Block maxima data.
    return_periods : array-like
        Return periods to compute.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    random_state : int or Generator, optional
        Random state for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap return levels, shape (n_bootstrap, len(return_periods)).

    Examples
    --------
    >>> import numpy as np
    >>> from xtimeseries import bootstrap_return_levels
    >>> data = np.random.gumbel(loc=30, scale=5, size=50)
    >>> boot_rls = bootstrap_return_levels(data, [10, 50, 100])
    >>> # Custom analysis of bootstrap distribution
    >>> print(f"Median 100-year level: {np.median(boot_rls[:, 2]):.2f}")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return_periods = np.asarray(return_periods)
    n = len(data)

    # Set up random generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    p = 1 - 1 / return_periods

    boot_return_levels = np.zeros((n_bootstrap, len(return_periods)))

    for i in range(n_bootstrap):
        boot_sample = rng.choice(data, size=n, replace=True)
        try:
            c, loc, scale = stats.genextreme.fit(boot_sample)
            boot_return_levels[i, :] = stats.genextreme.ppf(p, c=c, loc=loc, scale=scale)
        except Exception:
            boot_return_levels[i, :] = np.nan

    return boot_return_levels


def bootstrap_parameters(
    data: ArrayLike,
    n_bootstrap: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> dict:
    """
    Bootstrap confidence intervals for GEV parameters.

    Parameters
    ----------
    data : array-like
        Block maxima data.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    random_state : int or Generator, optional
        Random state for reproducibility.

    Returns
    -------
    dict
        Dictionary with parameter estimates and confidence intervals.

    Examples
    --------
    >>> import numpy as np
    >>> from xtimeseries import bootstrap_parameters
    >>> data = np.random.gumbel(loc=30, scale=5, size=50)
    >>> result = bootstrap_parameters(data)
    >>> print(f"Location: {result['loc']:.2f} ({result['loc_ci'][0]:.2f}, {result['loc_ci'][1]:.2f})")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Point estimates
    c, loc, scale = stats.genextreme.fit(data)
    shape = -c

    # Bootstrap
    boot_params = np.zeros((n_bootstrap, 3))

    for i in range(n_bootstrap):
        boot_sample = rng.choice(data, size=n, replace=True)
        try:
            c_b, loc_b, scale_b = stats.genextreme.fit(boot_sample)
            boot_params[i, :] = [loc_b, scale_b, -c_b]
        except Exception:
            boot_params[i, :] = np.nan

    # Remove failed fits
    boot_params = boot_params[~np.any(np.isnan(boot_params), axis=1), :]

    return {
        "loc": loc,
        "scale": scale,
        "shape": shape,
        "loc_ci": (np.percentile(boot_params[:, 0], 2.5), np.percentile(boot_params[:, 0], 97.5)),
        "scale_ci": (np.percentile(boot_params[:, 1], 2.5), np.percentile(boot_params[:, 1], 97.5)),
        "shape_ci": (np.percentile(boot_params[:, 2], 2.5), np.percentile(boot_params[:, 2], 97.5)),
        "loc_se": np.std(boot_params[:, 0]),
        "scale_se": np.std(boot_params[:, 1]),
        "shape_se": np.std(boot_params[:, 2]),
        "cov": np.cov(boot_params.T),
    }
