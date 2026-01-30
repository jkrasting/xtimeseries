"""
Non-stationary extreme value analysis.

This module provides functions for fitting GEV distributions with
time-varying parameters, which is essential for climate data where
trends due to anthropogenic forcing make stationarity assumptions invalid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats
from typing import Literal
import warnings


def fit_nonstationary_gev(
    data: ArrayLike,
    covariate: ArrayLike,
    trend_in: Literal["loc", "scale", "both"] = "loc",
    method: str = "L-BFGS-B",
) -> dict:
    """
    Fit GEV with time-varying parameters.

    Parameters
    ----------
    data : array-like
        Block maxima values.
    covariate : array-like
        Covariate for trend (e.g., year, global mean temperature).
        Should be same length as data.
    trend_in : {'loc', 'scale', 'both'}, default 'loc'
        Which parameters have trend:
        - 'loc': mu(t) = mu_0 + mu_1 * covariate
        - 'scale': log(sigma(t)) = sigma_0 + sigma_1 * covariate
        - 'both': both location and scale vary

    Returns
    -------
    dict
        Dictionary containing:
        - 'loc0': location intercept
        - 'loc1': location slope (0 if no trend)
        - 'scale0': log-scale intercept
        - 'scale1': log-scale slope (0 if no trend)
        - 'shape': shape parameter (constant)
        - 'nllh': negative log-likelihood
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion
        - 'covariate_mean': mean of covariate (for centering)
        - 'covariate_std': std of covariate (for scaling)

    Notes
    -----
    The model uses:

    .. math::

        \\mu(t) = \\mu_0 + \\mu_1 \\cdot t

    .. math::

        \\sigma(t) = \\exp(\\sigma_0 + \\sigma_1 \\cdot t)

    The log-linear model for scale ensures sigma > 0.

    The shape parameter is typically held constant because it is
    difficult to estimate trends in shape with realistic sample sizes.

    Examples
    --------
    >>> import numpy as np
    >>> from xtimeseries import fit_nonstationary_gev
    >>> years = np.arange(1950, 2020)
    >>> # Simulate data with trend
    >>> np.random.seed(42)
    >>> true_loc = 30 + 0.05 * (years - 1950)
    >>> data = np.array([np.random.gumbel(loc=loc, scale=5) for loc in true_loc])
    >>> result = fit_nonstationary_gev(data, years, trend_in='loc')
    >>> print(f"Location trend: {result['loc1']:.4f} per year")

    See Also
    --------
    likelihood_ratio_test : Compare stationary vs non-stationary models.
    nonstationary_return_level : Return levels for non-stationary GEV.
    """
    data = np.asarray(data)
    covariate = np.asarray(covariate)

    if len(data) != len(covariate):
        raise ValueError("data and covariate must have same length")

    # Remove NaN
    valid = ~(np.isnan(data) | np.isnan(covariate))
    data = data[valid]
    covariate = covariate[valid]
    n = len(data)

    if n < 10:
        raise ValueError("Need at least 10 data points")

    # Standardize covariate for numerical stability
    cov_mean = np.mean(covariate)
    cov_std = np.std(covariate)
    if cov_std < 1e-10:
        cov_std = 1.0
    cov_scaled = (covariate - cov_mean) / cov_std

    # Get initial estimates from stationary fit
    c_init, loc_init, scale_init = stats.genextreme.fit(data)
    shape_init = -c_init

    def neg_log_likelihood(params, trend_type):
        """Compute negative log-likelihood for non-stationary GEV."""
        if trend_type == "loc":
            loc0, loc1, log_scale0, shape = params
            log_scale1 = 0.0
        elif trend_type == "scale":
            loc0, log_scale0, log_scale1, shape = params
            loc1 = 0.0
        else:  # both
            loc0, loc1, log_scale0, log_scale1, shape = params

        # Time-varying parameters
        loc_t = loc0 + loc1 * cov_scaled
        scale_t = np.exp(log_scale0 + log_scale1 * cov_scaled)

        # Ensure valid parameters
        if np.any(scale_t <= 0):
            return 1e10

        # Compute log-likelihood
        # Using scipy's genextreme with c = -shape
        c = -shape
        try:
            log_pdf = stats.genextreme.logpdf(data, c=c, loc=loc_t, scale=scale_t)
            if np.any(~np.isfinite(log_pdf)):
                return 1e10
            return -np.sum(log_pdf)
        except Exception:
            return 1e10

    # Set up optimization
    if trend_in == "loc":
        x0 = [loc_init, 0.0, np.log(scale_init), shape_init]
        bounds = [
            (None, None),  # loc0
            (None, None),  # loc1
            (-5, 5),       # log_scale0
            (-1, 1),       # shape
        ]
        n_params = 4
    elif trend_in == "scale":
        x0 = [loc_init, np.log(scale_init), 0.0, shape_init]
        bounds = [
            (None, None),  # loc0
            (-5, 5),       # log_scale0
            (-2, 2),       # log_scale1
            (-1, 1),       # shape
        ]
        n_params = 4
    else:  # both
        x0 = [loc_init, 0.0, np.log(scale_init), 0.0, shape_init]
        bounds = [
            (None, None),  # loc0
            (None, None),  # loc1
            (-5, 5),       # log_scale0
            (-2, 2),       # log_scale1
            (-1, 1),       # shape
        ]
        n_params = 5

    # Optimize
    result = optimize.minimize(
        neg_log_likelihood,
        x0,
        args=(trend_in,),
        method=method,
        bounds=bounds,
        options={"maxiter": 1000},
    )

    if not result.success:
        warnings.warn(f"Optimization may not have converged: {result.message}")

    # Extract parameters
    if trend_in == "loc":
        loc0, loc1, log_scale0, shape = result.x
        log_scale1 = 0.0
    elif trend_in == "scale":
        loc0, log_scale0, log_scale1, shape = result.x
        loc1 = 0.0
    else:
        loc0, loc1, log_scale0, log_scale1, shape = result.x

    # Convert slope back to original covariate scale
    loc1_original = loc1 / cov_std
    log_scale1_original = log_scale1 / cov_std

    # Model selection criteria
    nllh = result.fun
    aic = 2 * n_params + 2 * nllh
    bic = n_params * np.log(n) + 2 * nllh

    return {
        "loc0": loc0 - loc1 * cov_mean / cov_std,  # Adjust intercept
        "loc1": loc1_original,
        "scale0": log_scale0 - log_scale1 * cov_mean / cov_std,
        "scale1": log_scale1_original,
        "shape": shape,
        "nllh": nllh,
        "aic": aic,
        "bic": bic,
        "n_params": n_params,
        "n_obs": n,
        "trend_in": trend_in,
        "covariate_mean": cov_mean,
        "covariate_std": cov_std,
    }


def nonstationary_return_level(
    return_period: ArrayLike,
    params: dict,
    covariate_value: float,
) -> np.ndarray:
    """
    Calculate return level for non-stationary GEV at a specific covariate value.

    Parameters
    ----------
    return_period : float or array-like
        Return period in years.
    params : dict
        Parameters from fit_nonstationary_gev.
    covariate_value : float
        Value of covariate at which to evaluate return level.

    Returns
    -------
    float or ndarray
        Return level value(s).

    Examples
    --------
    >>> # Get 100-year return level for year 2020
    >>> rl_2020 = nonstationary_return_level(100, params, 2020)
    >>> # Get 100-year return level for year 2050
    >>> rl_2050 = nonstationary_return_level(100, params, 2050)

    See Also
    --------
    effective_return_level : Compare return levels across time periods.
    """
    return_period = np.asarray(return_period)

    # Time-varying parameters at covariate_value
    loc_t = params["loc0"] + params["loc1"] * covariate_value
    scale_t = np.exp(params["scale0"] + params["scale1"] * covariate_value)
    shape = params["shape"]

    # Non-exceedance probability
    p = 1 - 1 / return_period

    return stats.genextreme.ppf(p, c=-shape, loc=loc_t, scale=scale_t)


def likelihood_ratio_test(
    data: ArrayLike,
    covariate: ArrayLike,
    trend_in: Literal["loc", "scale", "both"] = "loc",
) -> dict:
    """
    Test stationary vs non-stationary model using likelihood ratio test.

    Parameters
    ----------
    data : array-like
        Block maxima values.
    covariate : array-like
        Covariate for trend.
    trend_in : {'loc', 'scale', 'both'}, default 'loc'
        Which parameters have trend in alternative model.

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': likelihood ratio test statistic
        - 'df': degrees of freedom
        - 'p_value': p-value from chi-squared distribution
        - 'significant': True if p < 0.05
        - 'stationary_nllh': negative log-likelihood of stationary model
        - 'nonstationary_nllh': negative log-likelihood of non-stationary model
        - 'aic_stationary': AIC for stationary model
        - 'aic_nonstationary': AIC for non-stationary model

    Notes
    -----
    The test statistic is:

    .. math::

        D = 2(\\ell_1 - \\ell_0)

    where :math:`\\ell_1` and :math:`\\ell_0` are the log-likelihoods of the
    non-stationary and stationary models, respectively.

    Under the null hypothesis (stationarity), D follows a chi-squared
    distribution with degrees of freedom equal to the difference in the
    number of parameters.

    Examples
    --------
    >>> import numpy as np
    >>> years = np.arange(1950, 2020)
    >>> data = np.random.gumbel(loc=30, scale=5, size=len(years))
    >>> result = likelihood_ratio_test(data, years)
    >>> print(f"p-value: {result['p_value']:.4f}")
    """
    data = np.asarray(data)
    covariate = np.asarray(covariate)

    # Fit stationary model
    valid = ~(np.isnan(data) | np.isnan(covariate))
    data_clean = data[valid]
    n = len(data_clean)

    c, loc, scale = stats.genextreme.fit(data_clean)
    nllh_stationary = -np.sum(stats.genextreme.logpdf(data_clean, c=c, loc=loc, scale=scale))
    n_params_stationary = 3

    # Fit non-stationary model
    ns_result = fit_nonstationary_gev(data, covariate, trend_in=trend_in)
    nllh_nonstationary = ns_result["nllh"]
    n_params_nonstationary = ns_result["n_params"]

    # Likelihood ratio test
    df = n_params_nonstationary - n_params_stationary
    statistic = 2 * (nllh_stationary - nllh_nonstationary)

    # Ensure statistic is non-negative
    statistic = max(0, statistic)

    p_value = 1 - stats.chi2.cdf(statistic, df)

    # AIC
    aic_stationary = 2 * n_params_stationary + 2 * nllh_stationary
    aic_nonstationary = ns_result["aic"]

    return {
        "statistic": statistic,
        "df": df,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "stationary_nllh": nllh_stationary,
        "nonstationary_nllh": nllh_nonstationary,
        "aic_stationary": aic_stationary,
        "aic_nonstationary": aic_nonstationary,
        "preferred_model": "nonstationary" if aic_nonstationary < aic_stationary else "stationary",
    }


def effective_return_level(
    params: dict,
    reference_value: float,
    future_value: float,
    return_periods: ArrayLike | None = None,
) -> dict:
    """
    Calculate how return levels change between two covariate values.

    This is useful for understanding how the "100-year event" of the past
    compares to future conditions.

    Parameters
    ----------
    params : dict
        Parameters from fit_nonstationary_gev.
    reference_value : float
        Reference covariate value (e.g., historical year).
    future_value : float
        Future covariate value (e.g., projection year).
    return_periods : array-like, optional
        Return periods to evaluate. Default is [2, 5, 10, 20, 50, 100].

    Returns
    -------
    dict
        Dictionary containing:
        - 'return_periods': evaluated return periods
        - 'reference_levels': return levels at reference
        - 'future_levels': return levels at future
        - 'change': absolute change in return levels
        - 'change_percent': percent change
        - 'effective_period': effective return period of reference level at future

    Examples
    --------
    >>> # How does the historical 100-year event change?
    >>> result = effective_return_level(params, reference_value=1990, future_value=2050)
    >>> # A historical 100-year event might become a 20-year event
    >>> print(f"100-year event becomes: {result['effective_period'][5]:.1f}-year event")
    """
    if return_periods is None:
        return_periods = np.array([2, 5, 10, 20, 50, 100])
    else:
        return_periods = np.asarray(return_periods)

    # Return levels at reference and future
    ref_levels = nonstationary_return_level(return_periods, params, reference_value)
    future_levels = nonstationary_return_level(return_periods, params, future_value)

    # Change
    change = future_levels - ref_levels
    change_pct = 100 * change / np.abs(ref_levels)

    # Effective return period: what is the return period of the reference level at future?
    # This requires inverting the return level calculation
    shape = params["shape"]
    loc_future = params["loc0"] + params["loc1"] * future_value
    scale_future = np.exp(params["scale0"] + params["scale1"] * future_value)

    # CDF of reference levels under future distribution
    p_future = stats.genextreme.cdf(ref_levels, c=-shape, loc=loc_future, scale=scale_future)
    p_future = np.clip(p_future, 0, 1 - 1e-10)
    effective_periods = 1 / (1 - p_future)

    return {
        "return_periods": return_periods,
        "reference_levels": ref_levels,
        "future_levels": future_levels,
        "change": change,
        "change_percent": change_pct,
        "effective_period": effective_periods,
        "reference_value": reference_value,
        "future_value": future_value,
    }
