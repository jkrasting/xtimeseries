"""
Return period and return level calculations.

This module provides functions for calculating return levels (the value
expected to be exceeded once every T years) and return periods (the average
time between exceedances of a given value).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def return_level(
    return_period: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
) -> np.ndarray:
    """
    Calculate the GEV return level for a given return period.

    The return level is the value expected to be exceeded on average
    once every T years, where T is the return period.

    Parameters
    ----------
    return_period : float or array-like
        Return period in years (e.g., 100 for "100-year event").
        Must be > 1.
    loc : float
        GEV location parameter (mu).
    scale : float
        GEV scale parameter (sigma). Must be positive.
    shape : float
        GEV shape parameter (xi). Uses climate convention.

    Returns
    -------
    float or ndarray
        Return level value(s).

    Notes
    -----
    The return level is calculated using:

    .. math::

        x_T = \\mu + \\frac{\\sigma}{\\xi}\\left[y_p^{-\\xi} - 1\\right]

    where :math:`y_p = -\\ln(1 - 1/T)`.

    For the Gumbel case (:math:`\\xi = 0`):

    .. math::

        x_T = \\mu - \\sigma \\ln(-\\ln(1 - 1/T))

    Examples
    --------
    >>> from xtimeseries import return_level
    >>> # 100-year return level
    >>> rl = return_level(100, loc=30, scale=5, shape=0.1)
    >>> print(f"{rl:.2f}")
    54.89

    >>> # Multiple return periods
    >>> rls = return_level([10, 50, 100], loc=30, scale=5, shape=0.1)

    See Also
    --------
    return_period : Calculate return period for a given value.
    return_level_gpd : Return level for GPD (POT analysis).
    """
    return_period = np.asarray(return_period)

    if np.any(return_period <= 1):
        raise ValueError("Return period must be > 1")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive")

    # Non-exceedance probability
    p = 1 - 1 / return_period

    # Use scipy with converted shape parameter
    return stats.genextreme.ppf(p, c=-shape, loc=loc, scale=scale)


def return_period(
    value: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
) -> np.ndarray:
    """
    Calculate the return period for a given value.

    The return period is the average time (in years) between occurrences
    of events at least as extreme as the given value.

    Parameters
    ----------
    value : float or array-like
        The extreme value(s) to evaluate.
    loc : float
        GEV location parameter (mu).
    scale : float
        GEV scale parameter (sigma). Must be positive.
    shape : float
        GEV shape parameter (xi). Uses climate convention.

    Returns
    -------
    float or ndarray
        Return period in years.

    Notes
    -----
    The return period is calculated as:

    .. math::

        T = \\frac{1}{1 - F(x)}

    where :math:`F(x)` is the GEV CDF.

    Examples
    --------
    >>> from xtimeseries import return_period
    >>> # What is the return period of an event with value 50?
    >>> T = return_period(50, loc=30, scale=5, shape=0.1)
    >>> print(f"{T:.1f} years")
    54.5 years

    See Also
    --------
    return_level : Calculate return level for a given period.
    """
    value = np.asarray(value)

    if scale <= 0:
        raise ValueError("Scale parameter must be positive")

    # CDF value
    p = stats.genextreme.cdf(value, c=-shape, loc=loc, scale=scale)

    # Avoid division by zero
    p = np.clip(p, 0, 1 - 1e-10)

    return 1 / (1 - p)


def return_level_gpd(
    return_period: ArrayLike,
    threshold: float,
    scale: float,
    shape: float,
    rate: float,
) -> np.ndarray:
    """
    Calculate GPD return level for peaks-over-threshold analysis.

    Parameters
    ----------
    return_period : float or array-like
        Return period in years.
    threshold : float
        Threshold value (u).
    scale : float
        GPD scale parameter (sigma).
    shape : float
        GPD shape parameter (xi).
    rate : float
        Exceedance rate (average number of exceedances per year).
        Also called lambda or zeta.

    Returns
    -------
    float or ndarray
        Return level value(s).

    Notes
    -----
    The GPD return level is:

    .. math::

        x_T = u + \\frac{\\sigma}{\\xi}\\left[(T \\cdot \\lambda)^{\\xi} - 1\\right]

    for :math:`\\xi \\neq 0`, and

    .. math::

        x_T = u + \\sigma \\ln(T \\cdot \\lambda)

    for :math:`\\xi = 0`.

    Examples
    --------
    >>> from xtimeseries import return_level_gpd
    >>> # 100-year return level with 10 exceedances per year on average
    >>> rl = return_level_gpd(100, threshold=20, scale=5, shape=0.2, rate=10)

    See Also
    --------
    return_level : GEV return level for block maxima.
    peaks_over_threshold : Extract exceedances.
    """
    return_period = np.asarray(return_period)

    if np.any(return_period <= 0):
        raise ValueError("Return period must be positive")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive")

    if rate <= 0:
        raise ValueError("Exceedance rate must be positive")

    # m = expected number of exceedances in return_period years
    m = return_period * rate

    if abs(shape) < 1e-10:
        # Exponential case
        exceedance = scale * np.log(m)
    else:
        exceedance = (scale / shape) * (m**shape - 1)

    return threshold + exceedance


def return_period_gpd(
    value: ArrayLike,
    threshold: float,
    scale: float,
    shape: float,
    rate: float,
) -> np.ndarray:
    """
    Calculate return period for a given value using GPD.

    Parameters
    ----------
    value : float or array-like
        The value(s) to evaluate (must be > threshold).
    threshold : float
        Threshold value (u).
    scale : float
        GPD scale parameter (sigma).
    shape : float
        GPD shape parameter (xi).
    rate : float
        Exceedance rate (events per year).

    Returns
    -------
    float or ndarray
        Return period in years.

    Examples
    --------
    >>> from xtimeseries import return_period_gpd
    >>> T = return_period_gpd(40, threshold=20, scale=5, shape=0.2, rate=10)
    """
    value = np.asarray(value)
    exceedance = value - threshold

    if np.any(exceedance <= 0):
        raise ValueError("Values must be above threshold")

    if scale <= 0:
        raise ValueError("Scale parameter must be positive")

    # Survival probability for exceedance
    survival = 1 - stats.genpareto.cdf(exceedance, c=shape, loc=0, scale=scale)

    # Return period in years
    # P(X > x in a year) = rate * P(Y > y) where Y is exceedance
    annual_prob = rate * survival

    return 1 / annual_prob


def return_level_with_ci(
    return_period: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
    loc_se: float | None = None,
    scale_se: float | None = None,
    shape_se: float | None = None,
    cov: ArrayLike | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Calculate return level with confidence interval using delta method.

    Parameters
    ----------
    return_period : float or array-like
        Return period in years.
    loc, scale, shape : float
        GEV parameters.
    loc_se, scale_se, shape_se : float, optional
        Standard errors of parameters. Required if cov not provided.
    cov : array-like, optional
        Covariance matrix of parameters (3x3). If provided, overrides
        individual standard errors.
    alpha : float, default 0.05
        Significance level for confidence interval.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'return_level': point estimate
        - 'lower': lower CI bound
        - 'upper': upper CI bound
        - 'se': standard error of return level

    Notes
    -----
    Uses the delta method to propagate parameter uncertainty to the
    return level. For more accurate confidence intervals with small
    samples, use bootstrap_ci instead.

    See Also
    --------
    bootstrap_ci : Bootstrap confidence intervals.
    """
    from scipy.stats import norm

    return_period = np.asarray(return_period)
    rl = return_level(return_period, loc, scale, shape)

    if cov is not None:
        cov = np.asarray(cov)
        if cov.shape != (3, 3):
            raise ValueError("Covariance matrix must be 3x3")
    elif loc_se is not None and scale_se is not None and shape_se is not None:
        # Construct diagonal covariance matrix (assumes independence)
        cov = np.diag([loc_se**2, scale_se**2, shape_se**2])
    else:
        raise ValueError("Provide either cov or all three standard errors")

    # Compute gradient of return level with respect to parameters
    p = 1 - 1 / return_period
    yp = -np.log(p)

    if abs(shape) < 1e-10:
        # Gumbel case
        drl_dloc = 1.0
        drl_dscale = -np.log(yp)
        drl_dshape = 0.0
    else:
        # General case
        drl_dloc = 1.0
        drl_dscale = (yp**(-shape) - 1) / shape
        drl_dshape = (
            -scale / shape**2 * (yp**(-shape) - 1)
            + scale / shape * yp**(-shape) * np.log(yp)
        )

    # Gradient vector
    grad = np.array([drl_dloc, drl_dscale, drl_dshape])

    # Variance via delta method
    var_rl = grad @ cov @ grad
    se_rl = np.sqrt(var_rl)

    # Confidence interval
    z = norm.ppf(1 - alpha / 2)
    lower = rl - z * se_rl
    upper = rl + z * se_rl

    return {
        "return_level": rl,
        "lower": lower,
        "upper": upper,
        "se": se_rl,
    }
