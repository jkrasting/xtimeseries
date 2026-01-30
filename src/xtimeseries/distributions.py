"""
Distribution fitting functions for extreme value analysis.

This module provides wrappers around scipy.stats distributions with
the correct sign conventions for climate science applications.

IMPORTANT: scipy.stats.genextreme uses c = -xi, where xi is the standard
climate convention shape parameter. This module handles the conversion
automatically.

Shape parameter conventions:
    - xi > 0: Fréchet (heavy tail, unbounded above) - e.g., precipitation
    - xi = 0: Gumbel (light exponential tail)
    - xi < 0: Weibull (bounded upper tail) - e.g., temperature maxima
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from typing import Literal


def fit_gev(
    data: ArrayLike,
    method: Literal["mle", "mom"] = "mle",
) -> dict[str, float]:
    """
    Fit a Generalized Extreme Value (GEV) distribution to data.

    This function wraps scipy.stats.genextreme.fit() and converts the
    shape parameter to the standard climate convention.

    Parameters
    ----------
    data : array-like
        Sample data (e.g., annual maxima). NaN values are removed.
    method : {'mle', 'mom'}, default 'mle'
        Fitting method. 'mle' for maximum likelihood estimation,
        'mom' for method of moments.

    Returns
    -------
    dict
        Dictionary with keys 'loc', 'scale', 'shape' containing the
        fitted GEV parameters. Shape uses climate convention (xi).

    Notes
    -----
    The GEV cumulative distribution function is:

    .. math::

        F(x) = \\exp\\left\\{-\\left[1 + \\xi\\frac{x-\\mu}{\\sigma}\\right]^{-1/\\xi}\\right\\}

    for :math:`\\xi \\neq 0`, and

    .. math::

        F(x) = \\exp\\left\\{-\\exp\\left[-\\frac{x-\\mu}{\\sigma}\\right]\\right\\}

    for :math:`\\xi = 0` (Gumbel case).

    Examples
    --------
    >>> import numpy as np
    >>> from xtimeseries import fit_gev
    >>> np.random.seed(42)
    >>> data = np.random.gumbel(loc=30, scale=5, size=100)
    >>> params = fit_gev(data)
    >>> print(f"Location: {params['loc']:.2f}")
    Location: 29.47

    See Also
    --------
    fit_gpd : Fit Generalized Pareto Distribution.
    gev_ppf : GEV quantile function.
    """
    # Remove NaN values
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if len(data) < 3:
        raise ValueError("Need at least 3 non-NaN values to fit GEV")

    if method == "mle":
        # scipy uses c = -xi convention
        c, loc, scale = stats.genextreme.fit(data)
        shape = -c  # Convert to climate convention
    elif method == "mom":
        # Method of moments fitting
        shape, loc, scale = _fit_gev_mom(data)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mle' or 'mom'.")

    return {"loc": loc, "scale": scale, "shape": shape}


def _fit_gev_mom(data: ArrayLike) -> tuple[float, float, float]:
    """
    Fit GEV using method of moments (probability weighted moments).

    This is a simplified L-moments based approach.
    """
    data = np.sort(data)
    n = len(data)

    # Probability weighted moments
    b0 = np.mean(data)
    b1 = np.sum(np.arange(1, n) * data[1:]) / (n * (n - 1))
    b2 = np.sum(np.arange(1, n - 1) * np.arange(2, n) * data[2:]) / (n * (n - 1) * (n - 2))

    # L-moments
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0

    # L-moment ratios
    t3 = l3 / l2  # L-skewness

    # Approximate shape parameter from L-skewness
    # Using the approximation from Hosking (1990)
    c = 2 / (3 + t3) - np.log(2) / np.log(3)
    shape = 7.8590 * c + 2.9554 * c**2

    # Scale and location from L-moments
    import math
    if abs(shape) > 1e-6:
        g1 = math.gamma(1 + shape)
        scale = l2 * shape / (g1 * (1 - 2**(-shape)))
        loc = l1 - scale * (g1 - 1) / shape
    else:
        # Gumbel case
        scale = l2 / np.log(2)
        loc = l1 - 0.5772 * scale

    return shape, loc, scale


def fit_gpd(
    data: ArrayLike,
    threshold: float = 0.0,
    method: Literal["mle", "mom"] = "mle",
) -> dict[str, float]:
    """
    Fit a Generalized Pareto Distribution (GPD) to exceedances.

    Parameters
    ----------
    data : array-like
        Exceedance data (values above threshold). If data contains values
        below threshold, they are automatically converted to exceedances.
    threshold : float, default 0.0
        Threshold value. If non-zero, data is treated as raw values and
        exceedances are computed as (data - threshold).
    method : {'mle', 'mom'}, default 'mle'
        Fitting method.

    Returns
    -------
    dict
        Dictionary with keys 'scale', 'shape', 'threshold' containing
        the fitted GPD parameters.

    Notes
    -----
    The GPD cumulative distribution function is:

    .. math::

        F(x) = 1 - \\left[1 + \\xi\\frac{x}{\\sigma}\\right]^{-1/\\xi}

    for :math:`\\xi \\neq 0`, and

    .. math::

        F(x) = 1 - \\exp\\left(-\\frac{x}{\\sigma}\\right)

    for :math:`\\xi = 0` (exponential case).

    Examples
    --------
    >>> import numpy as np
    >>> from xtimeseries import fit_gpd
    >>> np.random.seed(42)
    >>> exceedances = np.random.exponential(scale=5, size=100)
    >>> params = fit_gpd(exceedances)
    >>> print(f"Scale: {params['scale']:.2f}")

    See Also
    --------
    fit_gev : Fit Generalized Extreme Value distribution.
    peaks_over_threshold : Extract exceedances from time series.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # Convert to exceedances if threshold is specified
    if threshold != 0.0:
        exceedances = data[data > threshold] - threshold
    else:
        exceedances = data[data > 0]  # GPD is defined for positive values

    if len(exceedances) < 3:
        raise ValueError("Need at least 3 exceedances to fit GPD")

    if method == "mle":
        # scipy genpareto uses c = xi (same convention as climate)
        shape, _, scale = stats.genpareto.fit(exceedances, floc=0)
    elif method == "mom":
        # Method of moments
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances, ddof=1)

        # From mean and variance, solve for shape and scale
        # mean = scale / (1 - shape) for shape < 1
        # var = scale^2 / ((1-shape)^2 * (1-2*shape)) for shape < 0.5
        if var_exc > 0:
            ratio = mean_exc**2 / var_exc
            shape = 0.5 * (1 - ratio)
            scale = mean_exc * (1 - shape)
        else:
            shape = 0.0
            scale = mean_exc
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mle' or 'mom'.")

    return {"scale": scale, "shape": shape, "threshold": threshold}


def gev_cdf(
    x: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
) -> np.ndarray:
    """
    Compute the GEV cumulative distribution function.

    Parameters
    ----------
    x : array-like
        Values at which to evaluate the CDF.
    loc : float
        Location parameter (mu).
    scale : float
        Scale parameter (sigma). Must be positive.
    shape : float
        Shape parameter (xi). Uses climate convention.

    Returns
    -------
    ndarray
        CDF values in [0, 1].

    Examples
    --------
    >>> from xtimeseries import gev_cdf
    >>> gev_cdf(35, loc=30, scale=5, shape=0.1)
    0.692...
    """
    x = np.asarray(x)
    # Convert climate convention (shape) to scipy convention (c = -shape)
    return stats.genextreme.cdf(x, c=-shape, loc=loc, scale=scale)


def gev_pdf(
    x: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
) -> np.ndarray:
    """
    Compute the GEV probability density function.

    Parameters
    ----------
    x : array-like
        Values at which to evaluate the PDF.
    loc : float
        Location parameter (mu).
    scale : float
        Scale parameter (sigma). Must be positive.
    shape : float
        Shape parameter (xi). Uses climate convention.

    Returns
    -------
    ndarray
        PDF values.
    """
    x = np.asarray(x)
    return stats.genextreme.pdf(x, c=-shape, loc=loc, scale=scale)


def gev_ppf(
    p: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
) -> np.ndarray:
    """
    Compute the GEV percent point function (quantile function).

    Parameters
    ----------
    p : array-like
        Probability values in (0, 1).
    loc : float
        Location parameter (mu).
    scale : float
        Scale parameter (sigma). Must be positive.
    shape : float
        Shape parameter (xi). Uses climate convention.

    Returns
    -------
    ndarray
        Quantile values.

    Notes
    -----
    The quantile function is the inverse of the CDF:

    .. math::

        x_p = \\mu + \\frac{\\sigma}{\\xi}\\left[(-\\ln p)^{-\\xi} - 1\\right]

    for :math:`\\xi \\neq 0`.

    Examples
    --------
    >>> from xtimeseries import gev_ppf
    >>> # 99th percentile (approximately 100-year return level)
    >>> gev_ppf(0.99, loc=30, scale=5, shape=0.1)
    54.89...
    """
    p = np.asarray(p)
    return stats.genextreme.ppf(p, c=-shape, loc=loc, scale=scale)


def gpd_cdf(
    x: ArrayLike,
    scale: float,
    shape: float,
) -> np.ndarray:
    """
    Compute the GPD cumulative distribution function.

    Parameters
    ----------
    x : array-like
        Values at which to evaluate the CDF (exceedances above threshold).
    scale : float
        Scale parameter (sigma). Must be positive.
    shape : float
        Shape parameter (xi).

    Returns
    -------
    ndarray
        CDF values in [0, 1].
    """
    x = np.asarray(x)
    return stats.genpareto.cdf(x, c=shape, loc=0, scale=scale)


def gpd_pdf(
    x: ArrayLike,
    scale: float,
    shape: float,
) -> np.ndarray:
    """
    Compute the GPD probability density function.

    Parameters
    ----------
    x : array-like
        Values at which to evaluate the PDF.
    scale : float
        Scale parameter (sigma). Must be positive.
    shape : float
        Shape parameter (xi).

    Returns
    -------
    ndarray
        PDF values.
    """
    x = np.asarray(x)
    return stats.genpareto.pdf(x, c=shape, loc=0, scale=scale)


def gpd_ppf(
    p: ArrayLike,
    scale: float,
    shape: float,
) -> np.ndarray:
    """
    Compute the GPD percent point function (quantile function).

    Parameters
    ----------
    p : array-like
        Probability values in (0, 1).
    scale : float
        Scale parameter (sigma). Must be positive.
    shape : float
        Shape parameter (xi).

    Returns
    -------
    ndarray
        Quantile values (exceedances above threshold).
    """
    p = np.asarray(p)
    return stats.genpareto.ppf(p, c=shape, loc=0, scale=scale)
