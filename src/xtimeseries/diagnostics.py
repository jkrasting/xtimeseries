"""
Diagnostic plots for extreme value analysis.

This module provides plotting functions for assessing goodness-of-fit
and visualizing extreme value analysis results.

Note: All plotting functions return figure and axes objects but do not
call plt.show(), allowing users to customize plots before display.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from typing import Any


def probability_plot(
    data: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
    ax: Any = None,
    **kwargs,
) -> tuple:
    """
    Create a GEV probability plot.

    Plots empirical vs theoretical probabilities. Points should lie
    along the 1:1 line if the GEV fit is good.

    Parameters
    ----------
    data : array-like
        Block maxima data.
    loc, scale, shape : float
        GEV parameters.
    ax : matplotlib axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to scatter plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    Examples
    --------
    >>> from xtimeseries import fit_gev, probability_plot
    >>> params = fit_gev(annual_max)
    >>> fig, ax = probability_plot(annual_max, **params)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    data = np.sort(data[~np.isnan(data)])
    n = len(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Empirical probabilities (Weibull plotting position)
    empirical_p = np.arange(1, n + 1) / (n + 1)

    # Theoretical probabilities from fitted GEV
    theoretical_p = stats.genextreme.cdf(data, c=-shape, loc=loc, scale=scale)

    # Plot
    scatter_kwargs = {"alpha": 0.7, "edgecolors": "k", "linewidth": 0.5}
    scatter_kwargs.update(kwargs)
    ax.scatter(theoretical_p, empirical_p, **scatter_kwargs)

    # 1:1 line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="1:1 line")

    ax.set_xlabel("Theoretical probability")
    ax.set_ylabel("Empirical probability")
    ax.set_title("Probability Plot")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()

    return fig, ax


def qq_plot(
    data: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
    ax: Any = None,
    **kwargs,
) -> tuple:
    """
    Create a GEV Q-Q (quantile-quantile) plot.

    Plots empirical vs theoretical quantiles. Points should lie
    along the 1:1 line if the fit is good.

    Parameters
    ----------
    data : array-like
        Block maxima data.
    loc, scale, shape : float
        GEV parameters.
    ax : matplotlib axes, optional
        Axes to plot on.
    **kwargs
        Additional arguments for scatter plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    Examples
    --------
    >>> fig, ax = qq_plot(annual_max, **params)
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    data = np.sort(data[~np.isnan(data)])
    n = len(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Empirical quantiles
    empirical_q = data

    # Theoretical quantiles
    p = np.arange(1, n + 1) / (n + 1)
    theoretical_q = stats.genextreme.ppf(p, c=-shape, loc=loc, scale=scale)

    # Plot
    scatter_kwargs = {"alpha": 0.7, "edgecolors": "k", "linewidth": 0.5}
    scatter_kwargs.update(kwargs)
    ax.scatter(theoretical_q, empirical_q, **scatter_kwargs)

    # 1:1 line
    min_val = min(theoretical_q.min(), empirical_q.min())
    max_val = max(theoretical_q.max(), empirical_q.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1, label="1:1 line")

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("Q-Q Plot")
    ax.legend()

    return fig, ax


def return_level_plot(
    data: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
    return_periods: ArrayLike | None = None,
    ci_lower: ArrayLike | None = None,
    ci_upper: ArrayLike | None = None,
    ax: Any = None,
    log_scale: bool = True,
    **kwargs,
) -> tuple:
    """
    Create a return level plot.

    Shows return levels vs return periods with optional confidence bands.

    Parameters
    ----------
    data : array-like
        Block maxima data.
    loc, scale, shape : float
        GEV parameters.
    return_periods : array-like, optional
        Return periods to plot. Default is 2 to 1000 years.
    ci_lower, ci_upper : array-like, optional
        Confidence interval bounds (same length as return_periods).
    ax : matplotlib axes, optional
        Axes to plot on.
    log_scale : bool, default True
        Use log scale for x-axis.
    **kwargs
        Additional arguments for line plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    Examples
    --------
    >>> from xtimeseries import bootstrap_ci, return_level_plot
    >>> ci = bootstrap_ci(annual_max, [10, 50, 100, 200])
    >>> fig, ax = return_level_plot(
    ...     annual_max, **params,
    ...     return_periods=ci['return_periods'],
    ...     ci_lower=ci['lower'],
    ...     ci_upper=ci['upper']
    ... )
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    data = np.sort(data[~np.isnan(data)])
    n = len(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    if return_periods is None:
        return_periods = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    return_periods = np.asarray(return_periods)

    # Theoretical return levels
    p = 1 - 1 / return_periods
    theoretical_rl = stats.genextreme.ppf(p, c=-shape, loc=loc, scale=scale)

    # Plot theoretical line
    line_kwargs = {"lw": 2, "color": "C0"}
    line_kwargs.update(kwargs)
    ax.plot(return_periods, theoretical_rl, **line_kwargs, label="GEV fit")

    # Confidence intervals
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            return_periods,
            ci_lower,
            ci_upper,
            alpha=0.3,
            color=line_kwargs.get("color", "C0"),
            label="95% CI",
        )

    # Empirical return periods (plotting positions)
    empirical_rp = (n + 1) / np.arange(n, 0, -1)
    ax.scatter(empirical_rp, data, color="k", s=20, zorder=5, label="Observations")

    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Return level")
    ax.set_title("Return Level Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def density_plot(
    data: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
    ax: Any = None,
    bins: int = 30,
    **kwargs,
) -> tuple:
    """
    Compare histogram of data with fitted GEV density.

    Parameters
    ----------
    data : array-like
        Block maxima data.
    loc, scale, shape : float
        GEV parameters.
    ax : matplotlib axes, optional
        Axes to plot on.
    bins : int, default 30
        Number of histogram bins.
    **kwargs
        Additional arguments for density line.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    # Histogram
    ax.hist(data, bins=bins, density=True, alpha=0.5, color="gray", label="Data")

    # Fitted density
    x = np.linspace(data.min() - 0.1 * np.ptp(data), data.max() + 0.1 * np.ptp(data), 200)
    pdf = stats.genextreme.pdf(x, c=-shape, loc=loc, scale=scale)

    line_kwargs = {"lw": 2, "color": "C0"}
    line_kwargs.update(kwargs)
    ax.plot(x, pdf, **line_kwargs, label="GEV fit")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Density Comparison")
    ax.legend()

    return fig, ax


def diagnostic_plots(
    data: ArrayLike,
    loc: float,
    scale: float,
    shape: float,
    figsize: tuple = (10, 10),
) -> tuple:
    """
    Create a 4-panel diagnostic plot.

    Includes: probability plot, Q-Q plot, return level plot, density plot.

    Parameters
    ----------
    data : array-like
        Block maxima data.
    loc, scale, shape : float
        GEV parameters.
    figsize : tuple, default (10, 10)
        Figure size.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes array.

    Examples
    --------
    >>> from xtimeseries import fit_gev, diagnostic_plots
    >>> params = fit_gev(annual_max)
    >>> fig, axes = diagnostic_plots(annual_max, **params)
    >>> plt.tight_layout()
    >>> plt.show()
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    probability_plot(data, loc, scale, shape, ax=axes[0, 0])
    qq_plot(data, loc, scale, shape, ax=axes[0, 1])
    return_level_plot(data, loc, scale, shape, ax=axes[1, 0])
    density_plot(data, loc, scale, shape, ax=axes[1, 1])

    fig.suptitle("GEV Diagnostic Plots", fontsize=14, y=1.02)
    fig.tight_layout()

    return fig, axes


def mrl_plot(
    data: ArrayLike,
    thresholds: ArrayLike | None = None,
    ax: Any = None,
    **kwargs,
) -> tuple:
    """
    Create mean residual life plot for threshold selection.

    Parameters
    ----------
    data : array-like
        Input data.
    thresholds : array-like, optional
        Thresholds to evaluate.
    ax : matplotlib axes, optional
        Axes to plot on.
    **kwargs
        Additional arguments for errorbar plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    See Also
    --------
    mean_residual_life : Compute MRL data.
    """
    import matplotlib.pyplot as plt
    from .pot import mean_residual_life

    mrl = mean_residual_life(data, thresholds)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    eb_kwargs = {"capsize": 3, "capthick": 1, "fmt": "o-", "markersize": 4}
    eb_kwargs.update(kwargs)
    ax.errorbar(mrl["thresholds"], mrl["mean_excess"], yerr=mrl["se"], **eb_kwargs)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean excess")
    ax.set_title("Mean Residual Life Plot")
    ax.grid(True, alpha=0.3)

    return fig, ax


def stability_plot(
    data: ArrayLike,
    thresholds: ArrayLike | None = None,
    ax: Any = None,
    **kwargs,
) -> tuple:
    """
    Create parameter stability plot for threshold selection.

    Shows GPD shape parameter vs threshold. Look for where it stabilizes.

    Parameters
    ----------
    data : array-like
        Input data.
    thresholds : array-like, optional
        Thresholds to evaluate.
    ax : matplotlib axes, optional
        Axes to plot on.
    **kwargs
        Additional arguments for plot.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.

    See Also
    --------
    threshold_stability : Compute stability data.
    """
    import matplotlib.pyplot as plt
    from .pot import threshold_stability

    stab = threshold_stability(data, thresholds)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    plot_kwargs = {"marker": "o", "markersize": 4}
    plot_kwargs.update(kwargs)
    ax.plot(stab["thresholds"], stab["shape"], **plot_kwargs)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Shape parameter")
    ax.set_title("GPD Shape Parameter Stability")
    ax.grid(True, alpha=0.3)

    return fig, ax
