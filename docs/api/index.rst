API Reference
=============

This page documents the public API of xtimeseries. All functions listed here
are available from the top-level namespace:

.. code-block:: python

   import xtimeseries as xts
   params = xts.fit_gev(data)

.. contents:: Contents
   :local:
   :depth: 1

Distribution Fitting
--------------------

Functions for fitting GEV and GPD distributions to data.

.. autofunction:: xtimeseries.fit_gev

.. autofunction:: xtimeseries.fit_gpd

.. autofunction:: xtimeseries.gev_cdf

.. autofunction:: xtimeseries.gev_pdf

.. autofunction:: xtimeseries.gev_ppf

.. autofunction:: xtimeseries.gpd_cdf

.. autofunction:: xtimeseries.gpd_pdf

.. autofunction:: xtimeseries.gpd_ppf

Block Maxima
------------

Functions for extracting block maxima and minima from time series.

.. autofunction:: xtimeseries.block_maxima

.. autofunction:: xtimeseries.block_minima

Return Periods
--------------

Functions for calculating return levels and return periods.

.. autofunction:: xtimeseries.return_level

.. autofunction:: xtimeseries.return_period

.. autofunction:: xtimeseries.return_level_gpd

.. autofunction:: xtimeseries.return_period_gpd

Confidence Intervals
--------------------

Bootstrap methods for uncertainty quantification.

.. autofunction:: xtimeseries.bootstrap_ci

.. autofunction:: xtimeseries.bootstrap_return_levels

Non-Stationary Analysis
-----------------------

Functions for detecting and modeling trends in extremes.

.. autofunction:: xtimeseries.fit_nonstationary_gev

.. autofunction:: xtimeseries.nonstationary_return_level

.. autofunction:: xtimeseries.likelihood_ratio_test

.. autofunction:: xtimeseries.effective_return_level

Peaks Over Threshold
--------------------

Functions for threshold exceedance analysis with the GPD.

.. autofunction:: xtimeseries.peaks_over_threshold

.. autofunction:: xtimeseries.decluster

.. autofunction:: xtimeseries.mean_residual_life

.. autofunction:: xtimeseries.threshold_stability

.. autofunction:: xtimeseries.select_threshold

Synthetic Data Generation
-------------------------

Functions for generating synthetic data for testing and validation.

.. autofunction:: xtimeseries.generate_gev_series

.. autofunction:: xtimeseries.generate_gpd_series

.. autofunction:: xtimeseries.generate_nonstationary_series

.. autofunction:: xtimeseries.generate_temperature_like

.. autofunction:: xtimeseries.generate_precipitation_like

.. autofunction:: xtimeseries.generate_gev_return_levels

.. autofunction:: xtimeseries.generate_test_dataset

xarray Operations
-----------------

Vectorized operations for gridded climate data.

.. autofunction:: xtimeseries.xr_fit_gev

.. autofunction:: xtimeseries.xr_return_level

.. autofunction:: xtimeseries.xr_block_maxima

.. autofunction:: xtimeseries.xr_fit_nonstationary_gev

Diagnostics
-----------

Plotting functions for model validation and visualization.

.. autofunction:: xtimeseries.probability_plot

.. autofunction:: xtimeseries.return_level_plot

.. autofunction:: xtimeseries.qq_plot

.. autofunction:: xtimeseries.diagnostic_plots
