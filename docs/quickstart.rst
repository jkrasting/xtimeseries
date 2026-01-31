Quick Start Guide
=================

This guide provides a hands-on introduction to xtimeseries. All examples
use the convention:

.. code-block:: python

   import xtimeseries as xts
   import numpy as np

Fitting a GEV Distribution
--------------------------

The most common task is fitting a Generalized Extreme Value (GEV) distribution
to block maxima data:

.. code-block:: python

   import xtimeseries as xts

   # Generate synthetic annual maxima (100 years)
   data = xts.generate_gev_series(100, loc=30, scale=5, shape=0.1, seed=42)

   # Fit GEV distribution
   params = xts.fit_gev(data)

   print(f"Location (μ): {params['loc']:.2f}")
   print(f"Scale (σ):    {params['scale']:.2f}")
   print(f"Shape (ξ):    {params['shape']:.3f}")

The shape parameter follows the climate convention:

- ξ > 0: Fréchet (heavy tail, unbounded) - typical for precipitation
- ξ = 0: Gumbel (light exponential tail)
- ξ < 0: Weibull (bounded upper tail) - typical for temperature maxima

Calculating Return Levels
-------------------------

A return level is the value expected to be exceeded on average once every
T years:

.. code-block:: python

   # Calculate the 100-year return level
   rl_100 = xts.return_level(100, **params)
   print(f"100-year return level: {rl_100:.2f}")

   # Multiple return periods at once
   return_periods = [10, 25, 50, 100]
   levels = xts.return_level(return_periods, **params)

   for T, level in zip(return_periods, levels):
       print(f"{T:3d}-year: {level:.2f}")

You can also calculate the return period for a given value:

.. code-block:: python

   # What is the return period of a value of 50?
   T = xts.return_period(50, **params)
   print(f"Return period for 50: {T:.1f} years")

Block Maxima from xarray
------------------------

Extract annual maxima from a time series stored in an xarray DataArray:

.. code-block:: python

   import xarray as xr

   # Create sample daily data (10 years)
   times = xr.date_range("2000-01-01", periods=365 * 10, freq="D")
   temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(3650) / 365) + np.random.randn(3650) * 3

   da = xr.DataArray(temperature, dims=["time"], coords={"time": times}, name="temperature")

   # Extract annual maxima
   annual_max = xts.block_maxima(da, freq="YE")
   print(annual_max)

For climate model data with non-standard calendars:

.. code-block:: python

   # Works with cftime calendars (noleap, 360_day, etc.)
   times = xr.date_range(
       "0001-01-01", periods=365 * 100, freq="D",
       calendar="noleap", use_cftime=True
   )
   da = xr.DataArray(np.random.randn(36500), dims=["time"], coords={"time": times})

   annual_max = xts.block_maxima(da)  # Automatically detects calendar

Bootstrap Confidence Intervals
------------------------------

Estimate uncertainty in return levels using bootstrap resampling:

.. code-block:: python

   # Bootstrap confidence intervals
   ci = xts.bootstrap_ci(
       data,
       return_periods=[10, 50, 100],
       n_bootstrap=1000,
       ci_level=0.95
   )

   for i, T in enumerate(ci["return_periods"]):
       print(f"{T}-year: {ci['return_levels'][i]:.2f} "
             f"[{ci['lower'][i]:.2f}, {ci['upper'][i]:.2f}]")

Working with Gridded Data
-------------------------

Fit GEV distributions across a spatial grid using xarray integration:

.. code-block:: python

   # Create a test dataset with spatial dimensions
   ds = xts.generate_test_dataset(nlat=10, nlon=15, nyears=50, seed=42)

   # Fit GEV at each grid point
   params_ds = xts.xr_fit_gev(ds["data"], dim="time")
   print(params_ds)

   # Calculate return levels at each grid point
   rl = xts.xr_return_level([10, 50, 100], params_ds)
   print(rl)

Non-Stationary Analysis
-----------------------

Detect and model trends in extreme values:

.. code-block:: python

   # Generate data with a trend
   ns_data = xts.generate_nonstationary_series(
       n=100, loc0=30, loc1=0.1, scale=5, shape=0.1, seed=42
   )

   # Test for significant trend
   years = np.arange(100)
   result = xts.likelihood_ratio_test(ns_data["data"], years)

   print(f"Likelihood ratio statistic: {result['statistic']:.2f}")
   print(f"p-value: {result['p_value']:.4f}")
   print(f"Significant trend: {result['significant']}")

   if result["significant"]:
       # Fit non-stationary GEV
       ns_params = xts.fit_nonstationary_gev(ns_data["data"], years, trend_in="loc")
       print(f"Location trend: {ns_params['loc1']:.4f} per year")

Peaks Over Threshold (POT)
--------------------------

For threshold exceedance analysis:

.. code-block:: python

   # Generate daily precipitation-like data
   precip = xts.generate_precipitation_like(n_years=30, seed=42)

   # Select threshold (95th percentile)
   threshold = xts.select_threshold(precip["data"], method="quantile", quantile=0.95)

   # Extract exceedances
   pot = xts.peaks_over_threshold(precip["data"], threshold, decluster=True)
   print(f"Threshold: {threshold:.2f}")
   print(f"Number of exceedances: {pot['n_exceedances']}")

   # Fit GPD to exceedances
   gpd_params = xts.fit_gpd(pot["exceedances"], threshold=0)  # Already shifted
   print(f"GPD scale: {gpd_params['scale']:.2f}")
   print(f"GPD shape: {gpd_params['shape']:.3f}")

Diagnostic Plots
----------------

Visualize the quality of your GEV fit:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Create diagnostic plots
   fig, axes = xts.diagnostic_plots(data, **params)
   plt.tight_layout()
   plt.show()

Individual plots are also available:

.. code-block:: python

   # Probability plot
   fig, ax = xts.probability_plot(data, **params)

   # Return level plot with confidence intervals
   fig, ax = xts.return_level_plot(data, **params, ci_level=0.95)

   # Q-Q plot
   fig, ax = xts.qq_plot(data, **params)

Next Steps
----------

- Learn about the :doc:`theory behind extreme value analysis <theory/extreme_value_theory>`
- See :doc:`worked examples <examples/temperature_extremes>` with real climate data
- Explore the full :doc:`API reference <api/index>`
