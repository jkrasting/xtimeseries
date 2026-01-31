Temperature Extremes Analysis
=============================

This example demonstrates a complete workflow for analyzing annual maximum
temperatures using the GEV distribution.

Setup
-----

.. code-block:: python

   import xtimeseries as xts
   import numpy as np
   import xarray as xr
   import matplotlib.pyplot as plt

Generating Synthetic Temperature Data
-------------------------------------

For this example, we generate realistic synthetic temperature data with
seasonal cycles and interannual variability:

.. code-block:: python

   # Generate 50 years of daily temperature data
   temp_data = xts.generate_temperature_like(
       n_years=50,
       mean_temp=15.0,      # Annual mean (°C)
       seasonal_amp=10.0,   # Seasonal amplitude (°C)
       trend=0.02,          # Trend (°C/year)
       ar1_coef=0.7,        # Autocorrelation
       noise_std=3.0,       # Daily variability
       seed=42,
       calendar='standard'
   )

   # Convert to xarray DataArray
   times = xr.date_range('1970-01-01', periods=len(temp_data['data']), freq='D')
   temperature = xr.DataArray(
       temp_data['data'],
       dims=['time'],
       coords={'time': times},
       attrs={'units': '°C', 'long_name': 'Daily maximum temperature'}
   )

   print(f"Data shape: {temperature.shape}")
   print(f"Time range: {temperature.time.values[0]} to {temperature.time.values[-1]}")

Extracting Annual Maxima
------------------------

Extract the maximum temperature for each year:

.. code-block:: python

   # Extract annual maxima
   annual_max = xts.block_maxima(temperature, freq='YE')

   print(f"Annual maxima shape: {annual_max.shape}")
   print(f"Mean annual max: {annual_max.mean().values:.1f}°C")
   print(f"Std annual max: {annual_max.std().values:.1f}°C")

Visualize the annual maxima time series:

.. code-block:: python

   fig, ax = plt.subplots(figsize=(10, 4))
   annual_max.plot(ax=ax, marker='o', linestyle='-', markersize=4)
   ax.set_ylabel('Annual Maximum Temperature (°C)')
   ax.set_title('Annual Maximum Temperature Time Series')
   plt.tight_layout()
   plt.show()

.. figure:: /_static/temperature_annual_max.png
   :width: 600px
   :align: center
   :alt: Annual maximum temperature time series

   Time series of annual maximum temperatures showing year-to-year variability.

Fitting the GEV Distribution
----------------------------

Fit a GEV distribution to the annual maxima:

.. code-block:: python

   # Fit GEV
   params = xts.fit_gev(annual_max.values)

   print("GEV Parameters:")
   print(f"  Location (μ): {params['loc']:.2f}°C")
   print(f"  Scale (σ):    {params['scale']:.2f}°C")
   print(f"  Shape (ξ):    {params['shape']:.3f}")

Interpret the shape parameter:

.. code-block:: python

   if params['shape'] > 0.05:
       print("\nFréchet-type (ξ > 0): Heavy upper tail")
   elif params['shape'] < -0.05:
       print("\nWeibull-type (ξ < 0): Bounded upper tail")
       upper_bound = params['loc'] - params['scale'] / params['shape']
       print(f"Upper bound: {upper_bound:.1f}°C")
   else:
       print("\nNear-Gumbel (ξ ≈ 0): Light exponential tail")

Temperature maxima typically show Weibull behavior (ξ < 0), indicating a
physical upper bound.

Diagnostic Plots
----------------

Validate the fit using diagnostic plots:

.. code-block:: python

   fig, axes = xts.diagnostic_plots(annual_max.values, **params)
   fig.suptitle('GEV Diagnostic Plots for Annual Maximum Temperature', y=1.02)
   plt.tight_layout()
   plt.show()

.. figure:: /_static/temperature_fit.png
   :width: 700px
   :align: center
   :alt: GEV diagnostic plots

   Four-panel diagnostic plots: (a) probability plot, (b) Q-Q plot,
   (c) return level plot, (d) density plot.

Return Level Analysis
---------------------

Calculate return levels for various return periods:

.. code-block:: python

   return_periods = [2, 5, 10, 25, 50, 100]
   return_levels = xts.return_level(return_periods, **params)

   print("\nReturn Level Analysis:")
   print("-" * 35)
   print(f"{'Return Period':>15} {'Return Level':>15}")
   print("-" * 35)
   for T, rl in zip(return_periods, return_levels):
       print(f"{T:>12} yr  {rl:>12.1f}°C")

Bootstrap Confidence Intervals
------------------------------

Estimate uncertainty in return levels:

.. code-block:: python

   ci = xts.bootstrap_ci(
       annual_max.values,
       return_periods=[10, 50, 100],
       n_bootstrap=1000,
       ci_level=0.95,
       seed=42
   )

   print("\nReturn Levels with 95% Confidence Intervals:")
   print("-" * 50)
   for i, T in enumerate(ci['return_periods']):
       print(f"{T:3d}-year: {ci['return_levels'][i]:5.1f}°C  "
             f"[{ci['lower'][i]:5.1f}, {ci['upper'][i]:5.1f}]")

Return Level Plot
-----------------

Create a publication-quality return level plot:

.. code-block:: python

   fig, ax = xts.return_level_plot(
       annual_max.values, **params,
       ci_level=0.95,
       max_return_period=200
   )
   ax.set_ylabel('Return Level (°C)')
   ax.set_title('Temperature Return Levels')
   plt.tight_layout()
   plt.show()

.. figure:: /_static/temperature_return_level.png
   :width: 600px
   :align: center
   :alt: Temperature return level plot

   Return level plot showing the fitted GEV with 95% confidence bands
   and observed annual maxima.

Probability of Exceeding a Threshold
------------------------------------

Calculate the probability of exceeding a specific temperature:

.. code-block:: python

   threshold = 40.0  # °C

   # Probability of NOT exceeding in a single year
   p_not_exceed = xts.gev_cdf(threshold, **params)

   # Probability of exceeding in a single year
   p_exceed = 1 - p_not_exceed

   # Return period
   T = xts.return_period(threshold, **params)

   print(f"\nAnalysis for threshold {threshold}°C:")
   print(f"  Annual exceedance probability: {p_exceed:.4f} ({p_exceed*100:.2f}%)")
   print(f"  Return period: {T:.1f} years")

   # Probability of at least one exceedance in 30 years
   p_30yr = 1 - (1 - p_exceed)**30
   print(f"  P(exceed at least once in 30 years): {p_30yr:.3f} ({p_30yr*100:.1f}%)")

Summary
-------

This example demonstrated:

1. **Data preparation**: Generating synthetic temperature data and extracting
   annual maxima
2. **GEV fitting**: Estimating distribution parameters with physical
   interpretation
3. **Diagnostics**: Validating the fit using probability, Q-Q, and density plots
4. **Return levels**: Calculating return levels with bootstrap confidence
   intervals
5. **Exceedance analysis**: Computing probabilities for specific thresholds

For real-world applications, replace the synthetic data with observed station
data or gridded reanalysis products.

See Also
--------

- :doc:`precipitation_idf` - Similar analysis for precipitation
- :doc:`trend_analysis` - Detecting trends in extremes
- :doc:`../theory/gev_distribution` - Mathematical background
