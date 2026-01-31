Trend Analysis in Extremes
==========================

This example demonstrates how to detect and model non-stationary trends in
extreme values using xtimeseries.

Setup
-----

.. code-block:: python

   import xtimeseries as xts
   import numpy as np
   import matplotlib.pyplot as plt

Generating Data with a Trend
----------------------------

Create synthetic annual maxima with a known trend:

.. code-block:: python

   # Generate data with trend in location
   ns_data = xts.generate_nonstationary_series(
       n=70,          # 70 years (e.g., 1950-2019)
       loc0=30.0,     # Baseline location (°C)
       loc1=0.03,     # Trend: +0.03°C per year
       scale=3.0,     # Scale parameter
       shape=-0.1,    # Shape parameter (Weibull-type)
       seed=42
   )

   years = np.arange(1950, 2020)
   annual_max = ns_data['data']

   # Known true parameters for comparison
   print("True parameters:")
   print(f"  Location intercept (μ₀): {ns_data['true_params']['loc0']:.2f}")
   print(f"  Location trend (μ₁):     {ns_data['true_params']['loc1']:.4f} per year")
   print(f"  Scale (σ):               {ns_data['true_params']['scale']:.2f}")
   print(f"  Shape (ξ):               {ns_data['true_params']['shape']:.3f}")

Visualizing the Data
--------------------

.. code-block:: python

   fig, ax = plt.subplots(figsize=(10, 5))

   ax.scatter(years, annual_max, alpha=0.7, s=40, label='Annual maxima')

   # Add trend line (true values)
   true_trend = ns_data['true_loc']
   ax.plot(years, true_trend, 'r-', linewidth=2, label='True location trend')

   ax.set_xlabel('Year')
   ax.set_ylabel('Annual Maximum (°C)')
   ax.set_title('Annual Maximum Temperature with Trend')
   ax.legend()
   plt.tight_layout()
   plt.show()

.. figure:: /_static/trend_detection.png
   :width: 600px
   :align: center
   :alt: Annual maxima with trend

   Annual maximum values showing an increasing trend over time.

Testing for Non-Stationarity
----------------------------

Before fitting a non-stationary model, test whether the trend is statistically
significant:

.. code-block:: python

   # Likelihood ratio test
   result = xts.likelihood_ratio_test(annual_max, years)

   print("\nLikelihood Ratio Test Results:")
   print(f"  Stationary log-likelihood:     {result['ll_stationary']:.2f}")
   print(f"  Non-stationary log-likelihood: {result['ll_nonstationary']:.2f}")
   print(f"  Test statistic (Λ):            {result['statistic']:.2f}")
   print(f"  p-value:                       {result['p_value']:.4f}")
   print(f"  Significant at α=0.05:         {result['significant']}")

Interpretation:

- **p < 0.05**: Reject the null hypothesis (stationarity); trend is significant
- **p ≥ 0.05**: Cannot reject stationarity; trend may be due to chance

Fitting Non-Stationary GEV
--------------------------

If the test indicates a significant trend, fit a non-stationary GEV:

.. code-block:: python

   # Fit non-stationary GEV with trend in location
   ns_params = xts.fit_nonstationary_gev(annual_max, years, trend_in='loc')

   print("\nNon-Stationary GEV Parameters:")
   print(f"  Location intercept (μ₀): {ns_params['loc0']:.2f}")
   print(f"  Location trend (μ₁):     {ns_params['loc1']:.4f} per year")
   print(f"  Scale (σ):               {ns_params['scale']:.2f}")
   print(f"  Shape (ξ):               {ns_params['shape']:.3f}")
   print(f"  AIC:                     {ns_params['aic']:.2f}")
   print(f"  BIC:                     {ns_params['bic']:.2f}")

Comparing with Stationary Fit
-----------------------------

.. code-block:: python

   # Fit stationary GEV for comparison
   stat_params = xts.fit_gev(annual_max)

   print("\nStationary GEV Parameters:")
   print(f"  Location (μ): {stat_params['loc']:.2f}")
   print(f"  Scale (σ):    {stat_params['scale']:.2f}")
   print(f"  Shape (ξ):    {stat_params['shape']:.3f}")

Interpreting the Trend
----------------------

Convert the trend to meaningful quantities:

.. code-block:: python

   trend_per_year = ns_params['loc1']
   trend_per_decade = trend_per_year * 10
   total_trend = trend_per_year * (years[-1] - years[0])

   print("\nTrend Interpretation:")
   print(f"  Change per year:   {trend_per_year:+.4f}°C/year")
   print(f"  Change per decade: {trend_per_decade:+.3f}°C/decade")
   print(f"  Total change ({years[0]}-{years[-1]}): {total_trend:+.2f}°C")

Time-Varying Return Levels
--------------------------

Calculate how return levels have changed over time:

.. code-block:: python

   # 100-year return level at different time points
   reference_years = [1950, 1970, 1990, 2010, 2019]

   print("\n100-year Return Level Over Time:")
   print("-" * 40)
   for year in reference_years:
       rl = xts.nonstationary_return_level(100, ns_params, covariate=year)
       print(f"  {year}: {rl:.2f}°C")

   # Change in return level
   rl_1950 = xts.nonstationary_return_level(100, ns_params, covariate=1950)
   rl_2019 = xts.nonstationary_return_level(100, ns_params, covariate=2019)
   print(f"\nChange in 100-year level: {rl_2019 - rl_1950:+.2f}°C")

Changing Return Periods
-----------------------

A fixed value now has a different return period than in the past:

.. code-block:: python

   # What was a 100-year event in 1950?
   rl_100_1950 = xts.nonstationary_return_level(100, ns_params, covariate=1950)

   # What is the return period of that value in 2019?
   T_2019 = xts.effective_return_level(rl_100_1950, ns_params, covariate=2019)

   print(f"\n{rl_100_1950:.1f}°C was a 100-year event in 1950")
   print(f"{rl_100_1950:.1f}°C is now a {T_2019:.0f}-year event in 2019")

Visualization of Non-Stationary Fit
-----------------------------------

.. code-block:: python

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # Left: Data with fitted trend
   ax1 = axes[0]
   ax1.scatter(years, annual_max, alpha=0.7, s=40, label='Observations')

   # Fitted location trend
   fitted_loc = ns_params['loc0'] + ns_params['loc1'] * (years - np.mean(years))
   ax1.plot(years, fitted_loc, 'r-', linewidth=2, label='Fitted μ(t)')

   # Quantile bands
   for q, color, label in [(0.05, 'blue', '5th percentile'),
                            (0.95, 'blue', '95th percentile')]:
       quantile = xts.gev_ppf(q, loc=fitted_loc, scale=ns_params['scale'],
                              shape=ns_params['shape'])
       ax1.plot(years, quantile, '--', color=color, alpha=0.5, label=label)

   ax1.set_xlabel('Year')
   ax1.set_ylabel('Annual Maximum (°C)')
   ax1.set_title('Non-Stationary GEV Fit')
   ax1.legend(loc='upper left')

   # Right: Return level curves for different years
   ax2 = axes[1]
   T_range = np.array([2, 5, 10, 20, 50, 100, 200])

   for year, color in [(1950, 'blue'), (1985, 'green'), (2019, 'red')]:
       loc_t = ns_params['loc0'] + ns_params['loc1'] * (year - np.mean(years))
       rl = xts.return_level(T_range, loc=loc_t, scale=ns_params['scale'],
                             shape=ns_params['shape'])
       ax2.semilogx(T_range, rl, '-o', color=color, label=f'{year}')

   ax2.set_xlabel('Return Period (years)')
   ax2.set_ylabel('Return Level (°C)')
   ax2.set_title('Return Level Curves by Year')
   ax2.legend(title='Reference Year')
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

.. figure:: /_static/nonstationary_visualization.png
   :width: 700px
   :align: center
   :alt: Non-stationary GEV visualization

   (Left) Non-stationary GEV fit showing observations, fitted location trend,
   and 5th/95th percentile bands. (Right) Return level curves for different
   reference years, illustrating how extremes have intensified over time.

Bootstrap Confidence Intervals for Trend
----------------------------------------

Estimate uncertainty in the trend parameter:

.. code-block:: python

   n_bootstrap = 1000
   trends = []
   rng = np.random.default_rng(42)

   for _ in range(n_bootstrap):
       # Resample with replacement
       idx = rng.choice(len(annual_max), size=len(annual_max), replace=True)
       boot_data = annual_max[idx]
       boot_years = years[idx]

       # Fit non-stationary GEV
       try:
           boot_params = xts.fit_nonstationary_gev(boot_data, boot_years, trend_in='loc')
           trends.append(boot_params['loc1'])
       except:
           pass  # Skip failed fits

   trends = np.array(trends)

   print("\nTrend Parameter Uncertainty:")
   print(f"  Point estimate: {ns_params['loc1']:.5f}°C/year")
   print(f"  Bootstrap mean: {np.mean(trends):.5f}°C/year")
   print(f"  Bootstrap std:  {np.std(trends):.5f}°C/year")
   print(f"  95% CI: [{np.percentile(trends, 2.5):.5f}, {np.percentile(trends, 97.5):.5f}]")

   # Is zero in the confidence interval?
   if np.percentile(trends, 2.5) > 0:
       print("\n  ✓ Trend is significantly positive (zero not in 95% CI)")
   elif np.percentile(trends, 97.5) < 0:
       print("\n  ✓ Trend is significantly negative (zero not in 95% CI)")
   else:
       print("\n  ✗ Trend is not significant (zero in 95% CI)")

Gridded Trend Analysis
----------------------

Apply non-stationary analysis to gridded data:

.. code-block:: python

   # Generate test dataset with spatial trend
   ds = xts.generate_test_dataset(nlat=5, nlon=8, nyears=50, seed=42)

   # Extract annual maxima
   annual_max_grid = xts.xr_block_maxima(ds['data'], freq='YE')

   # Fit non-stationary GEV at each grid point
   years_grid = annual_max_grid.time.dt.year.values
   ns_params_grid = xts.xr_fit_nonstationary_gev(annual_max_grid, years_grid, dim='time')

   # Plot the trend map
   fig, ax = plt.subplots(figsize=(8, 4))
   ns_params_grid['loc1'].plot(ax=ax, cmap='RdBu_r', center=0)
   ax.set_title('Location Trend (units/year)')
   plt.tight_layout()
   plt.show()

.. figure:: /_static/gridded_trend_map.png
   :width: 600px
   :align: center
   :alt: Gridded trend map

   Spatial pattern of location parameter trends showing regions with
   increasing (red) and decreasing (blue) extremes.

Summary
-------

This example demonstrated:

1. **Trend detection**: Using the likelihood ratio test to identify significant
   trends
2. **Non-stationary fitting**: Fitting GEV with time-varying location
3. **Trend interpretation**: Converting slopes to meaningful climate change
   metrics
4. **Time-varying return levels**: Understanding how return levels change
   over time
5. **Uncertainty quantification**: Bootstrap confidence intervals for trends
6. **Gridded analysis**: Mapping trends across spatial domains

Key considerations for real-world applications:

- Ensure the record is long enough (50+ years recommended)
- Consider other sources of non-stationarity (decadal variability, ENSO)
- Validate trends against physical understanding
- Report uncertainty in trend estimates

See Also
--------

- :doc:`../theory/nonstationary_eva` - Theory of non-stationary EVA
- :func:`xtimeseries.fit_nonstationary_gev` - API reference
- :func:`xtimeseries.likelihood_ratio_test` - Trend significance testing
