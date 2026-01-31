Non-Stationary EVA
==================

Classical extreme value analysis assumes **stationarity**: the statistical
properties of extremes remain constant over time. In a changing climate, this
assumption is often violated. Non-stationary EVA allows GEV parameters to
vary with time or other covariates.

Why Non-Stationary Analysis?
----------------------------

Climate change affects extremes through:

1. **Trends in mean conditions**: Warming temperatures shift the entire
   distribution, including the tail
2. **Changes in variability**: Increased variance amplifies extremes
3. **Changing tail behavior**: The shape of the distribution may change

Evidence for non-stationarity includes:

- Increasing frequency of record-breaking events
- Trends in annual maxima time series
- Physical understanding of climate forcing

Modeling Framework
------------------

Non-stationary GEV allows parameters to depend on covariates (typically time):

Location-Only Trend
^^^^^^^^^^^^^^^^^^^

The most common model assumes a linear trend in location:

.. math::

   \mu(t) = \mu_0 + \mu_1 t

where :math:`\mu_0` is the intercept and :math:`\mu_1` is the slope (trend
per unit time).

.. code-block:: python

   import xtimeseries as xts
   import numpy as np

   years = np.arange(1950, 2024)
   params = xts.fit_nonstationary_gev(annual_max, years, trend_in='loc')

   print(f"Location intercept: {params['loc0']:.2f}")
   print(f"Location trend: {params['loc1']:.4f} per year")

Scale Trend
^^^^^^^^^^^

For changes in variability, use a log-linear scale trend:

.. math::

   \sigma(t) = \exp(\sigma_0 + \sigma_1 t)

The exponential ensures :math:`\sigma(t) > 0` for all t.

.. code-block:: python

   params = xts.fit_nonstationary_gev(annual_max, years, trend_in='scale')

Combined Trends
^^^^^^^^^^^^^^^

Both location and scale can vary (not yet implemented in xtimeseries).

Covariate Standardization
-------------------------

For numerical stability, covariates are standardized internally:

.. math::

   t^* = \frac{t - \bar{t}}{s_t}

where :math:`\bar{t}` is the mean and :math:`s_t` is the standard deviation.

The returned parameters are in original units:

.. code-block:: python

   # params['loc1'] is the trend per year (original units)
   # Internal fitting uses standardized time

Model Selection
---------------

Likelihood Ratio Test
^^^^^^^^^^^^^^^^^^^^^

Compare stationary (null) vs. non-stationary (alternative) models:

.. math::

   \Lambda = 2(\ell_1 - \ell_0)

where :math:`\ell_0` and :math:`\ell_1` are the log-likelihoods of the
stationary and non-stationary models. Under the null hypothesis,
:math:`\Lambda \sim \chi^2_k` where k is the number of additional parameters.

.. code-block:: python

   result = xts.likelihood_ratio_test(annual_max, years)

   print(f"Test statistic: {result['statistic']:.2f}")
   print(f"p-value: {result['p_value']:.4f}")
   print(f"Significant trend (α=0.05): {result['significant']}")

A significant result (p < 0.05) suggests the non-stationary model is justified.

Information Criteria
^^^^^^^^^^^^^^^^^^^^

AIC and BIC balance fit quality against model complexity:

.. math::

   \text{AIC} = 2k - 2\ell

.. math::

   \text{BIC} = k\ln(n) - 2\ell

where k is the number of parameters, n is the sample size, and :math:`\ell`
is the log-likelihood.

.. code-block:: python

   # Fit both models
   stat_params = xts.fit_gev(annual_max)
   ns_params = xts.fit_nonstationary_gev(annual_max, years, trend_in='loc')

   # Compare (lower is better)
   # Note: stationary fit doesn't return AIC/BIC directly

Return Levels in Non-Stationary Context
---------------------------------------

When parameters vary with time, the concept of return level must be
reinterpreted. Two approaches:

Effective Return Level
^^^^^^^^^^^^^^^^^^^^^^

Compare return levels at different time points:

.. code-block:: python

   # Return level at start of record (1950)
   rl_1950 = xts.nonstationary_return_level(100, params, covariate=1950)

   # Return level at end of record (2023)
   rl_2023 = xts.nonstationary_return_level(100, params, covariate=2023)

   print(f"100-year level in 1950: {rl_1950:.1f}")
   print(f"100-year level in 2023: {rl_2023:.1f}")

Changing Return Period
^^^^^^^^^^^^^^^^^^^^^^

A fixed return level corresponds to different return periods over time:

.. code-block:: python

   # What return period corresponds to 50°C?
   T_1950 = xts.effective_return_level(50, params, covariate=1950)
   T_2023 = xts.effective_return_level(50, params, covariate=2023)

   print(f"50°C was a {T_1950:.0f}-year event in 1950")
   print(f"50°C is now a {T_2023:.0f}-year event in 2023")

Interpretation of Trends
------------------------

Converting Slope to Change
^^^^^^^^^^^^^^^^^^^^^^^^^^

The location slope :math:`\mu_1` gives change per unit time:

.. code-block:: python

   # Total change over the record
   total_change = params['loc1'] * (years[-1] - years[0])

   # Change per decade
   change_per_decade = params['loc1'] * 10

   print(f"Total change: {total_change:.2f}")
   print(f"Change per decade: {change_per_decade:.2f}")

Uncertainty in Trends
^^^^^^^^^^^^^^^^^^^^^

Bootstrap the trend estimate:

.. code-block:: python

   import numpy as np

   trends = []
   rng = np.random.default_rng(42)

   for _ in range(1000):
       idx = rng.choice(len(annual_max), size=len(annual_max), replace=True)
       boot_data = annual_max[idx]
       boot_years = years[idx]
       boot_params = xts.fit_nonstationary_gev(boot_data, boot_years, trend_in='loc')
       trends.append(boot_params['loc1'])

   trends = np.array(trends)
   print(f"Trend: {np.mean(trends):.4f} [{np.percentile(trends, 2.5):.4f}, {np.percentile(trends, 97.5):.4f}]")

Gridded Non-Stationary Analysis
-------------------------------

Apply non-stationary fitting to gridded data:

.. code-block:: python

   # ds has dimensions (time, lat, lon)
   years = ds.time.dt.year.values

   ns_params = xts.xr_fit_nonstationary_gev(ds['data'], years, dim='time')

   # Plot the trend map
   ns_params['loc1'].plot()  # Location trend per year at each grid point

Practical Considerations
------------------------

Sample Size
^^^^^^^^^^^

Non-stationary models have more parameters and require more data:

- **Stationary GEV:** 3 parameters, recommend 30+ years
- **Non-stationary (loc):** 4 parameters, recommend 50+ years
- **Non-stationary (loc + scale):** 5 parameters, recommend 70+ years

With insufficient data, trends may be spurious or parameters poorly constrained.

Physical Plausibility
^^^^^^^^^^^^^^^^^^^^^

Statistical significance alone is not sufficient. Trends should be:

- Consistent with physical understanding
- Robust to removal of individual extreme years
- Consistent with regional or global patterns

Attribution
^^^^^^^^^^^

A significant trend does not imply causation. For attribution to climate
change, additional analysis is needed (detection and attribution studies).

Limitations
^^^^^^^^^^^

Current xtimeseries implementation:

- Supports linear trends only (not polynomial or nonlinear)
- Single covariate (typically time)
- Either location or scale trend, not both simultaneously
- Shape parameter assumed constant

See Also
--------

- :doc:`extreme_value_theory` - EVT fundamentals
- :func:`xtimeseries.fit_nonstationary_gev` - API reference
- :func:`xtimeseries.likelihood_ratio_test` - Model comparison
- :doc:`../gallery/nonstationary_trends` - Worked example
