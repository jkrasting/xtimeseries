New Brunswick Precipitation Extremes
====================================

This example demonstrates extreme value analysis of observed daily precipitation
from the NOAA Cooperative Observer station at New Brunswick, NJ. The analysis
includes trend detection and stationary GEV fitting.

Introduction
------------

Precipitation extremes are a key concern for water resource management, flood
risk assessment, and infrastructure design. Changes in precipitation intensity
under climate change may alter the frequency and magnitude of extreme events.

This example uses 55+ years of daily precipitation observations from New
Brunswick, NJ to:

- Fit a stationary GEV distribution to annual maximum precipitation
- Test for significant trends using the likelihood ratio test
- Calculate return levels for infrastructure design

Data Source
-----------

**Station:** GHCND:USC00286055 (New Brunswick 3 SE, NJ)

**Period:** 1968-2023 (approximately 55 years)

**Variables:** Daily precipitation (PRCP) in inches, temperature (TMAX, TMIN) in °F

**Source:** NOAA Climate Data Online (CDO) API

To obtain the data, run the fetch script:

.. code-block:: bash

   export NOAA_API_TOKEN="your_token_here"
   python scripts/fetch_noaa_data.py

Get your API token from: https://www.ncdc.noaa.gov/cdo-web/token

Setup
-----

.. code-block:: python

   import xtimeseries as xts
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from pathlib import Path

   # Load data
   data_file = Path("tests/data/noaa_new_brunswick.csv")
   df = pd.read_csv(data_file, parse_dates=["date"], index_col="date")

Extracting Annual Maxima
------------------------

Extract the maximum daily precipitation for each year:

.. code-block:: python

   # Get annual maximum precipitation
   annual_max = df["PRCP"].resample("YE").max()
   annual_max = annual_max.dropna()

   values = annual_max.values
   years = annual_max.index.year.values

   print(f"Years: {years[0]} to {years[-1]} ({len(years)} years)")
   print(f"Mean annual max: {values.mean():.2f} inches")
   print(f"Max annual max: {values.max():.2f} inches")

.. figure:: /_static/newbrunswick_annual_max.png
   :width: 600px
   :align: center
   :alt: Annual maximum precipitation time series

   Time series of annual maximum daily precipitation at New Brunswick, NJ
   (1969-2023). The maximum event (~8 inches) occurred during Hurricane Irene in 2011.

Stationary GEV Fit
------------------

Fit a stationary GEV distribution:

.. code-block:: python

   params = xts.fit_gev(values)

   print("GEV Parameters:")
   print(f"  Location (μ): {params['loc']:.2f} inches")
   print(f"  Scale (σ):    {params['scale']:.2f} inches")
   print(f"  Shape (ξ):    {params['shape']:.3f}")

For this station, the fitted parameters are approximately:

- **Location (μ):** ~2.6 inches - typical annual maximum
- **Scale (σ):** ~0.75 inches - spread of extremes
- **Shape (ξ):** +0.35 - Fréchet type (heavy tail)

The positive shape parameter indicates heavy upper tails where very large
precipitation events are possible but rare. This is typical for precipitation
in the northeastern United States.

Diagnostic Plots
----------------

Validate the fit using diagnostic plots:

.. code-block:: python

   fig, axes = xts.diagnostic_plots(values, **params)
   fig.suptitle('GEV Diagnostic Plots - New Brunswick Precipitation')
   plt.tight_layout()

.. figure:: /_static/newbrunswick_diagnostics.png
   :width: 700px
   :align: center
   :alt: GEV diagnostic plots

   Four-panel diagnostic plots showing goodness-of-fit for the GEV distribution.
   The Q-Q plot and probability plot show good agreement with the fitted distribution.

Testing for Non-Stationarity
----------------------------

Test whether there is a significant trend in precipitation extremes:

.. code-block:: python

   result = xts.likelihood_ratio_test(values, years, trend_in="loc")

   print(f"Test statistic (D): {result['statistic']:.3f}")
   print(f"p-value:            {result['p_value']:.4f}")
   print(f"Significant:        {'Yes' if result['significant'] else 'No'}")
   print(f"Preferred model:    {result['preferred_model']}")

The likelihood ratio test compares:

- **Null hypothesis (H₀):** Stationary GEV (no trend)
- **Alternative hypothesis (H₁):** Non-stationary GEV with trend in location

For this dataset, the test yields:

- **p-value:** ~0.41 (not significant at α=0.05)
- **Preferred model:** Stationary

This indicates **no statistically significant trend** in precipitation extremes
at this station over the 55-year record. The stationary GEV model is appropriate.

.. figure:: /_static/newbrunswick_trend_test.png
   :width: 600px
   :align: center
   :alt: Trend test results

   Model comparison showing AIC values. The stationary model (blue) is preferred
   based on both the likelihood ratio test (p=0.41) and AIC.

Return Levels
-------------

Calculate return levels using the stationary model:

.. code-block:: python

   for rp in [10, 50, 100]:
       rl = xts.return_level(rp, **params)
       print(f"{rp:3d}-year return level: {rl:.2f} inches")

Results for New Brunswick:

==============  ======================
Return Period   Return Level (inches)
==============  ======================
10-year         ~5.2
50-year         ~9.0
100-year        ~11.3
==============  ======================

These values can be used for:

- **Stormwater infrastructure design** - sizing culverts, detention basins
- **Flood risk assessment** - evaluating building vulnerability
- **Climate adaptation planning** - establishing design standards

Non-Stationary Analysis (When Applicable)
-----------------------------------------

If the likelihood ratio test indicates a significant trend (p < 0.05), you
would proceed with non-stationary analysis:

.. code-block:: python

   # Only if lrt_result['significant'] or lrt_result['preferred_model'] == 'nonstationary'
   ns_params = xts.fit_nonstationary_gev(values, years, trend_in="loc")

   print(f"Location intercept (μ₀): {ns_params['loc0']:.2f} inches")
   print(f"Location trend (μ₁):     {ns_params['loc1']:.4f} inches/year")
   print(f"Scale:                   {np.exp(ns_params['scale0']):.2f} inches")
   print(f"Shape:                   {ns_params['shape']:.3f}")

   # Time-varying return levels
   for year in [1970, 1995, 2020]:
       rl = xts.nonstationary_return_level(100, ns_params, year)
       print(f"100-year level in {year}: {rl:.2f} inches")

   # Effective return periods
   eff = xts.effective_return_level(
       ns_params,
       reference_value=years[0],
       future_value=years[-1],
       return_periods=[10, 20, 50, 100]
   )

See the :doc:`atlantic_city_sealevel` example for a dataset that does show
significant non-stationarity.

Summary
-------

This example demonstrated:

1. **Data preparation**: Loading NOAA precipitation observations and extracting
   annual maxima

2. **Stationary analysis**: Fitting a stationary GEV distribution with
   diagnostic validation

3. **Trend detection**: Using the likelihood ratio test to check for significant
   trends in precipitation extremes

4. **Return levels**: Calculating design values for infrastructure planning

Key findings for New Brunswick, NJ precipitation:

- **Shape parameter:** +0.35 (Fréchet type with heavy upper tail)
- **Trend:** Not statistically significant (p=0.41)
- **100-year return level:** ~11.3 inches

The absence of a significant trend in this 55-year record does not mean climate
change is not affecting precipitation. The signal may be:

- Too small to detect with available data
- Obscured by natural variability
- Manifesting in other metrics (frequency, seasonality)

Longer records or regional analyses may reveal trends not detectable at a
single station.

See Also
--------

- :doc:`atlantic_city_sealevel` - Example with significant non-stationarity
- :doc:`model_comparison` - Comparing observations with climate models
- :func:`xtimeseries.fit_gev` - Stationary GEV fitting
- :func:`xtimeseries.likelihood_ratio_test` - Trend significance testing
