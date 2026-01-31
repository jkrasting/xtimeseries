Storm Surge Extremes at Atlantic City
=====================================

This example demonstrates extreme value analysis of storm surge from the
Atlantic City, NJ tide gauge, one of the longest continuous water level
records in the United States (110+ years).

Introduction
------------

Storm surge is the rise in water level caused by meteorological forcing
(wind, pressure) during storms. By analyzing surge separately from mean
sea level and astronomical tides, we can:

- Isolate the meteorological component of extreme water levels
- Detect changes in storm intensity independent of sea level rise
- Understand how storm risk is evolving over time

**Key scientific distinction:**

- **Observed water level** = Mean sea level + Astronomical tide + Storm surge
- **Storm surge** = Observed water level - Predicted tide

Analyzing surge directly allows us to ask: "Are storms getting more intense?"
rather than conflating storm changes with the well-documented rise in mean
sea level.

Data Source
-----------

**Station:** 8534720 (Atlantic City, NJ)

**Period:** 1911-2023 (110+ years)

**Products:**

- Hourly observed water levels (verified)
- Hourly tide predictions (astronomical)

**Datum:** Mean Sea Level (MSL)

**Source:** NOAA Tides & Currents CO-OPS API

To obtain the data:

.. code-block:: bash

   python scripts/fetch_noaa_tides.py

No API key is required for the NOAA Tides & Currents API.

Setup
-----

.. code-block:: python

   import xtimeseries as xts
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from pathlib import Path

   # Load tide gauge data
   data_file = Path("tests/data/noaa_atlantic_city_tides.csv")
   df = pd.read_csv(data_file, parse_dates=["time"], index_col="time")

Calculating Storm Surge
-----------------------

Storm surge is the difference between observed water level and predicted
astronomical tide:

.. code-block:: python

   # Calculate surge
   if 'surge' not in df.columns:
       df['surge'] = df['observed'] - df['predicted']

   # Extract daily maximum surge
   daily_max_surge = df['surge'].resample('D').max()

   # Extract annual maximum surge
   annual_max_surge = daily_max_surge.resample('YE').max()
   annual_max_surge = annual_max_surge.dropna()

   values = annual_max_surge.values
   years = annual_max_surge.index.year.values

   print(f"Record length: {len(years)} years ({years[0]}-{years[-1]})")
   print(f"Maximum surge: {values.max():.3f} m")

.. figure:: /_static/atlantic_city_surge_record.png
   :width: 700px
   :align: center
   :alt: Annual maximum surge record

   110+ year record of annual maximum storm surge at Atlantic City, NJ.

Understanding Water Level Components
------------------------------------

Visualize how water levels decompose into tide and surge:

.. figure:: /_static/atlantic_city_surge_decomposition.png
   :width: 700px
   :align: center
   :alt: Water level decomposition

   Decomposition of observed water level (top) into astronomical tide
   prediction (middle) and storm surge (bottom) during a storm event.

The surge component captures the meteorological forcing that produces
extreme water levels during storms.

Stationary GEV Fit
------------------

Fit a stationary GEV distribution to annual maximum surge:

.. code-block:: python

   params = xts.fit_gev(values)

   print("GEV Parameters:")
   print(f"  Location (μ): {params['loc']*1000:.1f} mm")
   print(f"  Scale (σ):    {params['scale']*1000:.1f} mm")
   print(f"  Shape (ξ):    {params['shape']:.3f}")

Storm surge extremes typically show positive shape parameters, indicating
heavy upper tails where rare, intense storms can produce very large surges.

.. figure:: /_static/atlantic_city_surge_diagnostics.png
   :width: 700px
   :align: center
   :alt: GEV diagnostic plots

   Diagnostic plots for the GEV fit to annual maximum storm surge.

Testing for Trends in Storm Intensity
-------------------------------------

The key scientific question: Are storm surge extremes changing over time?

.. code-block:: python

   result = xts.likelihood_ratio_test(values, years, trend_in="loc")

   print(f"Test statistic (D): {result['statistic']:.3f}")
   print(f"p-value:            {result['p_value']:.4f}")
   print(f"Significant:        {'Yes' if result['significant'] else 'No'}")

**Interpretation:**

- If p < 0.05, there is evidence that storm surge extremes are changing
- This would suggest storm intensity is evolving, independent of mean sea level
- No trend suggests relatively stable storm climatology over the record

Non-Stationary Analysis
-----------------------

If a trend is detected, fit a non-stationary GEV:

.. code-block:: python

   ns_params = xts.fit_nonstationary_gev(values, years, trend_in="loc")

   print(f"Location intercept: {ns_params['loc0']*1000:.1f} mm")
   print(f"Location trend:     {ns_params['loc1']*1000:.3f} mm/year")
   print(f"                    {ns_params['loc1']*10000:.1f} mm/decade")

.. figure:: /_static/atlantic_city_surge_trend.png
   :width: 600px
   :align: center
   :alt: Non-stationary fit

   Non-stationary GEV fit showing the time-varying location parameter
   with uncertainty bands (if trend is significant).

Return Level Analysis
---------------------

Calculate return levels for storm planning and design:

.. code-block:: python

   return_periods = [10, 50, 100]

   print("Return Levels (Stationary):")
   for T in return_periods:
       rl = xts.return_level(T, **params)
       print(f"  {T:3d}-year: {rl*1000:.0f} mm ({rl:.3f} m)")

For non-stationary analysis, compare return levels at different time points:

.. code-block:: python

   print("\n100-year Return Level Over Time:")
   for year in [1920, 1970, 2020]:
       rl = xts.nonstationary_return_level(100, ns_params, year)
       print(f"  {year}: {rl*1000:.0f} mm")

.. figure:: /_static/atlantic_city_surge_return_levels.png
   :width: 600px
   :align: center
   :alt: Return level curves

   Return level curves for different reference years (if non-stationary)
   compared with the stationary estimate.

Effective Return Periods
------------------------

Understand how storm risk is changing:

.. code-block:: python

   eff = xts.effective_return_level(
       ns_params,
       reference_value=years[0],
       future_value=years[-1],
       return_periods=[10, 20, 50, 100]
   )

   print("How historical storms are changing:")
   for i, rp in enumerate(eff['return_periods']):
       ref_mm = eff['reference_levels'][i] * 1000
       eff_period = eff['effective_period'][i]
       print(f"  {rp}-yr event ({ref_mm:.0f} mm) → {eff_period:.1f}-yr event")

This shows whether a historically rare storm surge would be more or less
common under current conditions.

.. figure:: /_static/atlantic_city_surge_effective.png
   :width: 700px
   :align: center
   :alt: Effective return periods

   How historical return periods have changed between the start and end
   of the record (if trend is significant).

Scientific Context
------------------

**Why analyze surge instead of total water level?**

Total water level at a tide gauge is affected by:

1. **Mean sea level rise** (~3-4 mm/year globally)
2. **Astronomical tides** (predictable lunar/solar cycles)
3. **Storm surge** (meteorological forcing)

By computing surge = observed - predicted tide, we remove the astronomical
component. By analyzing annual maxima of surge, we focus on the
meteorological extreme events.

A trend in surge extremes would indicate changing storm climatology,
separate from the well-documented rise in mean sea level.

**Relationship to hurricane intensity:**

Atlantic City experiences surge from:

- Tropical systems (hurricanes, tropical storms)
- Extratropical storms (nor'easters)

Changes in surge extremes may reflect changes in storm intensity, frequency,
or track patterns.

Summary
-------

This example demonstrated:

1. **Data preparation**: Calculating storm surge from tide gauge observations
   and predictions

2. **Physical interpretation**: Understanding the decomposition of water
   levels into tide and surge components

3. **Stationary analysis**: Fitting GEV to characterize surge extreme
   statistics

4. **Trend detection**: Testing whether storm surge extremes are changing
   independent of mean sea level

5. **Risk assessment**: Computing return levels and effective return periods
   to understand changing storm risk

Key considerations for coastal planning:

- Storm surge analysis complements mean sea level projections
- A 110+ year record provides robust statistical estimates
- Trend detection can inform adaptation planning
- Shape parameter indicates heavy-tailed behavior for rare storms

See Also
--------

- :doc:`trend_analysis` - Detailed trend analysis methodology
- :doc:`newbrunswick_precipitation` - Similar analysis for precipitation
- :func:`xtimeseries.fit_nonstationary_gev` - API reference
- :func:`xtimeseries.effective_return_level` - Changing return periods
