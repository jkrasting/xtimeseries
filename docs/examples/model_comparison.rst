Comparing Climate Model Precipitation
=====================================

This example demonstrates how to compare extreme precipitation between
observations and CMIP6 climate model output, assessing model skill in
reproducing observed extreme value statistics.

Introduction
------------

Climate model evaluation is essential for understanding model biases and
limitations when projecting future changes in extremes. This example compares:

- **Observations:** NOAA station data from New Brunswick, NJ
- **GFDL-CM4:** NOAA Geophysical Fluid Dynamics Laboratory climate model
- **CESM2:** NCAR Community Earth System Model

By comparing GEV parameters, return levels, and trends, we can assess how
well models capture the statistical properties of precipitation extremes.

Data Sources
------------

**Observations:**

- Station: GHCND:USC00286055 (New Brunswick 3 SE, NJ)
- Source: NOAA Climate Data Online

**Climate Models:**

- CMIP6 historical experiment (1850-2014)
- Daily precipitation (pr variable, converted to inches/day)
- Nearest grid point to New Brunswick, NJ

**Overlapping Period:** 1950-2014

Using a common time period ensures fair comparison across datasets.

To obtain the data:

.. code-block:: bash

   # Observations
   export NOAA_API_TOKEN="your_token"
   python scripts/fetch_noaa_data.py

   # Climate models
   pip install gcsfs intake-esm zarr
   python scripts/fetch_cmip6_data.py

Setup
-----

.. code-block:: python

   import xtimeseries as xts
   import numpy as np
   import pandas as pd
   import xarray as xr
   import matplotlib.pyplot as plt
   from pathlib import Path

   DATA_DIR = Path("tests/data")
   START_YEAR = 1950
   END_YEAR = 2014

Loading Data
------------

Load observations and model data, converting to inches/day:

.. code-block:: python

   # Observations (convert mm to inches)
   df_obs = pd.read_csv(DATA_DIR / "noaa_new_brunswick.csv",
                        parse_dates=["date"], index_col="date")
   obs_prcp = df_obs["PRCP"] / 25.4  # mm to inches
   obs_annual = obs_prcp.resample("YE").max()
   obs_annual = obs_annual[(obs_annual.index.year >= START_YEAR) &
                           (obs_annual.index.year <= END_YEAR)]

   # GFDL-CM4 (convert kg m-2 s-1 to inches/day)
   ds_gfdl = xr.open_dataset(DATA_DIR / "cmip6_gfdl-cm4_pr.nc")
   gfdl_pr = ds_gfdl["pr"].squeeze(drop=True) * 86400 / 25.4
   gfdl_annual = gfdl_pr.resample(time="YE").max()

   # CESM2 (convert kg m-2 s-1 to inches/day)
   ds_cesm = xr.open_dataset(DATA_DIR / "cmip6_cesm2_pr.nc")
   cesm_pr = ds_cesm["pr"].squeeze(drop=True) * 86400 / 25.4
   cesm_annual = cesm_pr.resample(time="YE").max()

.. figure:: /_static/model_comparison_timeseries.png
   :width: 700px
   :align: center
   :alt: Time series comparison

   Annual maximum precipitation from observations and climate models
   over the common analysis period.

Fitting GEV Distributions
-------------------------

Fit stationary GEV to each dataset:

.. code-block:: python

   datasets = {
       'Observations': obs_values,
       'GFDL-CM4': gfdl_values,
       'CESM2': cesm_values
   }

   gev_params = {}
   for name, values in datasets.items():
       gev_params[name] = xts.fit_gev(values)

   print(f"{'Dataset':<15} {'Location':>10} {'Scale':>10} {'Shape':>10}")
   print("-" * 50)
   for name, params in gev_params.items():
       print(f"{name:<15} {params['loc']:>10.2f} {params['scale']:>10.2f} "
             f"{params['shape']:>10.3f}")

Parameter Comparison
--------------------

Compare GEV parameters across datasets:

.. figure:: /_static/model_comparison_parameters.png
   :width: 700px
   :align: center
   :alt: Parameter comparison

   Comparison of GEV parameters (location, scale, shape) between
   observations and climate models.

Key aspects to evaluate:

**Location parameter (μ):**
   Represents the typical magnitude of annual maximum precipitation.
   A model with higher location produces more intense typical extremes.

**Scale parameter (σ):**
   Represents the variability of extremes. Higher scale means more
   year-to-year variation in annual maxima.

**Shape parameter (ξ):**
   Controls the tail behavior. For precipitation:

   - ξ > 0: Heavy tail (Fréchet) - rare events can be very large
   - ξ ≈ 0: Light tail (Gumbel)
   - ξ < 0: Bounded tail (Weibull)

Return Level Comparison
-----------------------

Compare return level curves:

.. code-block:: python

   return_periods = np.array([2, 5, 10, 20, 50, 100, 200])

   plt.figure(figsize=(10, 6))
   for name, params in gev_params.items():
       rl = xts.return_level(return_periods, **params)
       plt.semilogx(return_periods, rl, '-o', label=name)

   plt.xlabel('Return Period (years)')
   plt.ylabel('Return Level (inches/day)')
   plt.legend()
   plt.grid(True, alpha=0.3)

.. figure:: /_static/model_comparison_return_levels.png
   :width: 600px
   :align: center
   :alt: Return level comparison

   Return level curves for observations and climate models. Divergence
   at high return periods reflects differences in tail behavior.

Trend Detection
---------------

Test for trends in each dataset:

.. code-block:: python

   print(f"{'Dataset':<15} {'p-value':>10} {'Significant':>12} {'Preferred':>12}")
   print("-" * 55)

   for name, (values, years) in data_dict.items():
       lrt = xts.likelihood_ratio_test(values, years, trend_in='loc')
       sig = 'Yes' if lrt['significant'] else 'No'
       print(f"{name:<15} {lrt['p_value']:>10.4f} {sig:>12} "
             f"{lrt['preferred_model']:>12}")

Compare trend magnitudes where significant:

.. figure:: /_static/model_comparison_trends.png
   :width: 600px
   :align: center
   :alt: Trend comparison

   Precipitation trends (units per decade) for observations and models.
   Asterisks indicate statistically significant trends.

Q-Q Plot Comparison
-------------------

Use Q-Q plots to assess fit quality:

.. code-block:: python

   fig, axes = plt.subplots(1, 3, figsize=(12, 4))

   for ax, (name, values) in zip(axes, datasets.items()):
       params = gev_params[name]
       xts.qq_plot(values, **params, ax=ax)
       ax.set_title(f"Q-Q Plot: {name}")

.. figure:: /_static/model_comparison_qq.png
   :width: 700px
   :align: center
   :alt: Q-Q plot comparison

   Q-Q plots showing GEV fit quality for each dataset.

Model Skill Assessment
----------------------

Quantify model biases relative to observations:

.. code-block:: python

   obs_params = gev_params['Observations']

   for model in ['GFDL-CM4', 'CESM2']:
       model_params = gev_params[model]

       # Location bias
       loc_bias = model_params['loc'] - obs_params['loc']
       loc_bias_pct = 100 * loc_bias / obs_params['loc']

       # Scale bias
       scale_bias = model_params['scale'] - obs_params['scale']

       # 100-year return level bias
       obs_rl100 = xts.return_level(100, **obs_params)
       model_rl100 = xts.return_level(100, **model_params)
       rl100_bias = model_rl100 - obs_rl100

       print(f"\n{model} vs Observations:")
       print(f"  Location bias:   {loc_bias:+.2f} ({loc_bias_pct:+.1f}%)")
       print(f"  Scale bias:      {scale_bias:+.2f}")
       print(f"  100-yr RL bias:  {rl100_bias:+.2f}")

Interpretation Guidelines
-------------------------

**Positive location bias:**
   Model produces more intense typical extremes than observed.
   May indicate the model runs "too wet."

**Negative scale bias:**
   Model has less variability in extremes than observed.
   May indicate the model doesn't capture the full range of extreme events.

**Shape parameter differences:**
   Different tail behavior affects high return level estimates.
   A more positive shape means the model allows larger rare events.

**Trend differences:**
   If observations show a significant trend but models don't (or vice versa),
   this indicates disagreement about how extremes are changing.

Summary
-------

This example demonstrated:

1. **Data alignment**: Using a common time period for fair comparison

2. **Parameter comparison**: Comparing GEV location, scale, and shape
   parameters between observations and models

3. **Return level analysis**: Comparing return level curves to assess
   model performance across different recurrence intervals

4. **Trend comparison**: Testing whether observations and models agree
   on the presence and magnitude of trends

5. **Skill metrics**: Quantifying model biases in key statistics

Key considerations for model evaluation:

- Use overlapping periods for fair comparison
- Compare both typical extremes (location) and tail behavior (shape)
- Consider whether biases are consistent across return periods
- Evaluate trend agreement, not just mean climatology

See Also
--------

- :doc:`newbrunswick_precipitation` - Detailed analysis of observations
- :doc:`trend_analysis` - Non-stationary analysis methodology
- :doc:`../theory/gev_distribution` - GEV distribution background
