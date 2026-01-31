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

**Overlapping Period:** 1969-2014

Using a common time period ensures fair comparison across datasets. Since the
NOAA observations begin in 1969 and the CMIP6 historical experiment ends in 2014,
this 46-year overlap provides the basis for comparison.

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

   # Observations (already in inches)
   df_obs = pd.read_csv(DATA_DIR / "noaa_new_brunswick.csv",
                        parse_dates=["date"], index_col="date")
   obs_prcp = df_obs["PRCP"]  # already in inches
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

   Annual maximum precipitation from observations (black) and climate models
   (GFDL-CM4 in blue, CESM2 in orange) over the 1969-2014 analysis period.
   Note the higher interannual variability in the observations compared to models.

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

   Comparison of GEV parameters between observations and climate models.
   Both models underestimate the location (typical extreme magnitude), scale
   (variability), and shape (tail heaviness) parameters relative to observations.

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

   Return level curves for observations and climate models. The observations
   (black) show substantially higher return levels than both models, with the
   divergence increasing at longer return periods due to different tail behavior.
   The scatter points show empirical return periods from the data.

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

   Precipitation trends (inches per decade) for observations and models.
   None of the datasets show statistically significant trends over this period,
   as indicated by the absence of asterisks. This is consistent with the relatively
   short 46-year analysis period.

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

Results
-------

The analysis reveals systematic biases in how climate models represent
precipitation extremes at this location:

**Fitted GEV Parameters:**

===============  ==========  ==========  ==========
Dataset          Location    Scale       Shape
===============  ==========  ==========  ==========
Observations     2.63        0.78        +0.360
GFDL-CM4         2.16        0.45        +0.170
CESM2            2.06        0.40        +0.081
===============  ==========  ==========  ==========

**Model Skill Assessment:**

===============  ==============  ==============  ===============
Model            Location Bias   Scale Bias      100-yr RL Bias
===============  ==============  ==============  ===============
GFDL-CM4         -0.47 (-18%)    -0.33 (-43%)    -6.49 (-55%)
CESM2            -0.57 (-22%)    -0.38 (-48%)    -7.47 (-64%)
===============  ==============  ==============  ===============

Both models substantially underestimate precipitation extremes:

- **Location bias:** Models produce typical annual maxima 18-22% lower than observed
- **Scale bias:** Models show 43-48% less variability in extremes
- **Shape bias:** Models have much lighter tails (ξ ≈ 0.08-0.17 vs observed 0.36)
- **100-year return level:** Models underestimate by 55-64%

The lighter tails in the models mean they fail to capture the most extreme events.
This is a known issue with coarse-resolution climate models, which cannot resolve
convective processes that produce the most intense precipitation.

**Trend Detection:**

===============  ==========  =============  ==============
Dataset          p-value     Significant    Preferred
===============  ==========  =============  ==============
Observations     0.34        No             Stationary
GFDL-CM4         0.73        No             Stationary
CESM2            0.84        No             Stationary
===============  ==========  =============  ==============

None of the datasets show significant trends in precipitation extremes over
the 1969-2014 period. This consistency suggests that both observations and
models agree that trends in annual maximum precipitation are not detectable
at this single location over this time period.

Summary
-------

This example demonstrated:

1. **Data alignment**: Using the common 1969-2014 period for fair comparison

2. **Parameter comparison**: Both CMIP6 models underestimate all three GEV
   parameters, with particularly severe underestimation of the shape parameter

3. **Return level analysis**: Model biases amplify at longer return periods
   due to differences in tail behavior, reaching 55-64% underestimation of
   the 100-year event

4. **Trend comparison**: All datasets agree on the absence of significant
   trends in this period

5. **Implications**: Climate model projections of future precipitation extremes
   may be significantly biased low at local scales

Key considerations for model evaluation:

- Use overlapping periods for fair comparison
- Compare both typical extremes (location) and tail behavior (shape)
- Consider whether biases are consistent across return periods
- Recognize that point-to-grid comparisons are inherently limited by scale mismatch
- Use bias correction methods when applying model projections to local impact studies

See Also
--------

- :doc:`newbrunswick_precipitation` - Detailed analysis of observations
- :doc:`trend_analysis` - Non-stationary analysis methodology
- :doc:`../theory/gev_distribution` - GEV distribution background
