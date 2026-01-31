Precipitation IDF Curves
========================

This example demonstrates how to construct Intensity-Duration-Frequency (IDF)
curves for precipitation, a key tool in hydrological design and flood risk
assessment.

Background
----------

IDF curves relate:

- **Intensity**: Precipitation rate (e.g., mm/hour)
- **Duration**: Accumulation period (e.g., 1 hour, 24 hours)
- **Frequency**: Return period (e.g., 10-year, 100-year)

For each duration, a GEV distribution is fit to annual maxima, and return
levels are calculated for various return periods.

Setup
-----

.. code-block:: python

   import xtimeseries as xts
   import numpy as np
   import matplotlib.pyplot as plt

Generating Synthetic Precipitation Data
---------------------------------------

Generate realistic daily precipitation data:

.. code-block:: python

   # Generate 50 years of daily precipitation
   precip_data = xts.generate_precipitation_like(
       n_years=50,
       wet_fraction=0.3,      # 30% of days have precipitation
       mean_wet_day=5.0,      # Mean on wet days (mm)
       shape=0.5,             # Gamma shape (skewness)
       heavy_tail_prob=0.02,  # 2% extreme events
       heavy_tail_scale=3.0,  # Extreme event multiplier
       seed=42
   )

   # Create time coordinate
   import xarray as xr
   times = xr.date_range('1970-01-01', periods=len(precip_data['data']), freq='D')

   daily_precip = xr.DataArray(
       precip_data['data'],
       dims=['time'],
       coords={'time': times},
       attrs={'units': 'mm', 'long_name': 'Daily precipitation'}
   )

   print(f"Mean daily precip: {daily_precip.mean().values:.2f} mm")
   print(f"Max daily precip: {daily_precip.max().values:.1f} mm")
   print(f"Wet days: {(daily_precip > 0).sum().values} ({100*(daily_precip > 0).mean().values:.1f}%)")

Multi-Duration Analysis
-----------------------

For IDF curves, we need annual maxima at multiple durations. With daily data,
we can analyze 1-day, 2-day, 3-day, and 5-day accumulations:

.. code-block:: python

   durations = [1, 2, 3, 5]  # days
   annual_max_by_duration = {}

   for dur in durations:
       if dur == 1:
           data = daily_precip
       else:
           # Rolling sum for multi-day accumulations
           data = daily_precip.rolling(time=dur, center=False).sum()

       # Extract annual maxima
       annual_max = xts.block_maxima(data, freq='YE')
       annual_max_by_duration[dur] = annual_max.values

       print(f"\n{dur}-day annual maxima:")
       print(f"  Mean: {np.nanmean(annual_max.values):.1f} mm")
       print(f"  Max:  {np.nanmax(annual_max.values):.1f} mm")

Fitting GEV at Each Duration
----------------------------

Fit GEV distributions to annual maxima for each duration:

.. code-block:: python

   gev_params = {}

   for dur in durations:
       data = annual_max_by_duration[dur]
       data = data[~np.isnan(data)]  # Remove NaN

       params = xts.fit_gev(data)
       gev_params[dur] = params

       print(f"\n{dur}-day GEV parameters:")
       print(f"  Location: {params['loc']:.2f} mm")
       print(f"  Scale:    {params['scale']:.2f} mm")
       print(f"  Shape:    {params['shape']:.3f}")

Calculating Return Levels
-------------------------

Calculate return levels for various return periods at each duration:

.. code-block:: python

   return_periods = [2, 5, 10, 25, 50, 100]
   idf_table = {}

   print("\nIDF Table (mm)")
   print("=" * 60)
   header = f"{'Duration':>10}"
   for T in return_periods:
       header += f"  {T:>6}yr"
   print(header)
   print("-" * 60)

   for dur in durations:
       params = gev_params[dur]
       levels = xts.return_level(return_periods, **params)

       idf_table[dur] = dict(zip(return_periods, levels))

       row = f"{dur:>7} day"
       for rl in levels:
           row += f"  {rl:>7.1f}"
       print(row)

Converting to Intensity
-----------------------

For standard IDF curves, convert total precipitation to intensity (mm/hour):

.. code-block:: python

   idf_intensity = {}

   print("\nIDF Table (mm/hour)")
   print("=" * 60)
   print(header)
   print("-" * 60)

   for dur in durations:
       hours = dur * 24  # Convert days to hours
       idf_intensity[dur] = {}

       row = f"{hours:>6} hr"
       for T in return_periods:
           intensity = idf_table[dur][T] / hours
           idf_intensity[dur][T] = intensity
           row += f"  {intensity:>7.2f}"
       print(row)

Plotting IDF Curves
-------------------

Create the classic IDF curve plot:

.. code-block:: python

   fig, ax = plt.subplots(figsize=(10, 6))

   colors = plt.cm.viridis(np.linspace(0, 0.8, len(return_periods)))

   for i, T in enumerate(return_periods):
       durations_hr = [d * 24 for d in durations]
       intensities = [idf_intensity[d][T] for d in durations]

       ax.loglog(durations_hr, intensities, 'o-', color=colors[i],
                 label=f'{T}-year', linewidth=2, markersize=8)

   ax.set_xlabel('Duration (hours)')
   ax.set_ylabel('Intensity (mm/hour)')
   ax.set_title('Intensity-Duration-Frequency Curves')
   ax.legend(title='Return Period', loc='upper right')
   ax.grid(True, which='both', alpha=0.3)
   ax.set_xlim(10, 200)

   plt.tight_layout()
   plt.show()

.. figure:: /_static/precipitation_idf.png
   :width: 600px
   :align: center
   :alt: IDF curves

   Intensity-Duration-Frequency curves showing precipitation intensity
   decreasing with duration and increasing with return period.

Confidence Intervals for IDF
----------------------------

Add uncertainty estimates:

.. code-block:: python

   # Bootstrap CI for 100-year, 1-day precipitation
   data_1day = annual_max_by_duration[1]
   data_1day = data_1day[~np.isnan(data_1day)]

   ci = xts.bootstrap_ci(
       data_1day,
       return_periods=[10, 50, 100],
       n_bootstrap=1000,
       ci_level=0.95,
       seed=42
   )

   print("\n1-day precipitation return levels with 95% CI:")
   for i, T in enumerate(ci['return_periods']):
       print(f"  {T:3d}-year: {ci['return_levels'][i]:5.1f} mm "
             f"[{ci['lower'][i]:5.1f}, {ci['upper'][i]:5.1f}]")

Empirical IDF Formula
---------------------

IDF relationships often follow a power law:

.. math::

   I = \frac{a}{(d + b)^c}

where I is intensity, d is duration, and a, b, c are fitted parameters.

.. code-block:: python

   from scipy.optimize import curve_fit

   def idf_formula(d, a, b, c):
       return a / (d + b)**c

   # Fit for 100-year return period
   T = 100
   durations_hr = np.array([d * 24 for d in durations])
   intensities = np.array([idf_intensity[d][T] for d in durations])

   # Fit the IDF formula
   popt, _ = curve_fit(idf_formula, durations_hr, intensities, p0=[100, 10, 0.5])
   a, b, c = popt

   print(f"\n100-year IDF formula: I = {a:.1f} / (d + {b:.1f})^{c:.3f}")

   # Interpolate/extrapolate
   d_range = np.logspace(0.5, 2.5, 50)  # 3 to 300 hours
   i_fitted = idf_formula(d_range, a, b, c)

Design Storm Application
------------------------

Use IDF for hydrological design:

.. code-block:: python

   # Design a culvert for 25-year, 24-hour storm
   T_design = 25
   duration_design = 24  # hours

   # Interpolate from IDF table (1 day = 24 hours)
   params_1day = gev_params[1]
   design_depth = xts.return_level(T_design, **params_1day)
   design_intensity = design_depth / duration_design

   print(f"\nDesign storm ({T_design}-year, {duration_design}-hour):")
   print(f"  Total depth: {design_depth:.1f} mm")
   print(f"  Mean intensity: {design_intensity:.2f} mm/hour")

   # Peak factor (often 1.5-2.5 for convective storms)
   peak_factor = 2.0
   peak_intensity = design_intensity * peak_factor
   print(f"  Peak intensity (factor {peak_factor}): {peak_intensity:.2f} mm/hour")

Climate Change Considerations
-----------------------------

IDF curves may shift under climate change. A common approach is to apply
climate change factors:

.. code-block:: python

   # Example: 10% increase in short-duration, 5% increase in long-duration
   climate_factors = {1: 1.10, 2: 1.08, 3: 1.06, 5: 1.05}

   print("\nClimate-adjusted 100-year return levels:")
   for dur in durations:
       current = idf_table[dur][100]
       future = current * climate_factors[dur]
       print(f"  {dur}-day: {current:.1f} mm → {future:.1f} mm "
             f"(+{100*(climate_factors[dur]-1):.0f}%)")

Summary
-------

This example demonstrated:

1. **Multi-duration analysis**: Calculating rolling accumulations for
   different time windows
2. **GEV fitting**: Fitting separate distributions at each duration
3. **IDF table construction**: Organizing return levels by duration and
   return period
4. **Visualization**: Creating standard IDF curve plots
5. **Engineering applications**: Using IDF for design storm calculations

For operational IDF curves:

- Use sub-daily data (hourly or finer) for short durations
- Include more durations (15 min, 30 min, 1 hr, 2 hr, 6 hr, 12 hr, 24 hr)
- Consider regional frequency analysis for data-sparse locations

See Also
--------

- :doc:`temperature_extremes` - Similar analysis for temperature
- :doc:`../theory/gev_distribution` - GEV background
- :func:`xtimeseries.block_maxima` - Block maxima extraction
