Return Periods and Return Levels
=================================

Return periods and return levels are the primary outputs of extreme value
analysis, providing a standardized way to communicate the severity and
frequency of extreme events.

Definitions
-----------

Return Period
^^^^^^^^^^^^^

The **return period** (or recurrence interval) :math:`T` is the average time
between events exceeding a given magnitude. For example, a "100-year flood"
is expected to be exceeded on average once every 100 years.

More precisely, if :math:`p` is the annual exceedance probability, then:

.. math::

   T = \frac{1}{p}

So a 100-year event has :math:`p = 0.01` (1% annual probability of exceedance).

Return Level
^^^^^^^^^^^^

The **return level** :math:`x_T` is the value that is expected to be exceeded
on average once every :math:`T` years. It is the :math:`(1 - 1/T)` quantile
of the annual maximum distribution.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy import stats

   fig, ax = plt.subplots(figsize=(8, 5))

   # GEV parameters
   loc, scale, shape = 30, 5, 0.1

   # Return periods
   T = np.array([1.5, 2, 5, 10, 20, 50, 100, 200, 500])
   y_p = -np.log(1 - 1 / T)

   # Calculate return levels
   rl = loc + (scale / shape) * (y_p ** (-shape) - 1)

   # Approximate confidence band
   se = scale * 0.15 * np.log(T)
   lower = rl - 1.96 * se
   upper = rl + 1.96 * se

   ax.semilogx(T, rl, "b-", linewidth=2, label="Return level")
   ax.fill_between(T, lower, upper, alpha=0.2, color="blue", label="95% CI")

   # Add observed points
   rng = np.random.default_rng(42)
   n = 50
   obs = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=n, random_state=rng)
   obs.sort()
   plotting_pos = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
   T_obs = 1 / (1 - plotting_pos)
   ax.scatter(T_obs, obs, s=30, c="black", alpha=0.6, zorder=5, label="Observations")

   ax.set_xlabel("Return Period (years)")
   ax.set_ylabel("Return Level")
   ax.set_title("Return Level Plot")
   ax.legend(loc="lower right")
   ax.set_xlim(1, 600)
   ax.grid(True, which="both", alpha=0.3)
   plt.tight_layout()

Return level plotted against return period, showing the typical
logarithmic relationship and confidence bands.

Mathematical Formulas
---------------------

GEV Return Level
^^^^^^^^^^^^^^^^

For a GEV distribution with parameters :math:`(\mu, \sigma, \xi)`:

.. math::

   x_T = \begin{cases}
   \mu + \frac{\sigma}{\xi}\left[y_p^{-\xi} - 1\right] & \text{if } \xi \neq 0 \\
   \mu - \sigma \ln(-\ln(1 - 1/T)) & \text{if } \xi = 0
   \end{cases}

where :math:`y_p = -\ln(1 - 1/T)`.

.. code-block:: python

   import xtimeseries as xts

   # Calculate return level
   rl_100 = xts.return_level(100, loc=30, scale=5, shape=0.1)

GEV Return Period
^^^^^^^^^^^^^^^^^

Given a value :math:`x`, the return period is:

.. math::

   T = \frac{1}{1 - F(x)}

where :math:`F(x)` is the GEV CDF.

.. code-block:: python

   # Calculate return period for a given value
   T = xts.return_period(50, loc=30, scale=5, shape=0.1)

GPD Return Level
^^^^^^^^^^^^^^^^

For POT analysis with GPD parameters :math:`(\sigma, \xi)`, threshold :math:`u`,
and annual exceedance rate :math:`\lambda`:

.. math::

   x_T = \begin{cases}
   u + \frac{\sigma}{\xi}\left[(T\lambda)^\xi - 1\right] & \text{if } \xi \neq 0 \\
   u + \sigma \ln(T\lambda) & \text{if } \xi = 0
   \end{cases}

.. code-block:: python

   # GPD return level
   rl_100 = xts.return_level_gpd(
       100,
       threshold=10,
       exceedance_rate=2.5,  # 2.5 exceedances per year
       scale=3,
       shape=0.1
   )

Interpretation
--------------

Common Misconceptions
^^^^^^^^^^^^^^^^^^^^^

.. warning::

   A "100-year event" does NOT mean:

   - It happens exactly once every 100 years
   - It won't happen again for 100 years after occurring
   - It has never happened in the past 100 years

**Correct interpretation:** In any given year, there is a 1% probability of
exceeding the 100-year return level. Events are independent across years.

Probability in a Time Window
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The probability of experiencing at least one T-year event in n years:

.. math::

   P(\text{at least one exceedance in } n \text{ years}) = 1 - \left(1 - \frac{1}{T}\right)^n

For example, the probability of experiencing a 100-year flood in a 30-year
mortgage period:

.. math::

   P = 1 - (1 - 0.01)^{30} \approx 0.26

There is a 26% chance of experiencing a "100-year flood" over 30 years.

Waiting Time Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^

The expected waiting time until the first exceedance is :math:`T` years, but
the distribution is geometric (or exponential in continuous time):

.. math::

   P(\text{first exceedance in year } k) = \left(1 - \frac{1}{T}\right)^{k-1} \cdot \frac{1}{T}

Confidence Intervals
--------------------

Return level estimates have uncertainty that increases with return period.
The xtimeseries package provides several methods:

Bootstrap
^^^^^^^^^

The most flexible method, making no distributional assumptions:

.. code-block:: python

   ci = xts.bootstrap_ci(
       data,
       return_periods=[10, 50, 100],
       n_bootstrap=1000,
       ci_level=0.95,
       method='percentile'  # or 'basic', 'bca'
   )

   print(f"100-year: {ci['return_levels'][2]:.1f} "
         f"[{ci['lower'][2]:.1f}, {ci['upper'][2]:.1f}]")

Delta Method
^^^^^^^^^^^^

Analytical approximation using the Fisher information matrix:

.. code-block:: python

   rl, lower, upper, se = xts.return_level_with_ci(
       100,
       loc=30, scale=5, shape=0.1,
       data=data,
       ci_level=0.95
   )

Profile Likelihood
^^^^^^^^^^^^^^^^^^

The most accurate method for small samples but computationally intensive.
Not currently implemented in xtimeseries.

Extrapolation Considerations
----------------------------

Reliability
^^^^^^^^^^^

Return level estimates become increasingly uncertain for large return periods:

- **T ≤ n** (within data range): Generally reliable
- **n < T ≤ 2n**: Moderate extrapolation, use with caution
- **T > 2n**: Significant extrapolation, confidence intervals are wide

Where n is the number of block maxima (years of data).

Shape Parameter Sensitivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Return levels are highly sensitive to the shape parameter :math:`\xi`:

.. code-block:: python

   import xtimeseries as xts

   # Same loc and scale, different shape
   print(f"xi=0.0:  RL_100 = {xts.return_level(100, 30, 5, 0.0):.1f}")
   print(f"xi=0.1:  RL_100 = {xts.return_level(100, 30, 5, 0.1):.1f}")
   print(f"xi=0.2:  RL_100 = {xts.return_level(100, 30, 5, 0.2):.1f}")
   print(f"xi=-0.1: RL_100 = {xts.return_level(100, 30, 5, -0.1):.1f}")

Output::

   xi=0.0:  RL_100 = 53.0
   xi=0.1:  RL_100 = 54.9
   xi=0.2:  RL_100 = 57.1
   xi=-0.1: RL_100 = 51.3

For large return periods, shape parameter uncertainty dominates total
uncertainty.

Visualizing Return Levels
-------------------------

The return level plot is the standard way to visualize extreme value fits:

.. code-block:: python

   import xtimeseries as xts
   import matplotlib.pyplot as plt

   # Fit GEV
   params = xts.fit_gev(data)

   # Create return level plot with confidence bands
   fig, ax = xts.return_level_plot(data, **params, ci_level=0.95)
   plt.show()

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy import stats

   fig, ax = plt.subplots(figsize=(8, 5))

   # Generate data and fit
   rng = np.random.default_rng(123)
   loc, scale, shape = 30, 5, 0.1
   data = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=50, random_state=rng)

   # Fit GEV
   c, fit_loc, fit_scale = stats.genextreme.fit(data)
   fit_shape = -c

   # Return level curve
   T = np.logspace(np.log10(1.1), np.log10(500), 100)
   y_p = -np.log(1 - 1 / T)
   rl = fit_loc + (fit_scale / fit_shape) * (y_p ** (-fit_shape) - 1)

   # Confidence bands
   se = fit_scale * 0.15 * np.log(T)
   lower = rl - 1.96 * se
   upper = rl + 1.96 * se

   ax.semilogx(T, rl, "b-", linewidth=2, label="Fitted GEV")
   ax.fill_between(T, lower, upper, alpha=0.2, color="blue", label="95% CI")

   # Observations
   data_sorted = np.sort(data)
   n = len(data_sorted)
   emp_prob = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
   T_obs = 1 / (1 - emp_prob)
   ax.scatter(T_obs, data_sorted, s=40, c="black", alpha=0.7, zorder=5, label="Observations")

   ax.set_xlabel("Return Period (years)")
   ax.set_ylabel("Return Level")
   ax.set_title("Return Level Plot with 95% Confidence Bands")
   ax.legend(loc="lower right")
   ax.set_xlim(1, 600)
   ax.grid(True, which="both", alpha=0.3)
   plt.tight_layout()

Return level plot showing the fitted GEV curve with 95% confidence bands
and observed data as plotting positions.

The x-axis shows return period (often on a log scale), the y-axis shows
return level, with observed data shown as plotting positions.

See Also
--------

- :doc:`gev_distribution` - GEV mathematical background
- :doc:`gpd_distribution` - GPD for threshold exceedances
- :func:`xtimeseries.return_level` - API reference
- :func:`xtimeseries.bootstrap_ci` - Confidence interval estimation
