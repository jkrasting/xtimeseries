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

.. figure:: /_static/return_level_curve.png
   :width: 600px
   :align: center
   :alt: Return level curve

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

.. figure:: /_static/theory_return_level.png
   :width: 600px
   :align: center
   :alt: Return level plot

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
