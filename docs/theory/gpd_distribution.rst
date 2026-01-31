GPD Distribution
================

The Generalized Pareto Distribution (GPD) is used to model threshold
exceedances in the Peaks Over Threshold (POT) approach to extreme value
analysis.

Mathematical Definition
-----------------------

For exceedances :math:`Y = X - u` above a threshold :math:`u`, the GPD
cumulative distribution function (CDF) is:

.. math::

   F(y; \sigma, \xi) = 1 - \left(1 + \xi\frac{y}{\sigma}\right)^{-1/\xi}

for :math:`y > 0` and :math:`1 + \xi y/\sigma > 0`, with parameters:

- :math:`\sigma > 0`: **Scale** parameter
- :math:`\xi \in \mathbb{R}`: **Shape** parameter (same interpretation as GEV)

For the special case :math:`\xi = 0`, the CDF becomes:

.. math::

   F(y; \sigma, 0) = 1 - \exp\left(-\frac{y}{\sigma}\right)

which is the exponential distribution.

Probability Density Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GPD probability density function (PDF) is:

.. math::

   f(y; \sigma, \xi) = \frac{1}{\sigma}\left(1 + \xi\frac{y}{\sigma}\right)^{-1/\xi - 1}

For :math:`\xi = 0`:

.. math::

   f(y; \sigma, 0) = \frac{1}{\sigma}\exp\left(-\frac{y}{\sigma}\right)

Relationship to GEV
-------------------

If block maxima follow a GEV distribution with parameters :math:`(\mu, \sigma, \xi)`,
then exceedances above a sufficiently high threshold :math:`u` approximately follow
a GPD with:

- **Same shape parameter:** :math:`\xi_{\text{GPD}} = \xi_{\text{GEV}}`
- **Modified scale:** :math:`\sigma_u = \sigma + \xi(u - \mu)`

This relationship is fundamental to EVT and allows cross-validation between
block maxima and POT approaches.

Shape Parameter Interpretation
------------------------------

The shape parameter has the same interpretation as for the GEV:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Shape
     - Type
     - Behavior
   * - :math:`\xi > 0`
     - Pareto tail
     - Heavy tail, exceedances can be arbitrarily large
   * - :math:`\xi = 0`
     - Exponential
     - Light exponential decay
   * - :math:`\xi < 0`
     - Beta tail
     - Bounded above at :math:`y = -\sigma/\xi`

.. figure:: /_static/gpd_shapes.png
   :width: 600px
   :align: center
   :alt: GPD distribution shapes

   The GPD for different shape parameters showing Pareto (ξ > 0),
   Exponential (ξ = 0), and bounded (ξ < 0) tail behavior.

Support of the Distribution
---------------------------

- :math:`\xi \geq 0`: :math:`y \in [0, \infty)` (unbounded above)
- :math:`\xi < 0`: :math:`y \in [0, -\sigma/\xi]` (bounded above)

Moments
-------

The mean of the GPD exists only for :math:`\xi < 1`:

.. math::

   \mathbb{E}[Y] = \frac{\sigma}{1 - \xi}, \quad \xi < 1

The variance exists only for :math:`\xi < 0.5`:

.. math::

   \text{Var}[Y] = \frac{\sigma^2}{(1-\xi)^2(1-2\xi)}, \quad \xi < 0.5

Threshold Selection
-------------------

Choosing an appropriate threshold is critical for POT analysis. The threshold
must be:

1. **High enough** that the GPD approximation is valid
2. **Low enough** to retain sufficient exceedances for reliable estimation

Mean Residual Life Plot
^^^^^^^^^^^^^^^^^^^^^^^

The mean residual life (MRL) function is:

.. math::

   e(u) = \mathbb{E}[X - u | X > u]

For a GPD, this is linear in :math:`u`:

.. math::

   e(u) = \frac{\sigma_u}{1 - \xi} = \frac{\sigma + \xi u}{1 - \xi}

Plot :math:`e(u)` vs. :math:`u` and look for the threshold where it becomes linear.

.. code-block:: python

   import xtimeseries as xts

   # Generate MRL data
   thresholds, mrl, ci = xts.mean_residual_life(data, n_thresholds=50)

   # Plot and look for linearity

Parameter Stability
^^^^^^^^^^^^^^^^^^^

Above the correct threshold, GPD parameters should be stable. Plot :math:`\xi`
and the modified scale :math:`\sigma^* = \sigma - \xi u` against threshold:

.. code-block:: python

   # Check parameter stability
   stability = xts.threshold_stability(data, n_thresholds=30)

   # Parameters should be approximately constant above true threshold

Automated Selection
^^^^^^^^^^^^^^^^^^^

The :func:`xtimeseries.select_threshold` function provides automated methods:

.. code-block:: python

   # Quantile-based (e.g., 95th percentile)
   threshold = xts.select_threshold(data, method="quantile", quantile=0.95)

   # Rate-based (e.g., 3 exceedances per year on average)
   threshold = xts.select_threshold(data, method="rate", rate=3.0)

Declustering
------------

Temporal clustering violates the independence assumption. When exceedances
occur in clusters (e.g., multi-day heat waves), **declustering** extracts
independent events:

.. code-block:: python

   # Extract exceedances with declustering
   pot = xts.peaks_over_threshold(
       data, threshold,
       decluster=True,
       run_length=3  # Minimum separation between clusters
   )

The run-length declustering method groups consecutive exceedances into clusters
and keeps only the maximum from each cluster.

Return Level Calculation
------------------------

For GPD exceedances, the m-observation return level is:

.. math::

   x_m = u + \frac{\sigma}{\xi}\left[(m\zeta_u)^\xi - 1\right]

where :math:`\zeta_u = P(X > u)` is the exceedance probability.

For the T-year return level with :math:`n_y` observations per year:

.. math::

   x_T = u + \frac{\sigma}{\xi}\left[(T n_y \zeta_u)^\xi - 1\right]

.. code-block:: python

   import xtimeseries as xts

   # Fit GPD
   gpd_params = xts.fit_gpd(exceedances, threshold=u)

   # Calculate return level (need exceedance rate)
   rl_100 = xts.return_level_gpd(
       100,  # 100-year return period
       threshold=u,
       exceedance_rate=pot['rate'],  # Exceedances per year
       **gpd_params
   )

Usage in xtimeseries
--------------------

Complete POT workflow:

.. code-block:: python

   import xtimeseries as xts

   # 1. Select threshold
   threshold = xts.select_threshold(data, method="quantile", quantile=0.95)

   # 2. Extract exceedances
   pot = xts.peaks_over_threshold(data, threshold, decluster=True)

   # 3. Fit GPD to exceedances (shifted to start at 0)
   gpd_params = xts.fit_gpd(pot['exceedances'], threshold=0)

   # 4. Calculate return levels
   rl_100 = xts.return_level_gpd(
       100,
       threshold=threshold,
       exceedance_rate=pot['rate'],
       **gpd_params
   )

See Also
--------

- :doc:`extreme_value_theory` - Block maxima vs. POT comparison
- :doc:`gev_distribution` - The related GEV for block maxima
- :func:`xtimeseries.fit_gpd` - API reference for GPD fitting
- :func:`xtimeseries.peaks_over_threshold` - Extract exceedances
