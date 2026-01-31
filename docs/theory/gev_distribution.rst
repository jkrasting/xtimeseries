GEV Distribution
================

The Generalized Extreme Value (GEV) distribution is the limiting distribution
for block maxima and forms the foundation of extreme value analysis.

Mathematical Definition
-----------------------

The GEV cumulative distribution function (CDF) is:

.. math::

   F(x; \mu, \sigma, \xi) = \exp\left\{-\left[1 + \xi\frac{x-\mu}{\sigma}\right]^{-1/\xi}\right\}

for :math:`1 + \xi(x-\mu)/\sigma > 0`, with parameters:

- :math:`\mu \in \mathbb{R}`: **Location** parameter (shifts the distribution)
- :math:`\sigma > 0`: **Scale** parameter (controls spread)
- :math:`\xi \in \mathbb{R}`: **Shape** parameter (determines tail behavior)

For the special case :math:`\xi = 0` (Gumbel), the CDF becomes:

.. math::

   F(x; \mu, \sigma, 0) = \exp\left\{-\exp\left(-\frac{x-\mu}{\sigma}\right)\right\}

Probability Density Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GEV probability density function (PDF) is:

.. math::

   f(x; \mu, \sigma, \xi) = \frac{1}{\sigma}\left[1 + \xi\frac{x-\mu}{\sigma}\right]^{-1/\xi - 1}
   \exp\left\{-\left[1 + \xi\frac{x-\mu}{\sigma}\right]^{-1/\xi}\right\}

For :math:`\xi = 0`:

.. math::

   f(x; \mu, \sigma, 0) = \frac{1}{\sigma}\exp\left(-\frac{x-\mu}{\sigma}\right)
   \exp\left\{-\exp\left(-\frac{x-\mu}{\sigma}\right)\right\}

Shape Parameter Interpretation
------------------------------

The shape parameter :math:`\xi` determines the tail behavior and is the most
important parameter for extrapolation:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Shape
     - Type
     - Behavior
   * - :math:`\xi > 0`
     - Fréchet
     - Heavy tail, unbounded above. Typical for precipitation, wind speeds.
   * - :math:`\xi = 0`
     - Gumbel
     - Light exponential tail. Boundary case.
   * - :math:`\xi < 0`
     - Weibull
     - Bounded upper tail at :math:`\mu - \sigma/\xi`. Typical for temperature maxima.

.. figure:: /_static/gev_shapes.png
   :width: 600px
   :align: center
   :alt: GEV distribution shapes

   The GEV distribution for different shape parameters: Fréchet (ξ > 0),
   Gumbel (ξ = 0), and Weibull (ξ < 0).

Physical Interpretation
^^^^^^^^^^^^^^^^^^^^^^^

**Fréchet (ξ > 0):** Indicates that extreme events can be *much* larger than
typical events. This is common for:

- Daily precipitation (convective storms can produce extreme amounts)
- Wind speeds (hurricanes, severe thunderstorms)
- River discharge (flash floods)

**Gumbel (ξ = 0):** Extremes grow logarithmically with sample size. This is
the boundary case and is often assumed when data is insufficient to
estimate :math:`\xi` reliably.

**Weibull (ξ < 0):** There is a physical upper bound on extremes. This is
common for:

- Maximum temperatures (limited by available energy)
- Some wind speed records (thermodynamic limits)
- Relative humidity (bounded at 100%)

scipy Sign Convention
---------------------

.. warning::

   scipy.stats.genextreme uses a **different sign convention** for the shape
   parameter: ``c = -ξ`` where ξ is the standard climate/statistics convention.

The xtimeseries package handles this conversion automatically. When using
scipy directly, remember:

.. code-block:: python

   from scipy import stats

   # scipy's fit returns c, loc, scale
   c, loc, scale = stats.genextreme.fit(data)

   # Convert to climate convention
   shape = -c  # IMPORTANT!

   # When calling scipy functions, negate shape
   stats.genextreme.ppf(p, c=-shape, loc=loc, scale=scale)

The :func:`xtimeseries.fit_gev` function returns parameters in the standard
climate convention, so you can use them directly with all xtimeseries functions.

Support of the Distribution
---------------------------

The support (valid range of x) depends on the shape parameter:

- :math:`\xi > 0` (Fréchet): :math:`x > \mu - \sigma/\xi` (unbounded above)
- :math:`\xi = 0` (Gumbel): :math:`-\infty < x < \infty`
- :math:`\xi < 0` (Weibull): :math:`x < \mu - \sigma/\xi` (bounded above)

Values outside the support have zero probability density.

Moments
-------

The mean of the GEV distribution exists only for :math:`\xi < 1`:

.. math::

   \mathbb{E}[X] = \begin{cases}
   \mu + \sigma\frac{\Gamma(1-\xi) - 1}{\xi} & \text{if } \xi \neq 0, \xi < 1 \\
   \mu + \sigma\gamma & \text{if } \xi = 0
   \end{cases}

where :math:`\gamma \approx 0.5772` is the Euler-Mascheroni constant.

The variance exists only for :math:`\xi < 0.5`:

.. math::

   \text{Var}[X] = \begin{cases}
   \frac{\sigma^2}{\xi^2}\left[\Gamma(1-2\xi) - \Gamma^2(1-\xi)\right] & \text{if } \xi \neq 0, \xi < 0.5 \\
   \frac{\pi^2\sigma^2}{6} & \text{if } \xi = 0
   \end{cases}

Fitting Methods
---------------

Maximum Likelihood Estimation (MLE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common method, implemented in :func:`xtimeseries.fit_gev`. MLE finds
parameters that maximize:

.. math::

   \mathcal{L}(\mu, \sigma, \xi | x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \mu, \sigma, \xi)

MLE is asymptotically efficient but can be unstable for small samples or
extreme shape parameters.

L-Moments
^^^^^^^^^

L-moments provide robust estimates that are less sensitive to outliers. They
are particularly useful for:

- Small samples (n < 30)
- Data with potential outliers
- Regional frequency analysis

Usage in xtimeseries
--------------------

.. code-block:: python

   import xtimeseries as xts

   # Fit GEV to data
   params = xts.fit_gev(data)

   # Access parameters
   print(f"Location: {params['loc']}")
   print(f"Scale: {params['scale']}")
   print(f"Shape: {params['shape']}")

   # Evaluate CDF and PDF
   p = xts.gev_cdf(50, **params)
   density = xts.gev_pdf(50, **params)

   # Quantile function (inverse CDF)
   x_99 = xts.gev_ppf(0.99, **params)  # 99th percentile

See Also
--------

- :doc:`return_periods` - Using GEV parameters for return level calculations
- :doc:`gpd_distribution` - The related GPD for threshold exceedances
- :func:`xtimeseries.fit_gev` - API reference for GEV fitting
