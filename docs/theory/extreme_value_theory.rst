Extreme Value Theory
====================

Extreme Value Theory (EVT) provides the mathematical framework for analyzing
the statistical behavior of extreme events. It answers questions like:
"What is the probability of exceeding a given threshold?" and "What is the
expected maximum over a long period?"

The Fisher-Tippett-Gnedenko Theorem
-----------------------------------

The foundational result of EVT is the **Fisher-Tippett-Gnedenko theorem**,
which states that properly normalized maxima of independent, identically
distributed random variables converge in distribution to one of three types:

.. math::

   \lim_{n \to \infty} P\left(\frac{M_n - b_n}{a_n} \leq x\right) = G(x)

where :math:`M_n = \max(X_1, \ldots, X_n)` and :math:`G(x)` is one of:

1. **Gumbel** (Type I): Light-tailed distributions (e.g., Normal, Exponential)
2. **Fréchet** (Type II): Heavy-tailed distributions (e.g., Pareto, Cauchy)
3. **Weibull** (Type III): Bounded distributions (e.g., Uniform, Beta)

These three types can be unified into the **Generalized Extreme Value (GEV)**
distribution.

Two Approaches to EVA
---------------------

There are two main approaches to extreme value analysis:

Block Maxima
^^^^^^^^^^^^

The **block maxima** approach divides data into blocks (typically years) and
fits the GEV distribution to the maximum value within each block.

**Advantages:**

- Direct interpretation: fitting annual maxima answers questions about annual
  extremes
- Well-established theory
- Simple implementation

**Disadvantages:**

- Discards information (uses only one value per block)
- Requires long records for reliable fitting (typically 30+ years)

**When to use:**

- Questions about annual extremes (e.g., "What is the 100-year flood?")
- Sufficient data length (30+ years of annual maxima)
- Interest in a specific return period framework

Peaks Over Threshold (POT)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The **peaks over threshold** approach models all exceedances above a high
threshold using the Generalized Pareto Distribution (GPD).

**Advantages:**

- More efficient use of data (multiple exceedances per year)
- Can work with shorter records
- Better for very extreme events

**Disadvantages:**

- Requires threshold selection (subjective)
- Must ensure independence of exceedances (declustering)
- More complex interpretation

**When to use:**

- Limited data but need to estimate extreme quantiles
- Interest in tail behavior rather than annual maxima
- Clear physical threshold exists

Relationship Between GEV and GPD
--------------------------------

There is a fundamental connection between the GEV and GPD. If block maxima
follow a GEV distribution, then threshold exceedances follow a GPD with
the **same shape parameter**:

.. math::

   \text{GEV shape } \xi = \text{GPD shape } \xi

This allows the approaches to be combined and cross-validated.

Stationarity Assumption
-----------------------

Classical EVT assumes **stationarity**: the statistical properties of extremes
do not change over time. In climate science, this assumption is often violated
due to:

- Climate change (trends in mean and/or variability)
- Multi-decadal oscillations
- Changes in measurement methods

**Non-stationary EVA** addresses this by allowing GEV parameters to vary with
time or other covariates. The xtimeseries package supports:

- Linear trends in location parameter: :math:`\mu(t) = \mu_0 + \mu_1 t`
- Log-linear trends in scale: :math:`\sigma(t) = \exp(\sigma_0 + \sigma_1 t)`

See :doc:`nonstationary_eva` for details.

Practical Considerations
------------------------

Sample Size
^^^^^^^^^^^

EVA requires sufficient data to reliably estimate the tail. Rules of thumb:

- **Block maxima (GEV):** Minimum 30 blocks (years), ideally 50+
- **POT (GPD):** Minimum 50 exceedances, ideally 100+

With fewer data, shape parameter estimates become unreliable. Consider fixing
:math:`\xi = 0` (Gumbel) for small samples if supported by physical reasoning.

Independence
^^^^^^^^^^^^

EVT assumes independent observations. For block maxima, this is usually
satisfied if blocks are large enough (years). For POT, temporal clustering
requires **declustering** to extract independent exceedances.

Model Checking
^^^^^^^^^^^^^^

Always validate your fits using:

- **Probability plots:** Compare empirical vs. theoretical probabilities
- **Q-Q plots:** Compare empirical vs. theoretical quantiles
- **Return level plots:** Visualize return levels with confidence intervals

The xtimeseries package provides diagnostic plots via :func:`xtimeseries.diagnostic_plots`.

Further Reading
---------------

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Katz, R.W., Parlange, M.B., & Naveau, P. (2002). Statistics of extremes in hydrology. *Advances in Water Resources*, 25(8-12), 1287-1304.
- Cooley, D. (2013). Return periods and return levels under climate change. In *Extremes in a Changing Climate* (pp. 97-114). Springer.
