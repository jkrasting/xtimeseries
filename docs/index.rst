xtimeseries
===========

**xtimeseries** is an xarray-native Python package for extreme value analysis
of climate data. It provides GEV/GPD distribution fitting, return period
calculations, non-stationary analysis, and works seamlessly with cftime calendars.

.. code-block:: python

   import xtimeseries as xts

   # Fit GEV to annual maximum temperatures
   params = xts.fit_gev(annual_max)

   # Calculate 100-year return level
   rl_100 = xts.return_level(100, **params)

Key Features
------------

- **GEV/GPD Distribution Fitting**: Fit extreme value distributions with the
  standard climate sign convention (ξ > 0 for Fréchet, ξ < 0 for Weibull)

- **Block Maxima Extraction**: Extract annual or seasonal maxima/minima from
  xarray DataArrays with full cftime calendar support

- **Return Period Analysis**: Calculate return levels and return periods with
  bootstrap confidence intervals

- **Non-Stationary EVA**: Detect trends in extremes using likelihood ratio
  tests and fit GEV with time-varying parameters

- **Peaks Over Threshold**: GPD fitting for threshold exceedances with
  automated threshold selection

- **xarray Integration**: Vectorized operations on gridded data using
  ``apply_ufunc`` with Dask support

Installation
------------

.. code-block:: bash

   pip install xtimeseries

For development:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/extreme_value_theory
   theory/gev_distribution
   theory/gpd_distribution
   theory/return_periods
   theory/nonstationary_eva

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/temperature_extremes
   examples/precipitation_idf
   examples/trend_analysis

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
