Installation
============

Requirements
------------

xtimeseries requires Python 3.12 or later.

Core Dependencies
^^^^^^^^^^^^^^^^^

- **numpy** >= 1.26
- **scipy** >= 1.12
- **xarray** >= 2024.1
- **cftime** >= 1.6

Installing from PyPI
--------------------

The simplest way to install xtimeseries is using pip:

.. code-block:: bash

   pip install xtimeseries

Development Installation
------------------------

To install for development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/krasting/xtimeseries.git
   cd xtimeseries
   pip install -e ".[dev]"

This installs the package in editable mode along with testing dependencies
(pytest, pytest-cov).

Optional Dependencies
---------------------

Documentation
^^^^^^^^^^^^^

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"

This installs:

- sphinx >= 7.2
- sphinx-rtd-theme >= 2.0
- numpydoc >= 1.6
- myst-parser >= 2.0
- sphinx-copybutton >= 0.5

Data Fetching
^^^^^^^^^^^^^

For fetching climate data from NOAA and CMIP6 archives:

.. code-block:: bash

   pip install -e ".[data]"

This installs:

- requests >= 2.31
- gcsfs >= 2024.1
- intake-esm >= 2024.1
- zarr >= 2.16

Full Installation
^^^^^^^^^^^^^^^^^

To install all optional dependencies:

.. code-block:: bash

   pip install -e ".[all]"

Verifying the Installation
--------------------------

Verify that xtimeseries is installed correctly:

.. code-block:: python

   import xtimeseries as xts

   # Generate test data
   data = xts.generate_gev_series(100, loc=30, scale=5, shape=0.1, seed=42)

   # Fit GEV distribution
   params = xts.fit_gev(data)
   print(f"loc={params['loc']:.2f}, scale={params['scale']:.2f}, shape={params['shape']:.3f}")

You should see output similar to:

.. code-block:: text

   loc=29.87, scale=4.92, shape=0.102

Running Tests
-------------

To run the test suite:

.. code-block:: bash

   pytest tests/ -v

To run with coverage:

.. code-block:: bash

   pytest tests/ --cov=xtimeseries
