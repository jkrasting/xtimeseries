"""
Pytest configuration and fixtures for xtimeseries tests.

Provides synthetic data with known parameters for validating algorithms.
"""

import numpy as np
import pytest
import xarray as xr
from scipy import stats


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def gev_params():
    """Known GEV parameters for testing."""
    return {"loc": 30.0, "scale": 5.0, "shape": 0.1}


@pytest.fixture
def gev_params_weibull():
    """GEV with negative shape (Weibull type)."""
    return {"loc": 25.0, "scale": 4.0, "shape": -0.15}


@pytest.fixture
def gev_params_gumbel():
    """Gumbel distribution (GEV with shape=0)."""
    return {"loc": 30.0, "scale": 5.0, "shape": 0.0}


@pytest.fixture
def synthetic_gev_data(rng, gev_params):
    """Generate synthetic GEV data with known parameters."""
    n = 100
    data = stats.genextreme.rvs(
        c=-gev_params["shape"],
        loc=gev_params["loc"],
        scale=gev_params["scale"],
        size=n,
        random_state=rng,
    )
    return data


@pytest.fixture
def synthetic_gev_data_small(rng, gev_params):
    """Small sample for edge case testing."""
    n = 30
    return stats.genextreme.rvs(
        c=-gev_params["shape"],
        loc=gev_params["loc"],
        scale=gev_params["scale"],
        size=n,
        random_state=rng,
    )


@pytest.fixture
def synthetic_gpd_data(rng):
    """Generate synthetic GPD exceedances."""
    n = 200
    scale = 5.0
    shape = 0.2
    return {
        "data": stats.genpareto.rvs(c=shape, scale=scale, size=n, random_state=rng),
        "scale": scale,
        "shape": shape,
    }


@pytest.fixture
def daily_temperature_xr(rng):
    """Generate synthetic daily temperature xarray DataArray."""
    n_years = 30
    n_days = n_years * 365

    times = xr.date_range("1990-01-01", periods=n_days, freq="D")
    doy = np.array([t.timetuple().tm_yday for t in times.to_pydatetime()])

    # Seasonal cycle + random noise
    seasonal = 15.0 * np.sin(2 * np.pi * (doy - 105) / 365)
    noise = rng.normal(0, 3, n_days)
    temp = 15.0 + seasonal + noise

    da = xr.DataArray(
        temp,
        dims=["time"],
        coords={"time": times},
        name="temperature",
        attrs={"units": "degC"},
    )
    return da


@pytest.fixture
def daily_temperature_noleap(rng):
    """Generate temperature data with noleap calendar."""
    n_years = 30
    n_days = n_years * 365

    times = xr.cftime_range("0001-01-01", periods=n_days, freq="D", calendar="noleap")
    doy = np.array([t.dayofyr for t in times])

    seasonal = 15.0 * np.sin(2 * np.pi * (doy - 105) / 365)
    noise = rng.normal(0, 3, n_days)
    temp = 15.0 + seasonal + noise

    da = xr.DataArray(
        temp,
        dims=["time"],
        coords={"time": times},
        name="temperature",
        attrs={"units": "degC", "calendar": "noleap"},
    )
    return da


@pytest.fixture
def gridded_annual_max(rng, gev_params):
    """Generate gridded annual maxima data."""
    n_years = 50
    nlat = 5
    nlon = 5

    times = xr.date_range("1970-01-01", periods=n_years, freq="YE")
    lats = np.linspace(30, 50, nlat)
    lons = np.linspace(-100, -70, nlon)

    # Spatially varying location
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    loc_spatial = gev_params["loc"] + 0.3 * (lat_grid - lats.mean())

    data = np.zeros((n_years, nlat, nlon))
    for i in range(nlat):
        for j in range(nlon):
            data[:, i, j] = stats.genextreme.rvs(
                c=-gev_params["shape"],
                loc=loc_spatial[i, j],
                scale=gev_params["scale"],
                size=n_years,
                random_state=rng,
            )

    ds = xr.Dataset(
        {"annual_max": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds


@pytest.fixture
def nonstationary_data(rng):
    """Generate non-stationary GEV data with known trend."""
    n_years = 80
    years = np.arange(1940, 1940 + n_years)

    # True parameters
    loc_intercept = 28.0
    loc_slope = 0.05  # 0.05 per year = 0.5 per decade
    scale = 5.0
    shape = -0.1

    loc_t = loc_intercept + loc_slope * (years - years[0])

    data = np.array([
        stats.genextreme.rvs(c=-shape, loc=loc, scale=scale, random_state=rng)
        for loc in loc_t
    ])

    return {
        "data": data,
        "years": years,
        "loc_intercept": loc_intercept,
        "loc_slope": loc_slope,
        "scale": scale,
        "shape": shape,
    }


@pytest.fixture
def precipitation_data(rng):
    """Generate synthetic precipitation with wet/dry days."""
    n_days = 365 * 20

    # Simple wet/dry simulation
    is_wet = rng.random(n_days) < 0.3
    precip = np.zeros(n_days)
    precip[is_wet] = stats.gamma.rvs(a=0.7, scale=5, size=is_wet.sum(), random_state=rng)

    return precip


# Known return levels for validation
@pytest.fixture
def known_return_levels(gev_params):
    """Calculate true return levels for known parameters."""
    return_periods = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    p = 1 - 1 / return_periods
    return_levels = stats.genextreme.ppf(
        p, c=-gev_params["shape"], loc=gev_params["loc"], scale=gev_params["scale"]
    )
    return {"return_periods": return_periods, "return_levels": return_levels}
