"""
Tests for xarray integration functions.
"""

import numpy as np
import pytest
import xarray as xr

import xtimeseries as xts


class TestXrFitGEV:
    """Tests for xr_fit_gev function."""

    def test_returns_dataset(self, gridded_annual_max):
        """Test that output is xarray Dataset."""
        da = gridded_annual_max["annual_max"]
        result = xts.xr_fit_gev(da)

        assert isinstance(result, xr.Dataset)

    def test_has_parameter_variables(self, gridded_annual_max):
        """Test that output has loc, scale, shape variables."""
        da = gridded_annual_max["annual_max"]
        result = xts.xr_fit_gev(da)

        assert "loc" in result
        assert "scale" in result
        assert "shape" in result

    def test_preserves_spatial_dims(self, gridded_annual_max):
        """Test that spatial dimensions are preserved."""
        da = gridded_annual_max["annual_max"]
        result = xts.xr_fit_gev(da)

        assert "lat" in result.dims
        assert "lon" in result.dims
        assert len(result.lat) == len(da.lat)

    def test_recovers_parameters(self, gridded_annual_max, gev_params):
        """Test parameter recovery at a grid point."""
        da = gridded_annual_max["annual_max"]
        result = xts.xr_fit_gev(da)

        # Mean location should be close to base location
        mean_loc = result["loc"].mean().values

        # Allow significant tolerance due to random variability
        assert abs(mean_loc - gev_params["loc"]) < 5

    def test_handles_all_nan(self, rng):
        """Test handling of all-NaN locations."""
        times = xr.date_range("2000", periods=30, freq="YE")
        data = rng.random((30, 3, 3))
        data[:, 1, 1] = np.nan  # All NaN at center

        da = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": [0, 1, 2], "lon": [0, 1, 2]},
        )

        result = xts.xr_fit_gev(da)

        # Center should be NaN
        assert np.isnan(result["loc"].values[1, 1])

        # Other points should not be NaN
        assert not np.isnan(result["loc"].values[0, 0])


class TestXrReturnLevel:
    """Tests for xr_return_level function."""

    def test_returns_dataarray(self, gridded_annual_max):
        """Test output type."""
        da = gridded_annual_max["annual_max"]
        result = xts.xr_return_level(da, [10, 50, 100])

        assert isinstance(result, xr.DataArray)

    def test_has_return_period_dim(self, gridded_annual_max):
        """Test that return_period dimension exists."""
        da = gridded_annual_max["annual_max"]
        result = xts.xr_return_level(da, [10, 50, 100])

        assert "return_period" in result.dims
        assert len(result.return_period) == 3

    def test_return_levels_increase(self, gridded_annual_max):
        """Test that return levels increase with period."""
        da = gridded_annual_max["annual_max"]
        result = xts.xr_return_level(da, [10, 50, 100])

        # At each grid point, 100-year should exceed 10-year
        diff = result.sel(return_period=100) - result.sel(return_period=10)
        assert (diff > 0).all()


class TestXrBlockMaxima:
    """Tests for xr_block_maxima function."""

    def test_wrapper_works(self, daily_temperature_xr):
        """Test that wrapper produces same result as block_maxima."""
        result1 = xts.xr_block_maxima(daily_temperature_xr)
        result2 = xts.block_maxima(daily_temperature_xr)

        xr.testing.assert_equal(result1, result2)


class TestXrFitNonstationaryGEV:
    """Tests for xr_fit_nonstationary_gev function."""

    def test_returns_dataset(self, gridded_annual_max):
        """Test output type."""
        da = gridded_annual_max["annual_max"]
        years = da.time.dt.year.values

        result = xts.xr_fit_nonstationary_gev(da, years)

        assert isinstance(result, xr.Dataset)

    def test_has_trend_coefficients(self, gridded_annual_max):
        """Test that trend coefficients are included."""
        da = gridded_annual_max["annual_max"]
        years = da.time.dt.year.values

        result = xts.xr_fit_nonstationary_gev(da, years)

        assert "loc0" in result
        assert "loc1" in result
        assert "scale" in result
        assert "shape" in result


class TestXrBootstrapCI:
    """Tests for xr_bootstrap_ci function."""

    def test_returns_dataset(self, gridded_annual_max):
        """Test output type."""
        from xtimeseries._xarray import xr_bootstrap_ci

        # Use small grid for speed
        da = gridded_annual_max["annual_max"].isel(lat=slice(0, 2), lon=slice(0, 2))

        result = xr_bootstrap_ci(da, [10, 100], n_bootstrap=50, seed=42)

        assert isinstance(result, xr.Dataset)

    def test_has_ci_bounds(self, gridded_annual_max):
        """Test that CI bounds are included."""
        from xtimeseries._xarray import xr_bootstrap_ci

        da = gridded_annual_max["annual_max"].isel(lat=slice(0, 2), lon=slice(0, 2))

        result = xr_bootstrap_ci(da, [100], n_bootstrap=50)

        assert "return_level" in result
        assert "lower" in result
        assert "upper" in result
        assert "se" in result

    def test_ci_ordering(self, gridded_annual_max):
        """Test that lower < return_level < upper."""
        from xtimeseries._xarray import xr_bootstrap_ci

        da = gridded_annual_max["annual_max"].isel(lat=slice(0, 2), lon=slice(0, 2))

        result = xr_bootstrap_ci(da, [100], n_bootstrap=100)

        # Check ordering (where not NaN)
        valid = ~np.isnan(result["return_level"].values)
        assert np.all(
            result["lower"].values[valid] <= result["return_level"].values[valid]
        )
        assert np.all(
            result["return_level"].values[valid] <= result["upper"].values[valid]
        )
