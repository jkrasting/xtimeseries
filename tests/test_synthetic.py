"""
Tests for synthetic data generation.
"""

import numpy as np
import pytest
import xarray as xr

import xtimeseries as xts


class TestGenerateGEVSeries:
    """Tests for GEV series generation."""

    def test_generates_correct_length(self):
        """Test output length."""
        data = xts.generate_gev_series(100, loc=30, scale=5, shape=0.1)
        assert len(data) == 100

    def test_reproducible_with_seed(self):
        """Test reproducibility with seed."""
        data1 = xts.generate_gev_series(50, loc=30, scale=5, shape=0.1, seed=42)
        data2 = xts.generate_gev_series(50, loc=30, scale=5, shape=0.1, seed=42)

        np.testing.assert_array_equal(data1, data2)

    def test_parameters_recoverable(self, rng):
        """Test that parameters can be recovered by fitting."""
        loc, scale, shape = 30.0, 5.0, 0.1
        data = xts.generate_gev_series(500, loc=loc, scale=scale, shape=shape, seed=rng)

        params = xts.fit_gev(data)

        assert abs(params["loc"] - loc) < 0.2 * loc
        assert abs(params["scale"] - scale) < 0.3 * scale
        assert abs(params["shape"] - shape) < 0.2

    def test_invalid_scale_raises(self):
        """Test error for invalid scale."""
        with pytest.raises(ValueError, match="positive"):
            xts.generate_gev_series(100, loc=30, scale=-5, shape=0.1)


class TestGenerateGPDSeries:
    """Tests for GPD series generation."""

    def test_generates_positive_exceedances(self):
        """Test that exceedances are positive."""
        data = xts.generate_gpd_series(100, scale=5, shape=0.2)
        assert np.all(data >= 0)

    def test_threshold_added(self):
        """Test that threshold is added to values."""
        threshold = 20.0
        data = xts.generate_gpd_series(100, scale=5, shape=0.2, threshold=threshold)
        assert np.all(data >= threshold)


class TestGenerateNonstationarySeries:
    """Tests for non-stationary series generation."""

    def test_output_structure(self):
        """Test output dictionary structure."""
        result = xts.generate_nonstationary_series(
            n=50, loc_intercept=30, loc_slope=0.1, scale=5, shape=-0.1
        )

        assert "data" in result
        assert "time" in result
        assert "true_loc" in result
        assert "true_params" in result

    def test_trend_in_location(self):
        """Test that location trend is present."""
        result = xts.generate_nonstationary_series(
            n=100, loc_intercept=30, loc_slope=0.1, scale=5, shape=-0.1, seed=42
        )

        # True location should increase over time
        assert result["true_loc"][-1] > result["true_loc"][0]

        # Mean of data in second half should be higher than first half
        first_half = result["data"][:50]
        second_half = result["data"][50:]
        assert np.mean(second_half) > np.mean(first_half) - 5  # Allow some tolerance


class TestGenerateTemperatureLike:
    """Tests for temperature-like time series."""

    def test_returns_dataarray(self):
        """Test that output is xarray DataArray."""
        temp = xts.generate_temperature_like(10, seed=42)
        assert isinstance(temp, xr.DataArray)

    def test_has_time_dimension(self):
        """Test time dimension exists."""
        temp = xts.generate_temperature_like(10)
        assert "time" in temp.dims

    def test_seasonal_cycle(self):
        """Test presence of seasonal cycle."""
        temp = xts.generate_temperature_like(5, seasonal_amplitude=15, seed=42)

        # Summer should be warmer than winter (for standard seasonal pattern)
        summer = temp.sel(time=temp.time.dt.month.isin([6, 7, 8]))
        winter = temp.sel(time=temp.time.dt.month.isin([12, 1, 2]))

        assert summer.mean() > winter.mean()

    def test_trend(self):
        """Test trend is present when specified."""
        temp = xts.generate_temperature_like(
            50, trend_per_decade=0.5, seed=42
        )

        # Fit linear trend
        years = temp.time.dt.year.values + temp.time.dt.dayofyear.values / 365
        annual_mean = temp.groupby(temp.time.dt.year).mean()

        # Trend should be positive
        trend = np.polyfit(np.arange(len(annual_mean)), annual_mean.values, 1)[0]
        assert trend > 0

    def test_noleap_calendar(self):
        """Test noleap calendar support."""
        temp = xts.generate_temperature_like(10, calendar="noleap")

        # Should have exactly 365 days per year
        assert len(temp.time) == 10 * 365

    def test_360_day_calendar(self):
        """Test 360-day calendar support."""
        temp = xts.generate_temperature_like(10, calendar="360_day")
        assert len(temp.time) == 10 * 360


class TestGeneratePrecipitationLike:
    """Tests for precipitation-like time series."""

    def test_has_dry_days(self):
        """Test presence of dry days (zeros)."""
        precip = xts.generate_precipitation_like(10, seed=42)

        n_dry = (precip.values == 0).sum()
        assert n_dry > 0
        assert n_dry < len(precip)  # Not all dry

    def test_wet_days_positive(self):
        """Test wet days have positive values."""
        precip = xts.generate_precipitation_like(10, seed=42)

        wet_values = precip.values[precip.values > 0]
        assert len(wet_values) > 0
        assert np.all(wet_values > 0)

    def test_heavy_tail(self):
        """Test presence of extreme values (heavy tail)."""
        # Generate large sample to see extremes
        precip = xts.generate_precipitation_like(50, seed=42)

        # Maximum should be much larger than mean wet-day amount
        wet_values = precip.values[precip.values > 0]
        assert np.max(wet_values) > 3 * np.mean(wet_values)


class TestGenerateGEVReturnLevels:
    """Tests for analytical return level calculation."""

    def test_known_values(self):
        """Test against known analytical values."""
        from scipy import stats

        loc, scale, shape = 30.0, 5.0, 0.1
        result = xts.generate_gev_return_levels(loc, scale, shape)

        # Compare with direct scipy calculation
        p = 1 - 1 / result["return_periods"]
        expected = stats.genextreme.ppf(p, c=-shape, loc=loc, scale=scale)

        np.testing.assert_allclose(result["return_levels"], expected)

    def test_return_levels_increasing(self):
        """Test that return levels increase with period."""
        result = xts.generate_gev_return_levels(30, 5, 0.1)

        assert np.all(np.diff(result["return_levels"]) > 0)


class TestGenerateTestDataset:
    """Tests for gridded test dataset generation."""

    def test_returns_dataset(self):
        """Test that output is xarray Dataset."""
        ds = xts.generate_test_dataset(n_years=20, nlat=3, nlon=3, seed=42)
        assert isinstance(ds, xr.Dataset)

    def test_has_expected_variables(self):
        """Test for expected variables."""
        ds = xts.generate_test_dataset(n_years=20, seed=42)

        assert "annual_max" in ds
        assert "true_location" in ds

    def test_spatial_gradient(self):
        """Test spatial gradient in location parameter."""
        ds = xts.generate_test_dataset(
            n_years=50, nlat=5, nlon=5, loc_lat_gradient=0.5, seed=42
        )

        # Higher latitudes should have higher location
        loc = ds["true_location"]
        high_lat = float(loc.sel(lat=50, method="nearest").mean())
        low_lat = float(loc.sel(lat=20, method="nearest").mean())
        assert high_lat > low_lat
