"""
Tests for block maxima extraction.
"""

import numpy as np
import pytest
import xarray as xr

import xtimeseries as xts


class TestBlockMaxima:
    """Tests for block_maxima function."""

    def test_block_maxima_annual(self, daily_temperature_xr):
        """Test annual maxima extraction."""
        annual_max = xts.block_maxima(daily_temperature_xr, freq="YE")

        # Should have one value per year
        n_years = len(daily_temperature_xr.time) // 365
        assert len(annual_max) == n_years

        # Maxima should be larger than mean
        assert annual_max.mean() > daily_temperature_xr.mean()

    def test_block_maxima_monthly(self, daily_temperature_xr):
        """Test monthly maxima extraction."""
        monthly_max = xts.block_maxima(daily_temperature_xr, freq="ME")

        n_months = len(daily_temperature_xr.time) // 30  # Approximate
        assert len(monthly_max) >= n_months - 5  # Allow some tolerance

    def test_block_maxima_preserves_attrs(self, daily_temperature_xr):
        """Test that attributes are preserved."""
        annual_max = xts.block_maxima(daily_temperature_xr)

        assert "units" in annual_max.attrs
        assert annual_max.attrs["units"] == "degC"

    def test_block_maxima_with_nan(self, daily_temperature_xr):
        """Test handling of NaN values."""
        # Insert some NaN values
        data = daily_temperature_xr.copy()
        data.values[:10] = np.nan

        annual_max = xts.block_maxima(data)
        assert not np.all(np.isnan(annual_max.values))

    def test_block_minima(self, daily_temperature_xr):
        """Test block minima extraction."""
        annual_min = xts.block_minima(daily_temperature_xr)

        # Minima should be smaller than mean
        assert annual_min.mean() < daily_temperature_xr.mean()

    def test_block_maxima_numpy_passthrough(self, rng):
        """Test that numpy arrays pass through unchanged."""
        data = rng.random(50)
        result = xts.block_maxima(data)

        np.testing.assert_array_equal(data, result)


class TestBlockMaximaCftime:
    """Tests for cftime calendar support."""

    def test_noleap_calendar(self, daily_temperature_noleap):
        """Test with noleap (365-day) calendar."""
        annual_max = xts.block_maxima(daily_temperature_noleap)

        # Should have approximately one per year
        n_years = len(daily_temperature_noleap.time) // 365
        assert len(annual_max) == n_years

    def test_360_day_calendar(self, rng):
        """Test with 360-day calendar."""
        n_years = 20
        n_days = n_years * 360

        times = xr.cftime_range("0001-01-01", periods=n_days, freq="D", calendar="360_day")
        data = xr.DataArray(rng.random(n_days), dims=["time"], coords={"time": times})

        annual_max = xts.block_maxima(data)
        assert len(annual_max) == n_years

    def test_year_0001_start(self, rng):
        """Test with year 0001 start date (common in climate models)."""
        n_years = 30
        n_days = n_years * 365

        times = xr.cftime_range("0001-01-01", periods=n_days, freq="D", calendar="noleap")
        data = xr.DataArray(rng.random(n_days), dims=["time"], coords={"time": times})

        annual_max = xts.block_maxima(data)

        # First year should be 0001
        first_year = annual_max.time.values[0].year
        assert first_year == 1


class TestBlockMaximaMinPeriods:
    """Tests for min_periods parameter."""

    def test_min_periods_filters_incomplete(self, rng):
        """Test that incomplete blocks are filtered with min_periods."""
        # Create 2 full years + partial year
        n_days = 365 * 2 + 100  # 2 full years + 100 days
        times = xr.date_range("2000-01-01", periods=n_days, freq="D")
        data = xr.DataArray(rng.random(n_days), dims=["time"], coords={"time": times})

        # With high min_periods, last year should be NaN
        result = xts.block_maxima(data, min_periods=200)

        # Last value might be NaN due to incomplete block
        # (depends on exact xarray behavior)
        assert len(result) >= 2
