"""
Tests for cftime calendar support.

These tests verify that the package correctly handles non-standard
calendars commonly used in climate model output.
"""

import numpy as np
import pytest
import xarray as xr

import xtimeseries as xts


class TestNoleapCalendar:
    """Tests for noleap (365-day) calendar."""

    def test_block_maxima_noleap(self, daily_temperature_noleap):
        """Test block maxima with noleap calendar."""
        annual_max = xts.block_maxima(daily_temperature_noleap)

        # Should have correct number of years
        n_years = len(daily_temperature_noleap.time) // 365
        assert len(annual_max) == n_years

    def test_block_maxima_preserves_calendar(self, daily_temperature_noleap):
        """Test that cftime calendar is preserved."""
        annual_max = xts.block_maxima(daily_temperature_noleap)

        # Time coordinate should still be cftime
        first_time = annual_max.time.values[0]
        assert hasattr(first_time, "calendar")

    def test_fit_gev_with_noleap_data(self, daily_temperature_noleap):
        """Test GEV fitting on noleap calendar data."""
        annual_max = xts.block_maxima(daily_temperature_noleap)
        params = xts.fit_gev(annual_max.values)

        assert not np.isnan(params["loc"])
        assert params["scale"] > 0

    def test_synthetic_noleap(self, rng):
        """Test synthetic data generation with noleap calendar."""
        temp = xts.generate_temperature_like(10, calendar="noleap", seed=rng)

        # Check calendar
        first_time = temp.time.values[0]
        assert hasattr(first_time, "calendar")
        assert "noleap" in str(first_time.calendar) or "365" in str(first_time.calendar)

        # Check correct number of days
        assert len(temp.time) == 10 * 365


class Test360DayCalendar:
    """Tests for 360-day calendar."""

    def test_block_maxima_360day(self, rng):
        """Test block maxima with 360-day calendar."""
        n_years = 20
        n_days = n_years * 360

        times = xr.cftime_range("0001-01-01", periods=n_days, freq="D", calendar="360_day")
        data = xr.DataArray(rng.random(n_days), dims=["time"], coords={"time": times})

        annual_max = xts.block_maxima(data)
        assert len(annual_max) == n_years

    def test_synthetic_360day(self, rng):
        """Test synthetic data with 360-day calendar."""
        temp = xts.generate_temperature_like(10, calendar="360_day", seed=rng)

        assert len(temp.time) == 10 * 360


class TestYear0001Start:
    """Tests for data starting at year 0001 (common in control runs)."""

    def test_block_maxima_year_0001(self, rng):
        """Test block maxima with year 0001 start."""
        n_years = 50
        n_days = n_years * 365

        times = xr.cftime_range("0001-01-01", periods=n_days, freq="D", calendar="noleap")
        data = xr.DataArray(
            rng.random(n_days) + 20,
            dims=["time"],
            coords={"time": times},
        )

        annual_max = xts.block_maxima(data)

        # First year should be 0001
        first_year = annual_max.time.values[0].year
        assert first_year == 1

        # Last year should be 0001 + n_years - 1
        last_year = annual_max.time.values[-1].year
        assert last_year == n_years

    def test_nonstationary_with_year_0001(self, rng):
        """Test non-stationary analysis with year 0001."""
        n_years = 100
        n_days = n_years * 365

        times = xr.cftime_range("0001-01-01", periods=n_days, freq="D", calendar="noleap")

        # Add trend in daily data
        years_frac = np.arange(n_days) / 365
        temp = 20 + 0.02 * years_frac + rng.normal(0, 2, n_days)

        data = xr.DataArray(temp, dims=["time"], coords={"time": times})
        annual_max = xts.block_maxima(data)

        # Extract years for covariate
        years = np.array([t.year for t in annual_max.time.values])

        result = xts.fit_nonstationary_gev(annual_max.values, years)

        # Should detect positive trend
        assert result["loc1"] > 0

    def test_return_level_with_year_0001(self, rng):
        """Test return level calculation with year 0001 data."""
        n_years = 80
        n_days = n_years * 365

        times = xr.cftime_range("0001-01-01", periods=n_days, freq="D", calendar="noleap")
        data = xr.DataArray(rng.random(n_days) + 20, dims=["time"], coords={"time": times})

        annual_max = xts.block_maxima(data)
        params = xts.fit_gev(annual_max.values)
        rl_100 = xts.return_level(100, **params)

        assert rl_100 > annual_max.mean().values


class TestProlepticGregorian:
    """Tests for proleptic_gregorian calendar."""

    def test_block_maxima_proleptic(self, rng):
        """Test with proleptic_gregorian calendar."""
        n_years = 30
        n_days = n_years * 365

        times = xr.cftime_range(
            "0100-01-01", periods=n_days, freq="D", calendar="proleptic_gregorian"
        )
        data = xr.DataArray(rng.random(n_days), dims=["time"], coords={"time": times})

        annual_max = xts.block_maxima(data)

        # Should work without errors
        assert len(annual_max) >= n_years - 1


class TestCalendarMixing:
    """Tests to ensure calendar types are handled correctly."""

    def test_standard_vs_noleap_separate(self, rng):
        """Test that standard and noleap data can be processed separately."""
        n_years = 20

        # Standard calendar
        times_std = xr.date_range("2000-01-01", periods=n_years * 365, freq="D")
        data_std = xr.DataArray(
            rng.random(n_years * 365), dims=["time"], coords={"time": times_std}
        )
        annual_max_std = xts.block_maxima(data_std)

        # Noleap calendar
        times_noleap = xr.cftime_range(
            "0001-01-01", periods=n_years * 365, freq="D", calendar="noleap"
        )
        data_noleap = xr.DataArray(
            rng.random(n_years * 365), dims=["time"], coords={"time": times_noleap}
        )
        annual_max_noleap = xts.block_maxima(data_noleap)

        # Both should have similar number of years
        assert len(annual_max_std) == len(annual_max_noleap)


class TestSeasonalWithCftime:
    """Tests for seasonal operations with cftime."""

    def test_seasonal_block_maxima_noleap(self, rng):
        """Test seasonal maxima with noleap calendar."""
        from xtimeseries.block_maxima import seasonal_block_maxima

        n_years = 30
        n_days = n_years * 365

        times = xr.cftime_range("0001-01-01", periods=n_days, freq="D", calendar="noleap")

        # Create seasonal pattern
        doy = np.array([t.dayofyr for t in times])
        seasonal = 15 * np.sin(2 * np.pi * (doy - 105) / 365)
        data = xr.DataArray(
            15 + seasonal + rng.normal(0, 3, n_days),
            dims=["time"],
            coords={"time": times},
        )

        # Get summer (JJA) maxima
        jja_max = seasonal_block_maxima(data, season="JJA")

        # Summer maxima should be higher on average
        djf_max = seasonal_block_maxima(data, season="DJF")

        assert jja_max.mean() > djf_max.mean()
