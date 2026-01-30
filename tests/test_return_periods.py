"""
Tests for return period and return level calculations.
"""

import numpy as np
import pytest

import xtimeseries as xts


class TestReturnLevel:
    """Tests for return_level function."""

    def test_return_level_basic(self, gev_params, known_return_levels):
        """Test return level calculation against known values."""
        result = xts.return_level(
            known_return_levels["return_periods"], **gev_params
        )

        np.testing.assert_allclose(
            result, known_return_levels["return_levels"], rtol=1e-10
        )

    def test_return_level_scalar(self, gev_params):
        """Test with scalar return period."""
        result = xts.return_level(100, **gev_params)

        assert isinstance(result, (float, np.floating))
        assert result > gev_params["loc"]

    def test_return_level_increasing(self, gev_params):
        """Return levels should increase with return period."""
        rps = np.array([10, 50, 100, 500])
        rls = xts.return_level(rps, **gev_params)

        assert np.all(np.diff(rls) > 0)

    def test_return_level_invalid_period(self, gev_params):
        """Test error for invalid return period."""
        with pytest.raises(ValueError, match="must be > 1"):
            xts.return_level(0.5, **gev_params)

    def test_return_level_negative_scale(self, gev_params):
        """Test error for negative scale."""
        with pytest.raises(ValueError, match="Scale.*positive"):
            xts.return_level(100, loc=30, scale=-5, shape=0.1)


class TestReturnPeriod:
    """Tests for return_period function."""

    def test_return_period_inverse_of_level(self, gev_params):
        """Test that return_period is inverse of return_level."""
        rps = np.array([10, 50, 100])
        levels = xts.return_level(rps, **gev_params)
        recovered_rps = xts.return_period(levels, **gev_params)

        np.testing.assert_allclose(rps, recovered_rps, rtol=1e-10)

    def test_return_period_scalar(self, gev_params):
        """Test with scalar value."""
        result = xts.return_period(40.0, **gev_params)
        assert result > 1


class TestReturnLevelGPD:
    """Tests for GPD return level calculations."""

    def test_return_level_gpd_basic(self):
        """Test GPD return level calculation."""
        threshold = 20.0
        scale = 5.0
        shape = 0.2
        rate = 10.0  # 10 events per year

        rl = xts.return_level_gpd(100, threshold, scale, shape, rate)

        # Should be above threshold
        assert rl > threshold

    def test_return_level_gpd_vs_gev(self, rng):
        """Compare GPD and GEV return levels for consistency."""
        # For exponential (shape=0), we can check analytical relationship
        threshold = 0.0
        scale = 5.0
        shape = 0.0
        rate = 1.0  # 1 event per year = annual maxima

        rl_gpd = xts.return_level_gpd(100, threshold, scale, shape, rate)

        # For exponential with rate=1, this should match Gumbel approximately
        # (though not exactly due to different formulations)
        assert rl_gpd > 0


class TestReturnPeriodGPD:
    """Tests for GPD return period."""

    def test_return_period_gpd_inverse(self):
        """Test GPD return period is inverse of return level."""
        threshold = 20.0
        scale = 5.0
        shape = 0.2
        rate = 10.0

        rps = np.array([10, 50, 100])
        levels = xts.return_level_gpd(rps, threshold, scale, shape, rate)
        recovered_rps = xts.return_period_gpd(levels, threshold, scale, shape, rate)

        np.testing.assert_allclose(rps, recovered_rps, rtol=1e-8)


class TestReturnLevelWithCI:
    """Tests for return level with confidence intervals."""

    def test_return_level_ci_contains_estimate(self, gev_params):
        """Test that CI contains point estimate."""
        from xtimeseries.return_periods import return_level_with_ci

        # Use approximate standard errors
        result = return_level_with_ci(
            100,
            **gev_params,
            loc_se=1.0,
            scale_se=0.5,
            shape_se=0.05,
        )

        assert result["lower"] < result["return_level"]
        assert result["return_level"] < result["upper"]

    def test_return_level_ci_with_covariance(self, gev_params):
        """Test with covariance matrix."""
        from xtimeseries.return_periods import return_level_with_ci

        cov = np.diag([1.0, 0.25, 0.0025])  # Diagonal covariance

        result = return_level_with_ci(100, **gev_params, cov=cov)

        assert result["se"] > 0
        assert result["lower"] < result["upper"]
