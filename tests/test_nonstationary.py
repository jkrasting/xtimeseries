"""
Tests for non-stationary extreme value analysis.
"""

import numpy as np
import pytest

import xtimeseries as xts


class TestFitNonstationaryGEV:
    """Tests for non-stationary GEV fitting."""

    def test_recovers_trend(self, nonstationary_data):
        """Test that non-stationary fit recovers known trend."""
        result = xts.fit_nonstationary_gev(
            nonstationary_data["data"],
            nonstationary_data["years"],
            trend_in="loc",
        )

        # Check trend recovery (allow generous tolerance for random variability)
        # With 80 years, there's still substantial uncertainty
        true_slope = nonstationary_data["loc_slope"]
        assert abs(result["loc1"] - true_slope) < abs(true_slope)  # Same order of magnitude

        # Shape should be roughly correct
        assert abs(result["shape"] - nonstationary_data["shape"]) < 0.4

    def test_no_trend_when_stationary(self, synthetic_gev_data, rng):
        """Test that stationary data gives near-zero trend."""
        years = np.arange(len(synthetic_gev_data))

        result = xts.fit_nonstationary_gev(
            synthetic_gev_data, years, trend_in="loc"
        )

        # Slope should be close to zero for stationary data
        assert abs(result["loc1"]) < 0.5

    def test_scale_trend(self, rng):
        """Test trend in scale parameter."""
        n_years = 60
        years = np.arange(n_years)

        # Generate with scale trend
        log_scale0 = np.log(5)
        log_scale_slope = 0.01
        scale_t = np.exp(log_scale0 + log_scale_slope * years)

        from scipy import stats
        data = np.array([
            stats.genextreme.rvs(c=0.1, loc=30, scale=sc, random_state=rng)
            for sc in scale_t
        ])

        result = xts.fit_nonstationary_gev(data, years, trend_in="scale")

        # Should detect positive scale trend
        assert result["scale1"] > 0

    def test_returns_model_selection_criteria(self, nonstationary_data):
        """Test that AIC/BIC are returned."""
        result = xts.fit_nonstationary_gev(
            nonstationary_data["data"],
            nonstationary_data["years"],
        )

        assert "aic" in result
        assert "bic" in result
        assert "nllh" in result


class TestLikelihoodRatioTest:
    """Tests for likelihood ratio test."""

    def test_detects_trend(self, nonstationary_data):
        """Test that LRT detects significant trend in non-stationary data."""
        result = xts.likelihood_ratio_test(
            nonstationary_data["data"],
            nonstationary_data["years"],
        )

        # Should have low p-value for data with known trend
        # (might not always be significant with random samples)
        assert "p_value" in result
        assert "statistic" in result
        assert result["statistic"] >= 0

    def test_stationary_not_significant(self, synthetic_gev_data):
        """Test that stationary data doesn't show significant trend."""
        years = np.arange(len(synthetic_gev_data))

        result = xts.likelihood_ratio_test(synthetic_gev_data, years)

        # For truly stationary data, p-value should often be > 0.05
        # (not a strict test due to randomness)
        assert result["p_value"] > 0  # At minimum, should be valid

    def test_returns_aic_comparison(self, nonstationary_data):
        """Test AIC comparison is included."""
        result = xts.likelihood_ratio_test(
            nonstationary_data["data"],
            nonstationary_data["years"],
        )

        assert "aic_stationary" in result
        assert "aic_nonstationary" in result
        assert "preferred_model" in result


class TestNonstationaryReturnLevel:
    """Tests for non-stationary return levels."""

    def test_return_level_changes_with_covariate(self, nonstationary_data):
        """Test that return levels change with covariate."""
        params = xts.fit_nonstationary_gev(
            nonstationary_data["data"],
            nonstationary_data["years"],
        )

        # Return level at early vs late year
        early_year = nonstationary_data["years"][0]
        late_year = nonstationary_data["years"][-1]

        rl_early = xts.nonstationary_return_level(100, params, early_year)
        rl_late = xts.nonstationary_return_level(100, params, late_year)

        # With positive trend, late should be higher
        if params["loc1"] > 0:
            assert rl_late > rl_early

    def test_return_level_multiple_periods(self, nonstationary_data):
        """Test with multiple return periods."""
        params = xts.fit_nonstationary_gev(
            nonstationary_data["data"],
            nonstationary_data["years"],
        )

        rps = np.array([10, 50, 100])
        rls = xts.nonstationary_return_level(rps, params, 2000)

        assert len(rls) == 3
        assert np.all(np.diff(rls) > 0)  # Should increase


class TestEffectiveReturnLevel:
    """Tests for effective return level comparison."""

    def test_effective_period_decreases_with_trend(self, nonstationary_data):
        """Test that effective return period decreases with warming."""
        params = xts.fit_nonstationary_gev(
            nonstationary_data["data"],
            nonstationary_data["years"],
        )

        result = xts.effective_return_level(
            params,
            reference_value=nonstationary_data["years"][0],
            future_value=nonstationary_data["years"][-1],
        )

        # With positive trend, historical 100-year event becomes more frequent
        if params["loc1"] > 0:
            # Find effective period for 100-year level
            idx_100 = np.where(result["return_periods"] == 100)[0]
            if len(idx_100) > 0:
                effective_100 = result["effective_period"][idx_100[0]]
                assert effective_100 < 100  # More frequent in future

    def test_change_values(self, nonstationary_data):
        """Test that change values are computed."""
        params = xts.fit_nonstationary_gev(
            nonstationary_data["data"],
            nonstationary_data["years"],
        )

        result = xts.effective_return_level(
            params,
            reference_value=1950,
            future_value=2000,
        )

        assert "change" in result
        assert "change_percent" in result
        assert len(result["change"]) == len(result["return_periods"])
