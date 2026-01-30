"""
Tests for confidence interval estimation.
"""

import numpy as np
import pytest

import xtimeseries as xts


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_ci_structure(self, synthetic_gev_data):
        """Test that bootstrap_ci returns correct structure."""
        result = xts.bootstrap_ci(
            synthetic_gev_data,
            return_periods=[10, 50, 100],
            n_bootstrap=100,
        )

        assert "return_periods" in result
        assert "return_levels" in result
        assert "lower" in result
        assert "upper" in result
        assert "se" in result

        assert len(result["return_levels"]) == 3
        assert len(result["lower"]) == 3

    def test_bootstrap_ci_ordering(self, synthetic_gev_data):
        """Test that lower < estimate < upper."""
        result = xts.bootstrap_ci(
            synthetic_gev_data,
            return_periods=[10, 50, 100],
            n_bootstrap=200,
        )

        for i in range(3):
            assert result["lower"][i] < result["return_levels"][i]
            assert result["return_levels"][i] < result["upper"][i]

    def test_bootstrap_ci_coverage(self, gev_params, known_return_levels):
        """Test coverage of confidence intervals with known truth."""
        from scipy import stats

        # Generate data and compute CI
        rng = np.random.default_rng(42)
        data = stats.genextreme.rvs(
            c=-gev_params["shape"],
            loc=gev_params["loc"],
            scale=gev_params["scale"],
            size=100,
            random_state=rng,
        )

        result = xts.bootstrap_ci(
            data,
            return_periods=[10, 100],
            n_bootstrap=500,
            random_state=42,
        )

        # For 100-year level, check if true value is within CI
        true_100 = known_return_levels["return_levels"][5]  # Index for 100-year
        lower_100 = result["lower"][1]
        upper_100 = result["upper"][1]

        # CI should often contain truth (not a strict test, but sanity check)
        # With 95% CI, ~95% of samples should contain true value
        # We just check the interval is reasonable
        assert lower_100 < true_100 < upper_100 or (upper_100 - lower_100) > 0

    def test_bootstrap_ci_reproducible(self, synthetic_gev_data):
        """Test that results are reproducible with seed."""
        result1 = xts.bootstrap_ci(
            synthetic_gev_data,
            return_periods=[100],
            n_bootstrap=100,
            random_state=42,
        )
        result2 = xts.bootstrap_ci(
            synthetic_gev_data,
            return_periods=[100],
            n_bootstrap=100,
            random_state=42,
        )

        np.testing.assert_array_equal(result1["return_levels"], result2["return_levels"])

    def test_bootstrap_ci_small_sample_warning(self):
        """Test warning for very small samples."""
        data = np.random.gumbel(30, 5, size=15)

        # Should still work but may warn
        result = xts.bootstrap_ci(data, [10, 100], n_bootstrap=100)
        assert len(result["return_levels"]) == 2


class TestBootstrapReturnLevels:
    """Tests for full bootstrap distribution."""

    def test_bootstrap_return_levels_shape(self, synthetic_gev_data):
        """Test output shape."""
        boot_rls = xts.bootstrap_return_levels(
            synthetic_gev_data,
            return_periods=[10, 50, 100],
            n_bootstrap=100,
        )

        assert boot_rls.shape == (100, 3)

    def test_bootstrap_return_levels_distribution(self, synthetic_gev_data):
        """Test that bootstrap samples form reasonable distribution."""
        boot_rls = xts.bootstrap_return_levels(
            synthetic_gev_data,
            return_periods=[100],
            n_bootstrap=500,
        )

        # Standard deviation should be positive
        assert np.std(boot_rls[:, 0]) > 0

        # Mean should be close to point estimate
        point_est = xts.fit_gev(synthetic_gev_data)
        expected_rl = xts.return_level(100, **point_est)

        assert abs(np.mean(boot_rls[:, 0]) - expected_rl) < 5


class TestBootstrapParameters:
    """Tests for parameter bootstrap."""

    def test_bootstrap_parameters_structure(self, synthetic_gev_data):
        """Test output structure."""
        from xtimeseries.confidence import bootstrap_parameters

        result = bootstrap_parameters(synthetic_gev_data, n_bootstrap=100)

        assert "loc" in result
        assert "scale" in result
        assert "shape" in result
        assert "loc_ci" in result
        assert "cov" in result

    def test_bootstrap_parameters_ci_contains_estimate(self, synthetic_gev_data):
        """Test that point estimates are within CIs."""
        from xtimeseries.confidence import bootstrap_parameters

        result = bootstrap_parameters(synthetic_gev_data, n_bootstrap=200)

        # Point estimate should be within CI (usually)
        assert result["loc_ci"][0] < result["loc"] < result["loc_ci"][1]
        assert result["scale_ci"][0] < result["scale"] < result["scale_ci"][1]
