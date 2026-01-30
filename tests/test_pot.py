"""
Tests for peaks-over-threshold analysis.
"""

import numpy as np
import pytest

import xtimeseries as xts


class TestPeaksOverThreshold:
    """Tests for POT extraction."""

    def test_extracts_exceedances(self, rng):
        """Test that exceedances are correctly extracted."""
        data = rng.exponential(scale=10, size=1000)
        threshold = 20.0

        result = xts.peaks_over_threshold(data, threshold, decluster=False)

        # All values should be above threshold
        assert np.all(result["values"] > threshold)

        # Exceedances should be positive
        assert np.all(result["exceedances"] > 0)

        # Exceedances = values - threshold
        np.testing.assert_allclose(
            result["exceedances"], result["values"] - threshold
        )

    def test_returns_correct_structure(self, rng):
        """Test output dictionary structure."""
        data = rng.exponential(scale=10, size=1000)
        result = xts.peaks_over_threshold(data, threshold=15.0)

        assert "exceedances" in result
        assert "values" in result
        assert "indices" in result
        assert "threshold" in result
        assert "n_exceedances" in result
        assert "rate" in result

    def test_declustering_reduces_count(self, rng):
        """Test that declustering reduces number of exceedances."""
        # Create clustered exceedances
        data = np.zeros(1000)
        # Add clusters of high values
        data[100:105] = rng.uniform(50, 60, 5)
        data[300:310] = rng.uniform(50, 65, 10)
        data[500:503] = rng.uniform(50, 55, 3)

        threshold = 40.0

        result_no_decluster = xts.peaks_over_threshold(
            data, threshold, decluster=False
        )
        result_decluster = xts.peaks_over_threshold(
            data, threshold, decluster=True
        )

        # Declustered should have fewer exceedances
        assert result_decluster["n_exceedances"] < result_no_decluster["n_exceedances"]

    def test_empty_exceedances(self, rng):
        """Test behavior when no exceedances exist."""
        data = rng.uniform(0, 10, 100)
        result = xts.peaks_over_threshold(data, threshold=100.0)

        assert result["n_exceedances"] == 0
        assert len(result["exceedances"]) == 0


class TestDecluster:
    """Tests for declustering function."""

    def test_keeps_cluster_maximum(self):
        """Test that cluster maximum is kept."""
        indices = np.array([10, 11, 12, 50, 100, 101])
        values = np.array([25, 28, 26, 30, 22, 24])

        result = xts.decluster(indices, values, run_length=3)

        # Should keep indices 11 (max of first cluster), 50, 101
        assert len(result["indices"]) == 3
        assert 11 in result["indices"]
        assert 50 in result["indices"]
        assert 101 in result["indices"]

    def test_single_value_unchanged(self):
        """Test single value passes through."""
        result = xts.decluster([10], [50], run_length=3)

        assert len(result["indices"]) == 1
        assert result["values"][0] == 50


class TestMeanResidualLife:
    """Tests for mean residual life plot data."""

    def test_output_structure(self, rng):
        """Test output dictionary structure."""
        data = rng.exponential(scale=10, size=500)
        result = xts.mean_residual_life(data)

        assert "thresholds" in result
        assert "mean_excess" in result
        assert "n_exceed" in result
        assert "se" in result

    def test_mean_excess_decreases_for_exp(self, rng):
        """For exponential, mean excess should be constant."""
        # For exponential distribution, mean residual life is constant
        scale = 10.0
        data = rng.exponential(scale=scale, size=2000)

        result = xts.mean_residual_life(data, n_thresholds=20)

        # Filter to reasonable thresholds
        valid = result["n_exceed"] > 50
        mean_excess = result["mean_excess"][valid]

        # Should be roughly constant (equal to scale parameter)
        assert np.std(mean_excess) < 3  # Allow some variation


class TestThresholdStability:
    """Tests for parameter stability plot data."""

    def test_output_structure(self, rng):
        """Test output structure."""
        data = rng.exponential(scale=10, size=500)
        result = xts.threshold_stability(data)

        assert "thresholds" in result
        assert "shape" in result
        assert "scale" in result
        assert "modified_scale" in result

    def test_shape_stable_for_gpd_data(self, rng):
        """Test shape stability for true GPD data."""
        from scipy import stats

        # Generate from GPD
        shape = 0.2
        scale = 5.0
        data = stats.genpareto.rvs(c=shape, scale=scale, size=1000, random_state=rng)
        # Shift to make threshold selection meaningful
        data = data + 10

        result = xts.threshold_stability(data, n_thresholds=15)

        # Shape should be relatively stable
        valid = result["n_exceed"] > 50
        shapes = result["shape"][valid]
        shapes = shapes[~np.isnan(shapes)]

        if len(shapes) > 3:
            assert np.std(shapes) < 0.5  # Reasonable stability


class TestSelectThreshold:
    """Tests for threshold selection."""

    def test_quantile_method(self, rng):
        """Test quantile-based selection."""
        data = rng.random(1000)

        threshold_90 = xts.select_threshold(data, method="quantile", quantile=0.90)
        threshold_95 = xts.select_threshold(data, method="quantile", quantile=0.95)

        # 95th percentile should be higher
        assert threshold_95 > threshold_90

        # Should be close to actual percentiles
        assert abs(threshold_90 - np.percentile(data, 90)) < 0.01

    def test_rate_method(self, rng):
        """Test rate-based selection."""
        n_years = 20
        obs_per_year = 365
        data = rng.exponential(scale=10, size=n_years * obs_per_year)

        # Target 10 events per year
        threshold = xts.select_threshold(
            data,
            method="rate",
            n_per_year=10,
            observations_per_year=obs_per_year,
        )

        # Check approximately 10 events per year
        n_exceed = (data > threshold).sum()
        events_per_year = n_exceed / n_years

        assert 5 < events_per_year < 20  # Reasonable range
