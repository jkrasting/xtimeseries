"""
Tests for distribution fitting functions.
"""

import numpy as np
import pytest
from scipy import stats

import xtimeseries as xts


class TestFitGEV:
    """Tests for fit_gev function."""

    def test_fit_gev_recovers_parameters(self, synthetic_gev_data, gev_params):
        """Verify fitted parameters are close to true values."""
        result = xts.fit_gev(synthetic_gev_data)

        # Allow 20% tolerance for random sample
        assert abs(result["loc"] - gev_params["loc"]) < 0.2 * abs(gev_params["loc"])
        assert abs(result["scale"] - gev_params["scale"]) < 0.3 * gev_params["scale"]
        assert abs(result["shape"] - gev_params["shape"]) < 0.3

    def test_fit_gev_weibull_type(self, rng, gev_params_weibull):
        """Test fitting with negative shape (Weibull type)."""
        data = stats.genextreme.rvs(
            c=-gev_params_weibull["shape"],
            loc=gev_params_weibull["loc"],
            scale=gev_params_weibull["scale"],
            size=100,
            random_state=rng,
        )
        result = xts.fit_gev(data)

        # Shape should be negative
        assert result["shape"] < 0.1

    def test_fit_gev_gumbel_type(self, rng, gev_params_gumbel):
        """Test fitting Gumbel (shape near 0)."""
        data = stats.gumbel_r.rvs(
            loc=gev_params_gumbel["loc"],
            scale=gev_params_gumbel["scale"],
            size=100,
            random_state=rng,
        )
        result = xts.fit_gev(data)

        # Shape should be close to 0
        assert abs(result["shape"]) < 0.3

    def test_fit_gev_handles_nan(self, synthetic_gev_data):
        """Verify NaN values are handled correctly."""
        data_with_nan = synthetic_gev_data.copy()
        data_with_nan[:5] = np.nan

        result = xts.fit_gev(data_with_nan)
        assert not np.isnan(result["loc"])
        assert not np.isnan(result["scale"])
        assert not np.isnan(result["shape"])

    def test_fit_gev_requires_minimum_data(self):
        """Test error for insufficient data."""
        with pytest.raises(ValueError, match="at least 3"):
            xts.fit_gev([1, 2])

    def test_fit_gev_method_mom(self, synthetic_gev_data, gev_params):
        """Test method of moments fitting."""
        result = xts.fit_gev(synthetic_gev_data, method="mom")

        # Should produce reasonable estimates
        assert not np.isnan(result["loc"])
        assert result["scale"] > 0


class TestFitGPD:
    """Tests for fit_gpd function."""

    def test_fit_gpd_recovers_parameters(self, synthetic_gpd_data):
        """Verify GPD fitting recovers true parameters."""
        result = xts.fit_gpd(synthetic_gpd_data["data"])

        # Check recovery within tolerance
        assert abs(result["shape"] - synthetic_gpd_data["shape"]) < 0.2
        assert abs(result["scale"] - synthetic_gpd_data["scale"]) < 0.3 * synthetic_gpd_data["scale"]

    def test_fit_gpd_with_threshold(self, rng):
        """Test GPD fitting with explicit threshold."""
        # Generate data above threshold
        threshold = 20.0
        exceedances = stats.genpareto.rvs(c=0.1, scale=5, size=100, random_state=rng)
        values = exceedances + threshold

        result = xts.fit_gpd(values, threshold=threshold)
        assert result["threshold"] == threshold
        assert result["scale"] > 0


class TestGEVFunctions:
    """Tests for GEV CDF, PDF, PPF functions."""

    def test_gev_cdf_values(self, gev_params):
        """Test GEV CDF returns valid probabilities."""
        x = np.linspace(20, 50, 100)
        cdf = xts.gev_cdf(x, **gev_params)

        assert np.all(cdf >= 0)
        assert np.all(cdf <= 1)
        assert np.all(np.diff(cdf) >= 0)  # Monotonically increasing

    def test_gev_ppf_inverse_of_cdf(self, gev_params):
        """Test that PPF is inverse of CDF."""
        p = np.array([0.1, 0.5, 0.9, 0.99])
        x = xts.gev_ppf(p, **gev_params)
        p_recovered = xts.gev_cdf(x, **gev_params)

        np.testing.assert_allclose(p, p_recovered)

    def test_gev_pdf_integrates_to_one(self, gev_params):
        """Test PDF integrates to approximately 1."""
        from scipy import integrate

        result, _ = integrate.quad(
            lambda x: xts.gev_pdf(x, **gev_params),
            -np.inf,
            np.inf,
        )
        assert abs(result - 1.0) < 0.01


class TestGPDFunctions:
    """Tests for GPD functions."""

    def test_gpd_cdf_values(self):
        """Test GPD CDF returns valid probabilities."""
        x = np.linspace(0, 20, 50)
        cdf = xts.gpd_cdf(x, scale=5.0, shape=0.1)

        assert np.all(cdf >= 0)
        assert np.all(cdf <= 1)

    def test_gpd_ppf_inverse_of_cdf(self):
        """Test GPD PPF is inverse of CDF."""
        scale, shape = 5.0, 0.1
        p = np.array([0.1, 0.5, 0.9])
        x = xts.gpd_ppf(p, scale=scale, shape=shape)
        p_recovered = xts.gpd_cdf(x, scale=scale, shape=shape)

        np.testing.assert_allclose(p, p_recovered)


class TestSignConvention:
    """Tests to verify scipy sign convention is handled correctly."""

    def test_shape_convention_frechet(self, rng):
        """Verify positive shape gives heavy (Frechet) tail."""
        # Frechet: heavy tail, values can be very large
        shape = 0.3
        data = xts.generate_gev_series(1000, loc=0, scale=1, shape=shape, seed=rng)

        # Frechet should have some very large values
        assert np.max(data) > 5  # Typical for heavy tail

    def test_shape_convention_weibull(self, rng):
        """Verify negative shape gives bounded (Weibull) tail."""
        # Weibull: bounded upper tail
        shape = -0.3
        loc, scale = 0, 1
        data = xts.generate_gev_series(1000, loc=loc, scale=scale, shape=shape, seed=rng)

        # Upper bound for Weibull is loc - scale/shape
        upper_bound = loc - scale / shape
        assert np.max(data) < upper_bound + 0.1
