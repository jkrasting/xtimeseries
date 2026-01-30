# xtimeseries

Extreme value analysis toolkit for climate data.

## Features

- **Distribution Fitting**: GEV and GPD with correct climate sign conventions
- **Block Maxima**: Extract annual/seasonal extremes with cftime calendar support
- **Return Periods**: Calculate return levels and return periods
- **Non-Stationary Analysis**: Time-varying GEV parameters for trend detection
- **Peaks Over Threshold**: POT analysis with declustering
- **xarray Integration**: Vectorized operations on gridded data
- **Confidence Intervals**: Bootstrap methods for uncertainty quantification

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import xtimeseries as xts
import numpy as np

# Generate synthetic data
data = xts.generate_gev_series(100, loc=30, scale=5, shape=0.1, seed=42)

# Fit GEV distribution
params = xts.fit_gev(data)
print(f"Location: {params['loc']:.2f}")
print(f"Scale: {params['scale']:.2f}")
print(f"Shape: {params['shape']:.2f}")

# Calculate return levels
rl_100 = xts.return_level(100, **params)
print(f"100-year return level: {rl_100:.2f}")
```

## Convention Note

This package uses the **standard climate convention** for the GEV shape parameter:
- ξ > 0: Fréchet (heavy tail, unbounded above)
- ξ = 0: Gumbel (light exponential tail)
- ξ < 0: Weibull (bounded upper tail)

Note that `scipy.stats.genextreme` uses the opposite sign convention (c = -ξ).
This package handles the conversion automatically.

## License

MIT License - see LICENSE file.
