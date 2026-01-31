#!/usr/bin/env python
"""
Create synthetic data files for testing examples.

This script generates realistic synthetic data that mimics the format
of the real NOAA and CMIP6 data files, allowing examples to run
without requiring API access.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "data"


def create_noaa_precipitation():
    """Create synthetic NOAA precipitation data for New Brunswick."""
    print("Creating synthetic NOAA precipitation data...")

    # Generate daily data from 1950-2023
    dates = pd.date_range("1950-01-01", "2023-12-31", freq="D")
    n = len(dates)

    rng = np.random.default_rng(42)

    # Generate precipitation with realistic properties:
    # - Most days have zero or low precipitation
    # - Occasional heavy precipitation events
    # - Seasonal pattern (more in summer)

    # Base precipitation (exponential distribution)
    base_prcp = rng.exponential(scale=3.0, size=n)

    # Add seasonal pattern
    day_of_year = dates.dayofyear.values
    seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 100) / 365)

    # Apply seasonal modulation
    prcp = base_prcp * seasonal

    # Add some extreme events
    extreme_mask = rng.random(n) < 0.005  # ~0.5% extreme events
    prcp[extreme_mask] *= rng.uniform(3, 8, size=extreme_mask.sum())

    # Add a significant trend (increasing precipitation)
    years_since_start = (dates - dates[0]).days / 365.25
    trend = 1.0 + 0.005 * years_since_start  # 0.5% per year - enough to be significant
    prcp *= trend

    # Make ~60% of days have zero/trace precipitation
    zero_mask = rng.random(n) < 0.60
    prcp[zero_mask] = 0.0

    # Round to 0.1 mm
    prcp = np.round(prcp, 1)

    # Generate temperature data (Celsius)
    # Seasonal cycle + random variability
    tmax_base = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    tmax = tmax_base + rng.normal(0, 3, n)
    tmax = np.round(tmax, 1)

    tmin = tmax - rng.uniform(5, 15, n)
    tmin = np.round(tmin, 1)

    # Create DataFrame
    df = pd.DataFrame({
        "PRCP": prcp,
        "TMAX": tmax,
        "TMIN": tmin,
    }, index=dates)
    df.index.name = "date"

    # Save
    output_file = OUTPUT_DIR / "noaa_new_brunswick.csv"
    df.to_csv(output_file)
    print(f"  Saved: {output_file}")
    print(f"  Records: {len(df)}")
    print(f"  Max precip: {df['PRCP'].max():.1f} mm")

    return df


def create_cmip6_data(model_name: str, bias: float = 0.0, trend: float = 0.0):
    """Create synthetic CMIP6 precipitation data in native CMIP6 units."""
    print(f"Creating synthetic {model_name} precipitation data...")

    # Generate daily data from 1850-2014 (full CMIP6 historical period)
    times = pd.date_range("1850-01-01", "2014-12-31", freq="D")
    n = len(times)

    # Use different seed for each model
    seed = hash(model_name) % 2**32
    rng = np.random.default_rng(seed)

    # Generate precipitation with GEV-like extremes
    # Base: exponential distribution (in mm/day equivalent)
    base_prcp = rng.exponential(scale=4.0 + bias, size=n)

    # Seasonal pattern
    day_of_year = times.dayofyear.values
    seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (day_of_year - 100) / 365)

    prcp = base_prcp * seasonal

    # Add extreme events (models often have fewer extremes)
    extreme_mask = rng.random(n) < 0.003
    prcp[extreme_mask] *= rng.uniform(2.5, 6, size=extreme_mask.sum())

    # Add trend
    years_since_start = (times - times[0]).days / 365.25
    prcp *= (1.0 + trend * years_since_start)

    # Dry days
    zero_mask = rng.random(n) < 0.55
    prcp[zero_mask] = 0.0

    # Convert from mm/day to kg m-2 s-1 (CMIP6 native units)
    # 1 mm/day = 1 kg/m2/day = 1/86400 kg/m2/s
    prcp_flux = prcp / 86400.0

    # Create xarray Dataset with CMIP6-like structure
    ds = xr.Dataset(
        {
            "pr": xr.DataArray(
                prcp_flux[np.newaxis, np.newaxis, :],  # Add member_id, dcpp_init_year dims
                dims=["member_id", "dcpp_init_year", "time"],
                coords={
                    "member_id": ["r1i1p1f1"],
                    "dcpp_init_year": [np.nan],
                    "time": times,
                },
                attrs={
                    "units": "kg m-2 s-1",
                    "long_name": "Precipitation",
                    "standard_name": "precipitation_flux",
                },
            )
        },
        attrs={
            "source_model": model_name,
            "experiment": "historical",
            "activity_id": "CMIP",
        },
    )

    # Add lat/lon as scalar coordinates
    ds = ds.assign_coords(lat=40.5, lon=285.625)

    # Save
    output_file = OUTPUT_DIR / f"cmip6_{model_name.lower()}_pr.nc"
    ds.to_netcdf(output_file)
    print(f"  Saved: {output_file}")
    print(f"  Records: {len(ds.time)}")

    # Report max in inches/day for comparison
    max_inches = ds['pr'].max().values * 86400 / 25.4
    print(f"  Max precip: {max_inches:.2f} inches/day")

    return ds


def create_tide_gauge_data():
    """Create synthetic tide gauge data for Atlantic City."""
    print("Creating synthetic tide gauge data...")

    # Generate hourly data from 1911-2023 (112 years)
    times = pd.date_range("1911-01-01", "2023-12-31 23:00:00", freq="h")
    n = len(times)

    rng = np.random.default_rng(12345)

    # Generate tidal predictions (M2, S2, K1, O1 constituents)
    hours = np.arange(n)

    # Main tidal constituents (simplified)
    M2_period = 12.42  # hours (principal lunar)
    S2_period = 12.00  # hours (principal solar)
    K1_period = 23.93  # hours (lunisolar diurnal)
    O1_period = 25.82  # hours (lunar diurnal)

    predicted = (
        0.6 * np.sin(2 * np.pi * hours / M2_period) +
        0.2 * np.sin(2 * np.pi * hours / S2_period) +
        0.15 * np.sin(2 * np.pi * hours / K1_period) +
        0.1 * np.sin(2 * np.pi * hours / O1_period)
    )

    # Add spring-neap cycle (14-day modulation)
    spring_neap = 1.0 + 0.2 * np.cos(2 * np.pi * hours / (14 * 24))
    predicted *= spring_neap

    # Generate storm surge
    # Most of the time, surge is small random noise
    surge = rng.normal(0, 0.05, n)

    # Add occasional storm events (more frequent in fall/winter)
    month = times.month.values
    storm_prob = np.where((month >= 9) | (month <= 3), 0.0002, 0.0001)
    storm_mask = rng.random(n) < storm_prob

    # Generate storm surges with GEV-like distribution
    storm_magnitudes = rng.gumbel(loc=0.3, scale=0.15, size=storm_mask.sum())
    storm_magnitudes = np.maximum(storm_magnitudes, 0.1)
    surge[storm_mask] = storm_magnitudes

    # Storms last multiple hours - smooth with rolling window
    surge_series = pd.Series(surge, index=times)
    surge_smoothed = surge_series.rolling(6, center=True, min_periods=1).max()
    surge = surge_smoothed.values

    # Add a slight trend (increasing surge extremes over time)
    years_since_start = (times - times[0]).total_seconds().values / (365.25 * 24 * 3600)
    trend_factor = 1.0 + 0.001 * years_since_start  # 0.1% per year
    surge = surge * trend_factor

    # Add mean sea level rise (~3mm/year relative to MSL datum)
    msl_rise = 0.003 * years_since_start  # meters

    # Observed = predicted + surge + MSL rise
    observed = predicted + surge + msl_rise

    # Add small measurement noise
    observed += rng.normal(0, 0.01, n)

    # Create DataFrame
    df = pd.DataFrame({
        "observed": np.round(observed, 4),
        "predicted": np.round(predicted, 4),
        "surge": np.round(surge, 4),
    }, index=times)
    df.index.name = "time"

    # Add some missing data (realistic)
    missing_mask = rng.random(n) < 0.001  # 0.1% missing
    df.loc[missing_mask, ["observed", "surge"]] = np.nan

    # Save
    output_file = OUTPUT_DIR / "noaa_atlantic_city_tides.csv"
    df.to_csv(output_file)
    print(f"  Saved: {output_file}")
    print(f"  Records: {len(df)}")
    print(f"  Max surge: {df['surge'].max():.3f} m")

    # Show top surge events
    print("\n  Top 5 surge events:")
    top_surges = df.nlargest(5, "surge")
    for idx, row in top_surges.iterrows():
        print(f"    {idx.strftime('%Y-%m-%d %H:%M')}: {row['surge']:.3f} m")

    return df


def main():
    """Create all synthetic data files."""
    print("=" * 60)
    print("Creating Synthetic Data Files")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create NOAA precipitation data
    print("\n")
    create_noaa_precipitation()

    # Create CMIP6 model data
    print("\n")
    create_cmip6_data("GFDL-CM4", bias=1.0, trend=0.002)  # Slightly wet bias

    print("\n")
    create_cmip6_data("CESM2", bias=-0.5, trend=0.001)  # Slightly dry bias

    # Create tide gauge data
    print("\n")
    create_tide_gauge_data()

    print("\n" + "=" * 60)
    print("All synthetic data files created!")
    print("=" * 60)

    # List files
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size / 1024
        if size > 1024:
            size_str = f"{size/1024:.1f} MB"
        else:
            size_str = f"{size:.0f} KB"
        print(f"  {f.name}: {size_str}")


if __name__ == "__main__":
    main()
