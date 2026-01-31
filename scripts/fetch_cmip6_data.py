#!/usr/bin/env python
"""
Fetch CMIP6 model data from Google Cloud for testing.

This script extracts time series for New Brunswick, NJ from CMIP6 models
available on Google Cloud Public Datasets.

Usage:
    python scripts/fetch_cmip6_data.py

Requires: gcsfs, intake-esm, xarray, zarr
"""

from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd


# Configuration
TARGET_LAT = 40.49    # New Brunswick, NJ
TARGET_LON = -74.45
MODELS = ["GFDL-CM4", "CESM2"]
VARIABLES = ["tas", "tasmax", "pr"]
EXPERIMENT = "historical"
OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "data"


def load_cmip6_catalog():
    """Load the Pangeo CMIP6 catalog."""
    try:
        import intake
    except ImportError:
        print("ERROR: intake-esm not installed")
        print("Install with: pip install intake-esm gcsfs zarr")
        return None

    catalog_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    return intake.open_esm_datastore(catalog_url)


def extract_point_timeseries(ds, var_name, lat, lon):
    """Extract time series at nearest valid (non-NaN) grid point."""
    # Handle longitude convention
    lon_values = ds["lon"].values
    if lon_values.min() >= 0 and lon_values.max() > 180:
        lon_query = lon + 360 if lon < 0 else lon
    else:
        lon_query = lon

    # Select nearest point
    da = ds[var_name].sel(lat=lat, lon=lon_query, method="nearest")

    # Check if data is all NaN (ocean point)
    if np.all(np.isnan(da.values)):
        print(f"  Nearest point is masked (ocean/invalid). Searching for nearest land point...")

        # Search in a region around the target
        lat_slice = slice(lat - 2, lat + 2)
        lon_slice = slice(lon_query - 2, lon_query + 2)

        # Get regional subset
        regional = ds[var_name].sel(lat=lat_slice, lon=lon_slice)

        # Find valid points (check first time step)
        sample = regional.isel(time=0).values
        valid_mask = ~np.isnan(sample)

        if not np.any(valid_mask):
            print(f"  WARNING: No valid land points found in search region!")
            # Return the NaN data anyway
            actual_lat = float(da["lat"].values)
            actual_lon = float(da["lon"].values)
            print(f"  Nearest grid point: ({actual_lat:.2f}, {actual_lon:.2f})")
            return da

        # Find the closest valid point
        lat_idx, lon_idx = np.where(valid_mask)
        regional_lats = regional["lat"].values
        regional_lons = regional["lon"].values

        min_dist = float("inf")
        best_lat, best_lon = None, None

        for i, j in zip(lat_idx, lon_idx):
            dist = (regional_lats[i] - lat) ** 2 + (regional_lons[j] - lon_query) ** 2
            if dist < min_dist:
                min_dist = dist
                best_lat = regional_lats[i]
                best_lon = regional_lons[j]

        # Select the valid point
        da = ds[var_name].sel(lat=best_lat, lon=best_lon)
        print(f"  Found valid point at: ({best_lat:.2f}, {best_lon:.2f})")
    else:
        # Get actual coordinates
        actual_lat = float(da["lat"].values)
        actual_lon = float(da["lon"].values)
        print(f"  Nearest grid point: ({actual_lat:.2f}, {actual_lon:.2f})")

    return da


def convert_units(da, var_name):
    """Convert to user-friendly units."""
    if var_name in ["tas", "tasmax", "tasmin"]:
        # Kelvin to Celsius
        da = da - 273.15
        da.attrs["units"] = "degC"
    elif var_name == "pr":
        # kg m-2 s-1 to mm/day
        da = da * 86400
        da.attrs["units"] = "mm/day"
    return da


def fetch_model_data(catalog, model, variable):
    """Fetch data for a specific model and variable."""
    print(f"\nSearching for {model} {variable}...")

    query = catalog.search(
        source_id=model,
        experiment_id=EXPERIMENT,
        table_id="day",
        variable_id=variable,
        member_id="r1i1p1f1",
    )

    if len(query.df) == 0:
        print(f"  No data found")
        return None

    print(f"  Found {len(query.df)} dataset(s)")

    # Get zarr store path
    zstore = query.df["zstore"].values[0]
    print(f"  Loading from: {zstore[:60]}...")

    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="anon")
        mapper = fs.get_mapper(zstore)
        ds = xr.open_zarr(mapper, consolidated=True)
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None

    # Extract point
    da = extract_point_timeseries(ds, variable, TARGET_LAT, TARGET_LON)

    # Convert units
    da = convert_units(da, variable)

    return da


def main():
    """Main function."""
    print("=" * 60)
    print("CMIP6 Data Fetcher")
    print("=" * 60)
    print(f"Target location: {TARGET_LAT}°N, {TARGET_LON}°E")

    # Load catalog
    print("\nLoading CMIP6 catalog...")
    catalog = load_cmip6_catalog()

    if catalog is None:
        return

    print(f"Catalog contains {len(catalog.df)} datasets")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch data for each model and variable
    for model in MODELS:
        for variable in VARIABLES:
            da = fetch_model_data(catalog, model, variable)

            if da is None:
                continue

            # Save to NetCDF
            output_file = OUTPUT_DIR / f"cmip6_{model.lower()}_{variable}.nc"

            # Convert to dataset for saving
            ds_out = da.to_dataset(name=variable)
            ds_out.attrs["source_model"] = model
            ds_out.attrs["experiment"] = EXPERIMENT
            ds_out.attrs["target_lat"] = TARGET_LAT
            ds_out.attrs["target_lon"] = TARGET_LON

            ds_out.to_netcdf(output_file)
            print(f"  Saved to: {output_file}")

            # Also save summary stats
            print(f"  Time range: {da.time.values[0]} to {da.time.values[-1]}")
            print(f"  Mean: {da.mean().values:.2f} {da.attrs.get('units', '')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
