# Test Data Directory

This directory contains cached test data files for integration testing.

## Data Sources

### NOAA Cooperative Observer Data

**File:** `noaa_new_brunswick.csv` (to be generated)

- **Source:** NOAA Climate Data Online API v2
- **Station:** GHCND:USC00286055 (NEW BRUNSWICK 3 SE, NJ)
- **Variables:** TMAX, TMIN, PRCP (daily)
- **Coordinates:** ~40.49°N, -74.45°W

To fetch this data, run:
```bash
python scripts/fetch_noaa_data.py
```

Requires NOAA API token from: https://www.ncdc.noaa.gov/cdo-web/token

### CMIP6 Model Data

**Files:** `cmip6_*.nc` (to be generated)

- **Source:** Pangeo CMIP6 on Google Cloud
- **Models:** GFDL-ESM4, GFDL-CM4, CESM2
- **Experiment:** historical
- **Variables:** tas, tasmax, tasmin, pr
- **Location:** Point extraction for New Brunswick, NJ

To fetch this data, run:
```bash
python scripts/fetch_cmip6_data.py
```

Requires: `gcsfs`, `intake-esm`, `xarray`, `zarr`

## Data Format

All data files are stored in formats suitable for testing:

- **CSV:** Daily observations with columns: date, variable, value
- **NetCDF:** xarray-compatible with time, lat, lon dimensions

## Usage in Tests

Test fixtures in `conftest.py` automatically load these files when available.
If files are missing, tests use synthetic data with known parameters instead.

## Notes

- Large data files (>10MB) should be added to `.gitignore`
- Keep a small subset of data for CI/CD testing
- Full datasets can be regenerated using the fetch scripts
