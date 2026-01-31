# Test Data Directory

This directory contains cached test data files for integration testing.

## Data Sources

### NOAA Cooperative Observer Data

**File:** `noaa_new_brunswick.csv`

- **Source:** NOAA Climate Data Online API v2
- **Station:** GHCND:USC00286055 (NEW BRUNSWICK 3 SE, NJ)
- **Variables:** TMAX, TMIN (°F), PRCP (inches) - daily values
- **Coordinates:** ~40.49°N, -74.45°W
- **Period:** 1969-2023

To fetch this data, run:
```bash
python scripts/fetch_noaa_data.py
```

Requires NOAA API token from: https://www.ncdc.noaa.gov/cdo-web/token

### CMIP6 Model Data

**Files:** `cmip6_gfdl-cm4_pr.nc`, `cmip6_cesm2_pr.nc`

- **Source:** CMIP6 historical experiment
- **Models:** GFDL-CM4, CESM2
- **Experiment:** historical
- **Period:** 1850-2014
- **Variables:** pr (precipitation flux)
- **Units:** kg m-2 s-1 (CMIP6 native units)
- **Location:** Point extraction near Kennebunkport, ME (40.5°N, 285.625°E)

**Unit conversion in scripts:**
The data files contain native CMIP6 precipitation flux units. Scripts convert
to inches/day for analysis:

```python
# Convert kg m-2 s-1 to inches/day
# 1 kg/m2 = 1 mm of water, 86400 seconds/day, 25.4 mm/inch
pr_inches = pr * 86400 / 25.4
```

To fetch updated data, run:
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
