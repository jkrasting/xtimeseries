#!/usr/bin/env python
"""
Fetch NOAA tide gauge data from the Tides & Currents CO-OPS API.

This script downloads hourly water level observations and tide predictions
for Atlantic City, NJ, and calculates storm surge (observed - predicted).

Usage:
    python scripts/fetch_noaa_tides.py

No API key is required for the NOAA Tides & Currents API.
"""

import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import numpy as np


# Configuration
STATION_ID = "8534720"  # Atlantic City, NJ
BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
START_YEAR = 1911
END_YEAR = 2023
OUTPUT_FILE = Path(__file__).parent.parent / "tests" / "data" / "noaa_atlantic_city_tides.csv"

# Rate limit delay (seconds between requests)
REQUEST_DELAY = 0.5


def fetch_data(product: str, begin_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetch data from NOAA Tides & Currents API.

    Parameters
    ----------
    product : str
        Data product: 'hourly_height' for observations, 'predictions' for tide predictions.
    begin_date : str
        Start date in YYYYMMDD format.
    end_date : str
        End date in YYYYMMDD format.

    Returns
    -------
    DataFrame or None
        Data with 'time' and 'value' columns, or None if request failed.
    """
    params = {
        "station": STATION_ID,
        "product": product,
        "begin_date": begin_date,
        "end_date": end_date,
        "datum": "MSL",  # Mean Sea Level
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "xtimeseries",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"    Request error: {e}")
        return None
    except ValueError:
        print(f"    JSON decode error")
        return None

    # Check for API errors
    if "error" in data:
        print(f"    API error: {data['error'].get('message', 'Unknown error')}")
        return None

    # Handle different response formats
    if "data" in data:
        records = data["data"]
    elif "predictions" in data:
        records = data["predictions"]
    else:
        print(f"    Unexpected response format")
        return None

    if not records:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Parse time column
    if "t" in df.columns:
        df["time"] = pd.to_datetime(df["t"])
    elif "dateTime" in df.columns:
        df["time"] = pd.to_datetime(df["dateTime"])

    # Parse value column
    if "v" in df.columns:
        df["value"] = pd.to_numeric(df["v"], errors="coerce")
    elif "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df[["time", "value"]].dropna()


def fetch_year_data(year: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Fetch observed water levels and predictions for one year.

    Parameters
    ----------
    year : int
        Year to fetch.

    Returns
    -------
    tuple
        (observations DataFrame, predictions DataFrame) or (None, None).
    """
    begin_date = f"{year}0101"
    end_date = f"{year}1231"

    # Fetch hourly observations
    obs = fetch_data("hourly_height", begin_date, end_date)
    time.sleep(REQUEST_DELAY)

    # Fetch tide predictions
    pred = fetch_data("predictions", begin_date, end_date)
    time.sleep(REQUEST_DELAY)

    return obs, pred


def get_station_metadata() -> dict | None:
    """Get station metadata from API."""
    url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    params = {"id": STATION_ID}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("stations"):
            return data["stations"][0]
    except Exception as e:
        print(f"Error fetching metadata: {e}")

    return None


def main():
    """Main function."""
    print("=" * 60)
    print("NOAA Tides & Currents Data Fetcher")
    print("=" * 60)

    # Get station info
    print(f"\nStation: {STATION_ID} (Atlantic City, NJ)")
    metadata = get_station_metadata()
    if metadata:
        print(f"Name: {metadata.get('name', 'N/A')}")
        print(f"Coordinates: {metadata.get('lat', 'N/A')}°N, {metadata.get('lng', 'N/A')}°W")
        print(f"State: {metadata.get('state', 'N/A')}")

    print(f"\nFetching data from {START_YEAR} to {END_YEAR}...")
    print("This may take a while (rate limited to avoid API overload).\n")

    all_obs = []
    all_pred = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Fetching {year}...", end=" ")

        obs, pred = fetch_year_data(year)

        if obs is not None and len(obs) > 0:
            all_obs.append(obs)
            obs_count = len(obs)
        else:
            obs_count = 0

        if pred is not None and len(pred) > 0:
            all_pred.append(pred)
            pred_count = len(pred)
        else:
            pred_count = 0

        print(f"obs: {obs_count}, pred: {pred_count}")

    # Combine all data
    if not all_obs:
        print("\nNo observation data retrieved!")
        return

    df_obs = pd.concat(all_obs, ignore_index=True)
    df_obs = df_obs.rename(columns={"value": "observed"})
    df_obs = df_obs.set_index("time").sort_index()

    if all_pred:
        df_pred = pd.concat(all_pred, ignore_index=True)
        df_pred = df_pred.rename(columns={"value": "predicted"})
        df_pred = df_pred.set_index("time").sort_index()

        # Merge observations and predictions
        df = df_obs.join(df_pred, how="outer")

        # Calculate storm surge
        df["surge"] = df["observed"] - df["predicted"]
    else:
        print("\nWarning: No prediction data available, surge cannot be calculated.")
        df = df_obs
        df["predicted"] = np.nan
        df["surge"] = np.nan

    # Remove rows where observed is missing
    df = df.dropna(subset=["observed"])

    print(f"\n{'=' * 60}")
    print("Data Summary")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Years of data: {(df.index.max() - df.index.min()).days / 365.25:.1f}")

    print("\nObserved water level (m, MSL datum):")
    print(f"  Min:  {df['observed'].min():.3f}")
    print(f"  Max:  {df['observed'].max():.3f}")
    print(f"  Mean: {df['observed'].mean():.3f}")
    print(f"  Std:  {df['observed'].std():.3f}")

    if not df["surge"].isna().all():
        print("\nStorm surge (m):")
        print(f"  Min:  {df['surge'].min():.3f}")
        print(f"  Max:  {df['surge'].max():.3f}")
        print(f"  Mean: {df['surge'].mean():.3f}")
        print(f"  Std:  {df['surge'].std():.3f}")

        # Identify top surge events
        print("\nTop 10 storm surge events:")
        top_surges = df.nlargest(10, "surge")
        for idx, row in top_surges.iterrows():
            print(f"  {idx.strftime('%Y-%m-%d %H:%M')}: {row['surge']:.3f} m")

    # Save to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE)
    print(f"\nData saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
