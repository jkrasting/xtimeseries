#!/usr/bin/env python
"""
Fetch NOAA Cooperative Observer data for testing.

This script downloads daily climate data from the NOAA Climate Data Online API
for the New Brunswick, NJ area station.

Usage:
    export NOAA_API_TOKEN="your_token_here"
    python scripts/fetch_noaa_data.py

Get your API token from: https://www.ncdc.noaa.gov/cdo-web/token
"""

import os
import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd


# Configuration
API_TOKEN = os.environ.get("NOAA_API_TOKEN")
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
STATION_ID = "GHCND:USC00286055"  # New Brunswick 3 SE, NJ
DATA_TYPES = ["TMAX", "TMIN", "PRCP"]
START_YEAR = 1950
END_YEAR = 2023
OUTPUT_FILE = Path(__file__).parent.parent / "tests" / "data" / "noaa_new_brunswick.csv"


def make_api_request(endpoint, params=None):
    """Make request to NOAA API with rate limiting."""
    if not API_TOKEN:
        raise ValueError("NOAA_API_TOKEN environment variable not set")

    url = f"{BASE_URL}/{endpoint}"
    headers = {"token": API_TOKEN}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    # Rate limiting: 5 requests per second
    time.sleep(0.2)

    return response.json()


def get_station_info():
    """Get station metadata."""
    result = make_api_request(f"stations/{STATION_ID}")
    print(f"Station: {result.get('name', 'Unknown')}")
    print(f"Location: {result.get('latitude', 'N/A')}, {result.get('longitude', 'N/A')}")
    print(f"Date range: {result.get('mindate', 'N/A')} to {result.get('maxdate', 'N/A')}")
    return result


def fetch_year_data(year):
    """Fetch data for a single year."""
    params = {
        "datasetid": "GHCND",
        "stationid": STATION_ID,
        "datatypeid": ",".join(DATA_TYPES),
        "startdate": f"{year}-01-01",
        "enddate": f"{year}-12-31",
        "units": "standard",
        "limit": 1000,
    }

    all_results = []
    offset = 1

    while True:
        params["offset"] = offset
        try:
            response = make_api_request("data", params)
        except requests.exceptions.HTTPError as e:
            print(f"  Error fetching {year}: {e}")
            break

        if not response or "results" not in response:
            break

        results = response["results"]
        all_results.extend(results)

        metadata = response.get("metadata", {}).get("resultset", {})
        total = metadata.get("count", 0)

        if offset + len(results) > total:
            break

        offset += len(results)

    return all_results


def fetch_all_data():
    """Fetch data for all years."""
    all_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Fetching {year}...")
        year_data = fetch_year_data(year)

        if year_data:
            all_data.extend(year_data)
            print(f"  Retrieved {len(year_data)} records")
        else:
            print(f"  No data available")

    return all_data


def convert_to_dataframe(data):
    """Convert API response to DataFrame."""
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # Pivot to get columns per data type
    df_pivot = df.pivot_table(
        index="date",
        columns="datatype",
        values="value",
        aggfunc="first",
    ).reset_index()

    df_pivot.columns.name = None
    df_pivot = df_pivot.set_index("date")

    return df_pivot


def main():
    """Main function."""
    print("=" * 60)
    print("NOAA COOP Data Fetcher")
    print("=" * 60)

    if not API_TOKEN:
        print("\nERROR: NOAA_API_TOKEN environment variable not set")
        print("Get your token from: https://www.ncdc.noaa.gov/cdo-web/token")
        print("Then run: export NOAA_API_TOKEN='your_token'")
        return

    # Get station info
    print("\nStation Information:")
    get_station_info()

    # Fetch data
    print(f"\nFetching data from {START_YEAR} to {END_YEAR}...")
    data = fetch_all_data()

    if not data:
        print("\nNo data retrieved!")
        return

    # Convert to DataFrame
    df = convert_to_dataframe(data)
    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Summary
    print("\nData summary:")
    print(df.describe())

    # Save to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE)
    print(f"\nData saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
