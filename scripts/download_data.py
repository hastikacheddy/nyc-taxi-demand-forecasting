"""
Fetch NYC Yellow Taxi trip data (public TLC open data) and build the raw events
file the ingestion pipeline aggregates.

    python scripts/download_data.py --start 2024-01 --end 2024-04

Downloads one Parquet per month, keeps only the pickup timestamp, filters to the
trip month (the files contain a few stray dates), concatenates, and writes
data/raw/yellow_tripdata.parquet. Then run the ingestion (or run_pipeline.py) to
produce data/{daily,hourly}_demand.csv.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page  (open data)
"""
import argparse
import os
import urllib.request

import pandas as pd

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"
PICKUP = "tpep_pickup_datetime"
RAW_DIR = os.path.join("data", "raw")


def months(start: str, end: str):
    return [str(p) for p in pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download NYC taxi data and build the raw events file")
    ap.add_argument("--start", default="2024-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2024-04", help="last month, YYYY-MM")
    args = ap.parse_args(argv)

    os.makedirs(RAW_DIR, exist_ok=True)
    frames = []
    for ym in months(args.start, args.end):
        path = os.path.join(RAW_DIR, f"yellow_tripdata_{ym}.parquet")
        if not os.path.exists(path):
            print(f"downloading {ym} ...")
            urllib.request.urlretrieve(URL.format(ym=ym), path)
        df = pd.read_parquet(path, columns=[PICKUP]).rename(columns={PICKUP: "TimePeriod"})
        lo = pd.Timestamp(f"{ym}-01")
        hi = lo + pd.offsets.MonthBegin(1)
        df = df[(df["TimePeriod"] >= lo) & (df["TimePeriod"] < hi)]
        print(f"  {ym}: {len(df):,} trips")
        frames.append(df)

    trips = pd.concat(frames, ignore_index=True)
    out = os.path.join(RAW_DIR, "yellow_tripdata.parquet")
    trips.to_parquet(out, index=False)
    print(f"wrote {out} ({len(trips):,} trips). Now run: python run_pipeline.py")


if __name__ == "__main__":
    main()
