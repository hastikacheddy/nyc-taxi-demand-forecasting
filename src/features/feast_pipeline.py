import pandas as pd
import subprocess
import os
from datetime import datetime

from src.features.materializer import materialize_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def run_feast_pipeline():
    """
    1. Loads cleaned daily/hourly CSV datasets.
    2. Runs feature engineering and outputs them as Parquet files for the Feast offline store.
    3. Runs 'feast apply' to register the entities and feature views.
    4. Materializes features into the online store (SQLite) for real-time serving.
    """
    daily_csv = os.path.join(PROJECT_ROOT, 'data', 'daily_demand.csv')
    hourly_csv = os.path.join(PROJECT_ROOT, 'data', 'hourly_demand.csv')

    if not os.path.exists(daily_csv) or not os.path.exists(hourly_csv):
        raise FileNotFoundError("Cleaned CSV aggregates not found. Please run the Ingestion DAG first.")

    print("[Feast Pipeline] Loading clean daily and hourly data...")
    df_daily = pd.read_csv(daily_csv)
    df_hourly = pd.read_csv(hourly_csv)
    
    # 1. Generate feature Parquet files (offline store sources)
    print("[Feast Pipeline] Generating feature dataframes...")
    materialize_features(df_daily, 'D')
    materialize_features(df_hourly, 'H')
    
    # 2. Run 'feast apply' to update registry
    feature_repo_path = os.path.join(PROJECT_ROOT, 'feature_repo')
    print(f"[Feast Pipeline] Applying Feast schema at {feature_repo_path}...")

    # Run feast command (shell=False — fixed args, no shell injection surface)
    res = subprocess.run(["feast", "apply"], cwd=feature_repo_path, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[Feast Pipeline] Error applying Feast schemas: {res.stderr}")
        raise RuntimeError(res.stderr)
    print(res.stdout)
    
    # 3. Materialize features to Online Store (SQLite)
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[Feast Pipeline] Materializing features to Online Store up to {now_str}...")
    res_mat = subprocess.run(["feast", "materialize-incremental", now_str],
                             cwd=feature_repo_path, capture_output=True, text=True)
    if res_mat.returncode != 0:
        print(f"[Feast Pipeline] Error materializing features: {res_mat.stderr}")
        raise RuntimeError(res_mat.stderr)
    print(res_mat.stdout)
    print("[Feast Pipeline] Features successfully materialized into the Online Store!")

if __name__ == "__main__":
    run_feast_pipeline()
