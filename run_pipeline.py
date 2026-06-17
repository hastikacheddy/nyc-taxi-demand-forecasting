"""
Standalone end-to-end pipeline runner — executes each step's logic directly
without Airflow (handy on Windows, where Airflow can't run). Not a pytest file.
Run from project root: python run_pipeline.py
"""
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Raw NYC-taxi events built by scripts/download_data.py
RAW_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw', 'yellow_tripdata.parquet')

DAILY_CSV      = os.path.join(PROJECT_ROOT, 'data', 'daily_demand.csv')
HOURLY_CSV     = os.path.join(PROJECT_ROOT, 'data', 'hourly_demand.csv')
SHADOW_LOG_DIR = os.path.join(PROJECT_ROOT, 'data', 'shadow_log')

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── 1. Data Ingestion ──────────────────────────────────────────
separator("STEP 1: Data Ingestion & Cleaning")
from src.data.ingestion_cleaning import run_ingestion_pipeline
run_ingestion_pipeline(DAILY_CSV, HOURLY_CSV, raw_csv_path=RAW_CSV)
print("[OK] Ingestion complete")
print(f"     Daily rows:  {len(pd.read_csv(DAILY_CSV))}")
print(f"     Hourly rows: {len(pd.read_csv(HOURLY_CSV))}")

# ── 2. Feature Engineering check ──────────────────────────────
separator("STEP 2: Feature Engineering")
from src.features.feature_engineer import QuantFeatureEngineer

df_daily  = pd.read_csv(DAILY_CSV,  parse_dates=['TimePeriod'])
df_hourly = pd.read_csv(HOURLY_CSV, parse_dates=['TimePeriod'])

fe_d = QuantFeatureEngineer(df_daily, 'D')
daily_fe = fe_d.engineer_features()
print(f"[OK] Daily features: {list(daily_fe.columns)}")
print(f"     Rows after dropping NaN lags: {len(daily_fe)}")

fe_h = QuantFeatureEngineer(df_hourly, 'H')
hourly_fe = fe_h.engineer_features()
print(f"[OK] Hourly features: {list(hourly_fe.columns)}")
print(f"     Rows after dropping NaN lags: {len(hourly_fe)}")

# ── 3. Weekly Training ─────────────────────────────────────────
separator("STEP 3: Weekly Forecaster Training (champion-challenger gated)")
from src.forecasting.train_forecaster import train_daily_forecaster, train_hourly_forecaster

print("Training daily forecaster...")
train_daily_forecaster(df_daily)
print("[OK] Daily forecaster trained and registered to MLflow")

print("Training hourly forecaster...")
train_hourly_forecaster(df_hourly)
print("[OK] Hourly forecaster trained and registered to MLflow")

# ── 4. Daily Forecast (next day) ───────────────────────────────
separator("STEP 4: Daily Forecast (next period)")
from src.forecasting.engine import DemandForecastEngine

df_daily_inf = df_daily.sort_values('TimePeriod').tail(60).reset_index(drop=True)
engine_d = DemandForecastEngine(granularity='D')
results_d = engine_d.forecast_next(df_daily_inf)
print(f"[OK] Daily forecast: {results_d}")

log_row_d = pd.DataFrame([{
    'TimePeriod': results_d.pop('Forecast_Period'),
    'Inferred_At': pd.Timestamp.utcnow(),
    **results_d,
}])
os.makedirs(SHADOW_LOG_DIR, exist_ok=True)
daily_log_path = os.path.join(SHADOW_LOG_DIR, 'daily_shadow_log.parquet')
if os.path.exists(daily_log_path):
    log_row_d = pd.concat([pd.read_parquet(daily_log_path), log_row_d], ignore_index=True)
log_row_d.to_parquet(daily_log_path, index=False)
print(f"[OK] Shadow log written -> {daily_log_path}")

# ── 5. Hourly Forecast (next hour) ─────────────────────────────
separator("STEP 5: Hourly Forecast (next period)")
df_hourly_inf = df_hourly.sort_values('TimePeriod').tail(60).reset_index(drop=True)
engine_h = DemandForecastEngine(granularity='H')
results_h = engine_h.forecast_next(df_hourly_inf)
print(f"[OK] Hourly forecast: {results_h}")

log_row_h = pd.DataFrame([{
    'TimePeriod': results_h.pop('Forecast_Period'),
    'Inferred_At': pd.Timestamp.utcnow(),
    **results_h,
}])
hourly_log_path = os.path.join(SHADOW_LOG_DIR, 'hourly_shadow_log.parquet')
if os.path.exists(hourly_log_path):
    log_row_h = pd.concat([pd.read_parquet(hourly_log_path), log_row_h], ignore_index=True)
log_row_h.to_parquet(hourly_log_path, index=False)
print(f"[OK] Shadow log written -> {hourly_log_path}")

# ── Summary ────────────────────────────────────────────────────
separator("ALL STEPS PASSED")
print(f"Shadow logs:")
print(f"  daily_shadow_log.parquet  exists: {os.path.exists(daily_log_path)}")
print(f"  hourly_shadow_log.parquet exists: {os.path.exists(hourly_log_path)}")
print(f"MLflow DB: {os.path.join(PROJECT_ROOT, 'data', 'mlflow.db')}")
