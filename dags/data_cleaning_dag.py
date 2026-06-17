from datetime import datetime, timedelta
import os
import sys

from airflow import DAG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.orchestration.airflow_helpers import (
    DEFAULT_ARGS, SHADOW_DB_POOL, compute_operator, data_arrival_sensor,
)

# Production data contract: the raw CRM export the sensor waits on.
RAW_CSV_PATH = os.environ.get('RAW_CSV_PATH', '/opt/airflow/data/yellow_tripdata.parquet')

dag = DAG(
    'data_ingestion_and_cleaning',
    default_args={**DEFAULT_ARGS, 'start_date': datetime(2025, 1, 1)},
    description='Ingest + clean raw taxi trip records into daily/hourly aggregates',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'data_engineering', 'ingestion'],
    dagrun_timeout=timedelta(minutes=30),
)

# Non-blocking sensor: wait for a present, non-empty raw export before ingesting.
wait_raw = data_arrival_sensor(dag, 'wait_for_raw_export', RAW_CSV_PATH, pool=SHADOW_DB_POOL)
ingest = compute_operator(dag, 'run_ingestion', 'ingest', pool=SHADOW_DB_POOL)
validate_d = compute_operator(dag, 'validate_daily', 'validate-daily', pool=SHADOW_DB_POOL)
validate_h = compute_operator(dag, 'validate_hourly', 'validate-hourly', pool=SHADOW_DB_POOL)

wait_raw >> ingest >> [validate_d, validate_h]
