from datetime import datetime, timedelta
import os
import sys

from airflow import DAG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.orchestration.airflow_helpers import (
    DEFAULT_ARGS, SHADOW_DB_POOL, compute_operator, data_arrival_sensor,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HOURLY_CSV = os.path.join(PROJECT_ROOT, 'data', 'hourly_demand.csv')

dag = DAG(
    'hourly_demand_inference',
    default_args={**DEFAULT_ARGS, 'start_date': datetime(2025, 1, 1)},
    description='Hourly next-hour forecast — idempotent, compute-isolated',
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1,            # no task piling
    tags=['mlops', 'inference', 'hourly'],
    dagrun_timeout=timedelta(minutes=5),
)

wait = data_arrival_sensor(dag, 'wait_for_hourly_data', HOURLY_CSV, pool=SHADOW_DB_POOL)
validate = compute_operator(dag, 'validate_data', 'validate-hourly', pool=SHADOW_DB_POOL)
forecast = compute_operator(dag, 'run_hourly_inference', 'hourly-inference', pool=SHADOW_DB_POOL)

wait >> validate >> forecast
