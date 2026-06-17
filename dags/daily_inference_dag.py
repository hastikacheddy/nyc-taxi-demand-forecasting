from datetime import datetime, timedelta
import os
import sys

from airflow import DAG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.orchestration.airflow_helpers import (
    DEFAULT_ARGS, SHADOW_DB_POOL, compute_operator, data_arrival_sensor,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DAILY_CSV = os.path.join(PROJECT_ROOT, 'data', 'daily_demand.csv')

dag = DAG(
    'daily_demand_inference',
    default_args={**DEFAULT_ARGS, 'start_date': datetime(2025, 1, 1)},
    description='Daily next-day forecast — idempotent, compute-isolated',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,            # no task piling
    tags=['mlops', 'inference', 'daily'],
    dagrun_timeout=timedelta(minutes=15),
)

wait = data_arrival_sensor(dag, 'wait_for_daily_data', DAILY_CSV, pool=SHADOW_DB_POOL)
validate = compute_operator(dag, 'validate_data', 'validate-daily', pool=SHADOW_DB_POOL)
forecast = compute_operator(dag, 'run_daily_inference', 'daily-inference', pool=SHADOW_DB_POOL)

wait >> validate >> forecast
