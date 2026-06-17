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
HOURLY_CSV = os.path.join(PROJECT_ROOT, 'data', 'hourly_demand.csv')

dag = DAG(
    'weekly_continuous_training',
    default_args={**DEFAULT_ARGS, 'start_date': datetime(2025, 1, 1)},
    description='Weekly champion-challenger retraining — compute-isolated',
    schedule_interval='@weekly',
    catchup=False,
    max_active_runs=1,            # one training run at a time
    tags=['mlops', 'training', 'weekly'],
    dagrun_timeout=timedelta(hours=2),
)

wait_d = data_arrival_sensor(dag, 'wait_for_daily_data', DAILY_CSV, pool=SHADOW_DB_POOL)
wait_h = data_arrival_sensor(dag, 'wait_for_hourly_data', HOURLY_CSV, pool=SHADOW_DB_POOL)
validate_d = compute_operator(dag, 'validate_daily', 'validate-daily', pool=SHADOW_DB_POOL)
validate_h = compute_operator(dag, 'validate_hourly', 'validate-hourly', pool=SHADOW_DB_POOL)
train_d = compute_operator(dag, 'train_daily_forecaster', 'train-daily', pool=SHADOW_DB_POOL)
train_h = compute_operator(dag, 'train_hourly_forecaster', 'train-hourly', pool=SHADOW_DB_POOL)

wait_d >> validate_d >> train_d
wait_h >> validate_h >> train_h
