from datetime import datetime, timedelta
import os
import sys

from airflow import DAG
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.orchestration.airflow_helpers import DEFAULT_ARGS, SHADOW_DB_POOL


def _detect_drift(**_):
    # Lazy import keeps the heavy ML stack out of DAG parsing.
    from src.pipelines.tasks import monitor
    return monitor()

dag = DAG(
    'shadow_monitoring',
    default_args={**DEFAULT_ARGS, 'start_date': datetime(2025, 1, 1)},
    description='Score matured shadow forecasts vs actuals; auto-trigger CT on drift',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'monitoring'],
    dagrun_timeout=timedelta(minutes=15),
)

# ShortCircuit: proceed to trigger CT only when drift is detected.
detect = ShortCircuitOperator(
    task_id='score_shadow_and_detect_drift',
    python_callable=_detect_drift,
    pool=SHADOW_DB_POOL,
    dag=dag,
)

trigger_ct = TriggerDagRunOperator(
    task_id='trigger_continuous_training',
    trigger_dag_id='weekly_continuous_training',
    wait_for_completion=False,
    reset_dag_run=True,
    dag=dag,
)

detect >> trigger_ct
