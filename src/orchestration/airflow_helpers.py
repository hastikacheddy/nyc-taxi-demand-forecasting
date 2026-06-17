"""
Airflow orchestration helpers shared by every DAG.

Pillars:
  - Self-healing retries with exponential backoff (DEFAULT_ARGS).
  - Compute isolation: compute_operator() returns a KubernetesPodOperator in
    production (COMPUTE_BACKEND=kubernetes) so Airflow only orchestrates an
    external pod; locally it falls back to a PythonOperator running the same
    src.pipelines task code — identical behaviour, no cluster required.
  - Concurrency metering: SHADOW_DB_POOL caps concurrent shadow-DB access.
  - Non-blocking validation: data_arrival_sensor() is a reschedule-mode sensor.
"""
import os
from datetime import timedelta

from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

# Defensive retry strategy for transient network drops / API timeouts / locks.
DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}

# Explicit pool (create once: `airflow pools set shadow_db 2 "shadow DB slots"`).
SHADOW_DB_POOL = "shadow_db"

_IMAGE = os.environ.get(
    "PIPELINE_IMAGE", "ghcr.io/your-username/demand-forecast-inference:latest"
)
_NAMESPACE = os.environ.get("K8S_NAMESPACE", "mlops")
_AS_OF = "{{ data_interval_end }}"   # Jinja macro — the execution window end


def _run_local(task_name, as_of):
    from src.pipelines.tasks import run_task
    run_task(task_name, as_of=as_of)


def compute_operator(dag, task_id, task_name, pool=None):
    """Heavy compute task. KubernetesPodOperator in prod, PythonOperator locally."""
    backend = os.environ.get("COMPUTE_BACKEND", "local").lower()

    if backend == "kubernetes":
        # Imported lazily so local/dev and the DagBag test don't need the
        # cncf.kubernetes provider installed.
        from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
        return KubernetesPodOperator(
            task_id=task_id,
            dag=dag,
            pool=pool,
            name=f"demand-forecast-{task_id}",
            namespace=_NAMESPACE,
            image=_IMAGE,
            cmds=["python", "-m", "src.pipelines.cli"],
            arguments=[task_name, "--as-of", _AS_OF],
            get_logs=True,
            is_delete_operator_pod=True,
            container_resources={
                "request_memory": "512Mi", "request_cpu": "250m",
                "limit_memory": "1Gi", "limit_cpu": "1000m",
            },
            security_context={"runAsNonRoot": True, "runAsUser": 1000},
        )

    return PythonOperator(
        task_id=task_id,
        dag=dag,
        pool=pool,
        python_callable=_run_local,
        op_kwargs={"task_name": task_name, "as_of": _AS_OF},
    )


def data_arrival_sensor(dag, task_id, path, pool=None):
    """Non-blocking (reschedule) sensor: wait until `path` exists and is non-empty.
    reschedule mode frees the worker slot between checks (no worker starvation)."""
    def _present(p):
        return os.path.exists(p) and os.path.getsize(p) > 0

    return PythonSensor(
        task_id=task_id,
        dag=dag,
        pool=pool,
        python_callable=_present,
        op_args=[path],
        mode="reschedule",     # never poke — drop the slot and sleep
        poke_interval=60,
        timeout=60 * 60,
        soft_fail=False,
    )
