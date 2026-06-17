"""
Comprehensive DAG validation suite (loads the real DAGs via DagBag).

Verifies clean imports, no cyclic dependencies, and that every enterprise
pillar is wired in: max_active_runs=1 concurrency gate, exponential-backoff
retries, and reschedule-mode sensors.

Airflow is Unix-only (its operators import fcntl), so this runs in CI/Linux and
skips on Windows. It runs in COMPUTE_BACKEND=local so no Kubernetes provider is
required to import the DAGs.
"""
import os
import tempfile
from pathlib import Path

import pytest

_AIRFLOW_HOME = Path(tempfile.mkdtemp(prefix="airflow_home_")).as_posix()
_DAGS_DIR = (Path(__file__).resolve().parent.parent / "dags").as_posix()

os.environ.setdefault("AIRFLOW_HOME", _AIRFLOW_HOME)
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
os.environ["AIRFLOW__CORE__DAGS_FOLDER"] = _DAGS_DIR
os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = "sqlite://"   # in-memory
os.environ.setdefault("RAW_CSV_PATH", "")
os.environ.setdefault("COMPUTE_BACKEND", "local")   # PythonOperator path

pytest.importorskip("fcntl", reason="Airflow requires fcntl (Unix-only); runs in CI/Linux")
pytest.importorskip("airflow", reason="apache-airflow not installed")
from airflow.models import DagBag  # noqa: E402

EXPECTED_DAG_IDS = {
    "data_ingestion_and_cleaning",
    "weekly_continuous_training",
    "daily_demand_inference",
    "hourly_demand_inference",
    "shadow_monitoring",
}


@pytest.fixture(scope="module")
def dagbag():
    # Share one in-memory DB connection so DagBag's metadata lookups work.
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.orm import scoped_session, sessionmaker

    from airflow import settings
    from airflow.models.base import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    settings.engine = engine
    settings.Session = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
    return DagBag(dag_folder=_DAGS_DIR, include_examples=False)


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"


def test_all_expected_dags_present(dagbag):
    missing = EXPECTED_DAG_IDS - set(dagbag.dag_ids)
    assert not missing, f"Missing DAGs: {missing}"


@pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAG_IDS))
def test_dag_has_tasks_and_schedule(dagbag, dag_id):
    dag = dagbag.get_dag(dag_id)
    assert dag is not None, f"{dag_id} failed to load"
    assert len(dag.tasks) >= 1
    assert dag.schedule_interval is not None


@pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAG_IDS))
def test_no_cyclic_dependencies(dagbag, dag_id):
    from airflow.utils.dag_cycle_tester import check_cycle
    check_cycle(dagbag.get_dag(dag_id))   # raises on a cycle


@pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAG_IDS))
def test_concurrency_gate(dagbag, dag_id):
    assert dagbag.get_dag(dag_id).max_active_runs == 1


@pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAG_IDS))
def test_exponential_backoff_retries(dagbag, dag_id):
    da = dagbag.get_dag(dag_id).default_args
    assert da.get("retries", 0) >= 1
    assert da.get("retry_exponential_backoff") is True
    assert da.get("max_retry_delay") is not None


def test_all_sensors_use_reschedule_mode(dagbag):
    from airflow.sensors.base import BaseSensorOperator
    sensor_count = 0
    for dag_id in EXPECTED_DAG_IDS:
        for task in dagbag.get_dag(dag_id).tasks:
            if isinstance(task, BaseSensorOperator):
                sensor_count += 1
                assert task.mode == "reschedule", \
                    f"{dag_id}.{task.task_id} must use reschedule mode, got {task.mode}"
    assert sensor_count >= 1, "expected at least one data-arrival sensor"
