"""
Pure pipeline task logic — no Airflow imports, so it runs identically whether
invoked by a local PythonOperator, a KubernetesPodOperator (via cli.py), or a
unit test.

Every task is IDEMPOTENT and bounded by an explicit execution date (`as_of`,
from Airflow's {{ data_interval_end }}): inputs are filtered to data <= as_of,
shadow-log writes upsert on the forecast period (re-running a date never
duplicates rows), and audit timestamps use as_of (no datetime.now()).
"""
import logging
import os

import pandas as pd

from src.data.quality_gate import assert_data_quality
from src.forecasting.engine import DemandForecastEngine
from src.forecasting.train_forecaster import train_daily_forecaster, train_hourly_forecaster
from src.monitoring.scoring import run_monitoring

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
_RECENT_WINDOW = 60


def _csv(granularity):
    name = "daily" if granularity.upper() == "D" else "hourly"
    return os.path.join(PROJECT_ROOT, "data", f"{name}_demand.csv")


def _shadow(granularity):
    name = "daily" if granularity.upper() == "D" else "hourly"
    return os.path.join(PROJECT_ROOT, "data", "shadow_log", f"{name}_shadow_log.parquet")


def _bounded(df, as_of):
    """Keep only rows up to the execution window end (reproducible backfills)."""
    ts = pd.to_datetime(df["TimePeriod"], utc=True)
    return df[ts <= pd.to_datetime(as_of, utc=True)]


def _idempotent_upsert(path, row, key="TimePeriod"):
    """Write `row`, replacing any existing record for the same period — so
    re-running the same execution date overwrites rather than duplicates."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        existing = existing[existing[key] != row[key].iloc[0]]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_parquet(path, index=False)


def ingest():
    """Ingest + clean the raw CRM export into daily/hourly aggregates.
    to_csv overwrites, so this is naturally idempotent for a given input."""
    from src.data.ingestion_cleaning import run_ingestion_pipeline
    raw = os.environ.get("RAW_CSV_PATH") or None
    run_ingestion_pipeline(_csv("D"), _csv("H"), raw_csv_path=raw)


def validate(granularity):
    """Data-quality gate (raises -> DAG halts)."""
    return assert_data_quality(_csv(granularity))


def infer(granularity, as_of):
    """Forecast the period after `as_of`, idempotently logged to the shadow log."""
    gran = granularity.upper()
    df = pd.read_csv(_csv(gran), parse_dates=["TimePeriod"])
    df = _bounded(df, as_of).sort_values("TimePeriod").tail(_RECENT_WINDOW).reset_index(drop=True)

    results = DemandForecastEngine(granularity=gran).forecast_next(df)
    row = pd.DataFrame([{
        "TimePeriod": results["Forecast_Period"],
        "Inferred_At": pd.to_datetime(as_of, utc=True),   # deterministic, not now()
        "Point_Forecast": results["Point_Forecast"],
        "Safety_Buffer_99": results["Safety_Buffer_99"],
        "Capacity_Target": results["Capacity_Target"],
    }])
    _idempotent_upsert(_shadow(gran), row)
    logger.info("[infer] %s as_of=%s -> %s", gran, as_of, results)
    return results


def train(granularity, as_of):
    """Retrain the forecaster on data up to `as_of` (champion-challenger gated)."""
    gran = granularity.upper()
    df = pd.read_csv(_csv(gran), parse_dates=["TimePeriod"])
    df = _bounded(df, as_of)
    (train_daily_forecaster if gran == "D" else train_hourly_forecaster)(df)


def monitor():
    """Score matured forecasts vs actuals; return True if drift is detected."""
    drift = False
    for gran in ("D", "H"):
        summary = run_monitoring(gran)
        logger.info("[monitor] %s -> %s", gran, summary)
        drift = drift or summary.get("drift_detected", False)
    return drift


def run_task(task_name, as_of=None):
    """Single dispatch used by both the local PythonOperator and the pod CLI."""
    table = {
        "ingest": ingest,
        "validate-daily": lambda: validate("D"),
        "validate-hourly": lambda: validate("H"),
        "daily-inference": lambda: infer("D", as_of),
        "hourly-inference": lambda: infer("H", as_of),
        "train-daily": lambda: train("D", as_of),
        "train-hourly": lambda: train("H", as_of),
        "monitor": monitor,
    }
    if task_name not in table:
        raise ValueError(f"unknown task: {task_name}")
    return table[task_name]()
