"""Unit tests for the idempotent, execution-date-bounded pipeline tasks."""
import pandas as pd
import pytest

from src.pipelines import tasks


def test_idempotent_upsert_no_duplicate_for_same_period(tmp_path):
    path = str(tmp_path / "log.parquet")
    row = pd.DataFrame([{"TimePeriod": pd.Timestamp("2026-04-01", tz="UTC"), "Point_Forecast": 1.0}])
    tasks._idempotent_upsert(path, row)
    tasks._idempotent_upsert(path, row)        # re-run same date
    assert len(pd.read_parquet(path)) == 1     # not duplicated


def test_idempotent_upsert_appends_new_period(tmp_path):
    path = str(tmp_path / "log.parquet")
    tasks._idempotent_upsert(path, pd.DataFrame([{"TimePeriod": pd.Timestamp("2026-04-01", tz="UTC"), "v": 1}]))
    tasks._idempotent_upsert(path, pd.DataFrame([{"TimePeriod": pd.Timestamp("2026-04-02", tz="UTC"), "v": 2}]))
    assert len(pd.read_parquet(path)) == 2


def test_bounded_excludes_data_after_execution_window():
    df = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"),
        "Volume": range(10),
    })
    out = tasks._bounded(df, "2026-01-05")
    assert len(out) == 5
    assert pd.to_datetime(out["TimePeriod"]).max() <= pd.Timestamp("2026-01-05", tz="UTC")


def test_run_task_rejects_unknown():
    with pytest.raises(ValueError):
        tasks.run_task("not-a-task")
