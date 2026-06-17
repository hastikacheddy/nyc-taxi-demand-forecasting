"""Tests for the shadow-pilot monitoring loop (forecast vs actual scoring)."""
import pandas as pd

import src.monitoring.scoring as scoring
from src.monitoring.scoring import score_forecasts, run_monitoring


def _log(n, point, buffer, start="2026-01-01", freq="D"):
    return pd.DataFrame({
        "TimePeriod": pd.date_range(start, periods=n, freq=freq),
        "Point_Forecast": [float(point)] * n,
        "Safety_Buffer_99": [float(buffer)] * n,
    })


def test_realised_mae_and_coverage():
    log = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=3, freq="D"),
        "Point_Forecast": [100.0, 200.0, 300.0],
        "Safety_Buffer_99": [150.0, 250.0, 350.0],
    })
    actuals = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=3, freq="D"),
        "Volume": [110, 190, 360],
    })
    s = score_forecasts(log, actuals)
    assert s["n_matured"] == 3
    assert round(s["realised_mae"], 2) == 26.67          # mean(10, 10, 60)
    assert round(s["coverage_99"], 2) == 66.67           # 360 breaches its band


def test_only_matured_periods_join():
    log = _log(5, 100, 200)
    actuals = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=2, freq="D"),
        "Volume": [100, 100],
    })
    assert score_forecasts(log, actuals)["n_matured"] == 2


def test_no_overlap_returns_empty():
    log = _log(2, 1, 2, start="2026-01-01")
    actuals = pd.DataFrame({
        "TimePeriod": pd.date_range("2027-01-01", periods=2, freq="D"),
        "Volume": [1, 2],
    })
    s = score_forecasts(log, actuals)
    assert s["n_matured"] == 0 and s["realised_mae"] is None


def test_tz_aware_log_joins_naive_actuals():
    log = _log(2, 100, 200)
    log["TimePeriod"] = log["TimePeriod"].dt.tz_localize("UTC")
    actuals = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=2, freq="D"),
        "Volume": [100, 100],
    })
    assert score_forecasts(log, actuals)["n_matured"] == 2


def test_run_monitoring_flags_drift(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "shadow_log").mkdir(parents=True)
    # Forecast 100 but actuals 300 -> realised MAE 200, far above baseline 10.
    _log(5, 100, 400).to_parquet(data / "shadow_log" / "daily_shadow_log.parquet", index=False)
    pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Volume": [300] * 5,
    }).to_csv(data / "daily_demand.csv", index=False)

    monkeypatch.setattr(scoring, "load_config", lambda: {"mlflow": {}})
    monkeypatch.setattr(scoring, "champion_holdout_mae", lambda g, c: 10.0)

    s = run_monitoring("D", project_root=str(tmp_path))
    assert s["n_matured"] == 5
    assert s["drift_detected"] is True
