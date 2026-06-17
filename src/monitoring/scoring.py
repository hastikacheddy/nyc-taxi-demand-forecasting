"""
Close the shadow-pilot monitoring loop.

The shadow log records forecasts for FUTURE periods. Once those periods occur,
their actuals land in the cleaned aggregates. This module joins matured
forecasts to their actuals, computes realised MAE and VaR coverage, and flags
concept drift when realised error degrades materially past the champion's
training-time holdout MAE — the "automated ground truth" the pilot promises.
"""
import logging
import os

import pandas as pd

from src.common.config import load_config
from src.common.mlflow_config import resolve_tracking_uri

logger = logging.getLogger(__name__)

_MODEL_KEY = {'D': 'daily_forecast_model_name', 'H': 'hourly_forecast_model_name'}


def _to_utc_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert('UTC').dt.tz_localize(None)
    return s


def score_forecasts(log_df: pd.DataFrame, actuals_df: pd.DataFrame) -> dict:
    """Pure scoring: join forecasts to actuals on the period and compute metrics.
    Returns realised MAE, 99% VaR coverage and the matured-row count."""
    log = log_df.copy()
    actuals = actuals_df[['TimePeriod', 'Volume']].copy()
    log['TimePeriod'] = _to_utc_naive(log['TimePeriod'])
    actuals['TimePeriod'] = _to_utc_naive(actuals['TimePeriod'])

    merged = log.merge(actuals, on='TimePeriod', how='inner')
    if merged.empty:
        return {'n_matured': 0, 'realised_mae': None, 'coverage_99': None}

    err = (merged['Volume'] - merged['Point_Forecast']).abs()
    covered = merged['Volume'] <= merged['Safety_Buffer_99']
    return {
        'n_matured': int(len(merged)),
        'realised_mae': round(float(err.mean()), 4),
        'coverage_99': round(float(covered.mean() * 100), 2),
    }


def champion_holdout_mae(granularity: str, config: dict):
    """The champion forecaster's training-time holdout MAE, used as the drift
    baseline. None if unavailable."""
    import mlflow
    try:
        mlflow.set_tracking_uri(resolve_tracking_uri(config))
        client = mlflow.MlflowClient()
        name = config['mlflow'][_MODEL_KEY[granularity.upper()]]
        mv = client.get_model_version_by_alias(name, "champion")
        tag = mv.tags.get("holdout_mae")
        return float(tag) if tag else None
    except Exception as e:
        logger.warning("Could not read champion holdout_mae (%s)", e)
        return None


def run_monitoring(granularity: str, project_root: str = None,
                   degrade_threshold: float = 0.15) -> dict:
    """Resolve paths, score the matured shadow forecasts, and flag concept drift
    if realised MAE exceeds the champion's holdout MAE by degrade_threshold."""
    gran = granularity.upper()
    config = load_config()
    root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    key = 'daily' if gran == 'D' else 'hourly'

    shadow_path = os.path.join(root, 'data', 'shadow_log', f'{key}_shadow_log.parquet')
    actuals_path = os.path.join(root, 'data', f'{key}_demand.csv')
    if not os.path.exists(shadow_path) or not os.path.exists(actuals_path):
        return {'granularity': gran, 'n_matured': 0, 'drift_detected': False,
                'reason': 'missing shadow log or actuals'}

    summary = score_forecasts(pd.read_parquet(shadow_path),
                              pd.read_csv(actuals_path, parse_dates=['TimePeriod']))
    summary['granularity'] = gran

    baseline = champion_holdout_mae(gran, config)
    summary['baseline_mae'] = baseline
    drift = False
    if summary['n_matured'] > 0 and baseline:
        drift = summary['realised_mae'] > baseline * (1 + degrade_threshold)
    summary['drift_detected'] = bool(drift)

    if drift:
        logger.warning("[monitoring] %s CONCEPT DRIFT: realised MAE %.2f > baseline %.2f (+%.0f%%)",
                       gran, summary['realised_mae'], baseline, degrade_threshold * 100)
    else:
        logger.info("[monitoring] %s ok: matured=%d realised_mae=%s coverage_99=%s baseline=%s",
                    gran, summary['n_matured'], summary['realised_mae'],
                    summary['coverage_99'], baseline)
    return summary
