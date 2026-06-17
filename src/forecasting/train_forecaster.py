"""
Train the genuine next-period forecaster (leakage-free features) and register
it to MLflow as a separate model from the notebook-faithful nowcaster.
"""
import os
import json
import logging
import tempfile

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow

from src.common.config import load_config, get_git_revision_hash, get_hparams
from src.common.mlflow_config import resolve_tracking_uri
from src.common.promotion import promote_if_better
from src.inference.model_integrity import hash_file
from src.inference.risk import compute_garch_sigma, estimate_conditional_sigmas, FALLBACK_SIGMA
from src.forecasting.forecaster import build_training_frame, feature_columns
from src.governance.cards import render_model_card

logger = logging.getLogger(__name__)

_MODEL_KEY = {'D': 'daily_forecast_model_name', 'H': 'hourly_forecast_model_name'}


def _holdout(X, y, hparams):
    """Train on a chronological 80/20 split; return (residuals, index, mae) on
    the held-out tail — a genuine forward out-of-sample estimate. (None on too
    little data.)"""
    if len(X) < 40:
        return None, None, None
    split = int(len(X) * 0.8)
    params = dict(hparams)
    if len(X) < 200:
        params['num_leaves'] = min(params.get('num_leaves', 31), 15)
    tmp = lgb.LGBMRegressor(random_state=42, verbosity=-1, **params)
    tmp.fit(X.iloc[:split], y.iloc[:split])
    pred = tmp.predict(X.iloc[split:])
    resid = y.iloc[split:].values - pred
    mae = float(np.mean(np.abs(resid)))
    return resid, X.index[split:], mae


def train_forecast_model(df_raw: pd.DataFrame, granularity: str):
    gran = granularity.upper()
    config = load_config()
    name = config['mlflow'][_MODEL_KEY[gran]]
    hparams = get_hparams(config, 'forecast')
    min_improvement = config['training']['promotion'].get('min_improvement', 0.0)
    logger.info("Training %s forecast model (leakage-free) -> %s", gran, name)

    mlflow.set_tracking_uri(resolve_tracking_uri(config))
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=f"{gran}_Forecast_Run"):
        mlflow.set_tag("git_commit", get_git_revision_hash())
        mlflow.set_tag("model_type", "forecast")

        X, y = build_training_frame(df_raw, gran)

        # Forward-holdout residuals -> risk sigma(s) and the promotion metric.
        fallback = FALLBACK_SIGMA.get(gran, 50.0)
        resid, resid_index, holdout_mae = _holdout(X, y, hparams)
        if resid is None:
            garch_sigma, hour_sigmas, holdout_mae = fallback, None, float("inf")
        else:
            garch_sigma = compute_garch_sigma(resid, fallback=float(np.std(resid)) or fallback)
            # Hourly: a per-hour-of-day sigma (wide at peak, near-zero overnight)
            hour_sigmas = (estimate_conditional_sigmas(resid, resid_index.hour, fallback=garch_sigma)
                           if gran == 'H' else None)

        model = lgb.LGBMRegressor(random_state=42, verbosity=-1, **hparams)
        model.fit(X, y)

        # Numerical-stability gate: predictions must be finite (no NaN/inf) and
        # the holdout metric well-defined before this model is allowed near the
        # registry. Aborts the run rather than registering a broken model.
        preds = model.predict(X)
        if not np.isfinite(preds).all():
            raise ValueError(f"{gran} forecaster produced non-finite predictions — aborting.")
        if not np.isfinite(holdout_mae):
            raise ValueError(f"{gran} forecaster holdout MAE is not finite — aborting.")

        mlflow.log_metric("holdout_mae", holdout_mae)
        mlflow.log_metric("garch_sigma", garch_sigma)

        import joblib
        fd, tmp_path = tempfile.mkstemp(suffix='.joblib')   # mkstemp: no race condition
        os.close(fd)
        joblib.dump(model, tmp_path)
        artifact_hash = hash_file(tmp_path)
        os.unlink(tmp_path)
        mlflow.set_tag("artifact_sha256", artifact_hash)

        result = mlflow.lightgbm.log_model(lgb_model=model, artifact_path="model",
                                           registered_model_name=name)
        version = result.registered_model_version
        client = mlflow.MlflowClient()
        client.set_model_version_tag(name, version, "garch_sigma", f"{garch_sigma:.6f}")
        if hour_sigmas:
            client.set_model_version_tag(name, version, "hour_sigmas",
                                         json.dumps({str(k): round(v, 4) for k, v in hour_sigmas.items()}))

        # Auto-generated Model Card logged with the run (transparency/audit).
        model_card = render_model_card({
            "name": name, "granularity": gran, "version": version,
            "git_commit": get_git_revision_hash(), "sha256": artifact_hash,
            "holdout_mae": round(holdout_mae, 4), "garch_sigma": round(garch_sigma, 4),
            "features": feature_columns(gran),
        })
        mlflow.log_text(model_card, "model_card.md")

        # Champion-challenger gate: only promote if it beats the incumbent.
        promoted = promote_if_better(client, name, version, holdout_mae, min_improvement)
        logger.info("%s forecast v%s sha256=%s holdout_mae=%.4f garch_sigma=%.4f promoted=%s",
                    gran, version, artifact_hash[:12], holdout_mae, garch_sigma, promoted)
        return promoted


def train_daily_forecaster(df_raw):
    train_forecast_model(df_raw, 'D')


def train_hourly_forecaster(df_raw):
    train_forecast_model(df_raw, 'H')
