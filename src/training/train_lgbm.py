"""
DEPRECATED — notebook-faithful nowcaster.

This trains the original notebook model (LagDelta features include the current
period, so it nowcasts an already-known value). It is kept for parity with the
notebook and offline analysis, but it is NOT on the production path: the weekly
DAG and all serving/inference use the leakage-free forecaster in
``src.forecasting`` instead. Do not point production traffic at these models.
"""
import os
import tempfile
import logging

import lightgbm as lgb
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error

from src.common.config import load_config, get_git_revision_hash, get_hparams
from src.common.mlflow_config import resolve_tracking_uri
from src.inference.model_integrity import hash_file
from src.inference.risk import estimate_risk_sigma
from src.features.feature_engineer import QuantFeatureEngineer

logger = logging.getLogger(__name__)

_MODEL_KEY = {'D': 'daily_model_name', 'H': 'hourly_model_name'}


def _train_nowcast(df_raw: pd.DataFrame, granularity: str):
    gran = granularity.upper()
    config = load_config()
    name = config['mlflow'][_MODEL_KEY[gran]]
    features = config['features']['daily' if gran == 'D' else 'hourly']
    hparams = get_hparams(config, 'nowcast')
    logger.info("Training %s nowcaster (DEPRECATED reference model) -> %s", gran, name)

    mlflow.set_tracking_uri(resolve_tracking_uri(config))
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=f"{gran}_Nowcast_Run"):
        mlflow.lightgbm.autolog()
        mlflow.set_tag("git_commit", get_git_revision_hash())
        mlflow.set_tag("model_type", "nowcast_deprecated")

        df_fe = QuantFeatureEngineer(df_raw, gran).engineer_features()
        for col in ['Regime_0', 'Regime_1', 'Regime_2']:
            if col not in df_fe.columns:
                df_fe[col] = 1.0 if col == 'Regime_0' else 0.0

        X, y = df_fe[features], df_fe['Volume']
        model = lgb.LGBMRegressor(random_state=42, **hparams)
        model.fit(X, y)
        mlflow.log_metric("train_mae", mean_absolute_error(y, model.predict(X)))

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
        client.set_registered_model_alias(name=name, alias="champion", version=version)

        garch_sigma = estimate_risk_sigma(df_fe, features, gran)
        mlflow.log_metric("garch_sigma", garch_sigma)
        client.set_model_version_tag(name, version, "garch_sigma", f"{garch_sigma:.6f}")
        logger.info("%s nowcaster v%s @champion. sha256=%s garch_sigma=%.4f",
                    gran, version, artifact_hash[:12], garch_sigma)


def train_daily_model(df_raw: pd.DataFrame):
    _train_nowcast(df_raw, 'D')


def train_hourly_model(df_raw: pd.DataFrame):
    _train_nowcast(df_raw, 'H')
