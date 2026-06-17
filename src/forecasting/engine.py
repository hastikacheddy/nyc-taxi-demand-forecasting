"""
Forecasting inference engine — predicts the NEXT period (t+1) from history
ending at t, using the leakage-free forecaster. This is the genuine forward
forecast that the shadow pilot should evaluate (unlike the nowcaster, which
scores a period whose actual is already known).
"""
import os
import json
import logging

import numpy as np
import yaml
import mlflow
import mlflow.pyfunc

from src.inference.validation import validate_input_data
from src.inference.risk import monte_carlo_var, FALLBACK_SIGMA
from src.forecasting.forecaster import build_next_step_features
from src.common.mlflow_config import resolve_tracking_uri

logger = logging.getLogger(__name__)

_MODEL_KEY = {'D': 'daily_forecast_model_name', 'H': 'hourly_forecast_model_name'}


class DemandForecastEngine:
    def __init__(self, granularity='D'):
        self.granularity = granularity.upper()
        self.config = self._load_config()
        mlflow.set_tracking_uri(resolve_tracking_uri(self.config))

        self.model_name = self.config['mlflow'][_MODEL_KEY[self.granularity]]
        self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@champion")
        self._mv_tags = self._load_version_tags()
        self.error_sigma = self._load_error_sigma()
        self.hour_sigmas = self._load_hour_sigmas()

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(os.path.expandvars(f.read()))

    def _load_version_tags(self):
        try:
            client = mlflow.MlflowClient()
            mv = client.get_model_version_by_alias(self.model_name, "champion")
            return dict(mv.tags)
        except Exception as e:
            logger.warning("Could not read model version tags (%s)", e)
            return {}

    def _load_error_sigma(self):
        default = FALLBACK_SIGMA.get(self.granularity, 50.0)
        tag = self._mv_tags.get("garch_sigma")
        return float(tag) if tag else default

    def _load_hour_sigmas(self):
        """Per-hour-of-day sigma for the hourly band (None for daily)."""
        if self.granularity != 'H':
            return None
        raw = self._mv_tags.get("hour_sigmas")
        if not raw:
            return None
        try:
            return {int(k): float(v) for k, v in json.loads(raw).items()}
        except Exception:
            return None

    def _sigma_for(self, ts):
        """Pick the conditional sigma for the period being forecast."""
        if self.hour_sigmas:
            return self.hour_sigmas.get(int(ts.hour), self.error_sigma)
        return self.error_sigma

    def forecast_next(self, historical_data, confidence_level=0.99):
        """Forecast the period AFTER the latest row in historical_data."""
        historical_data = validate_input_data(historical_data)
        X_row, next_ts = build_next_step_features(historical_data, self.granularity)

        point_forecast = max(0.0, float(self.model.predict(X_row)[0]))
        risk_buffer = monte_carlo_var(point_forecast, self._sigma_for(next_ts), confidence_level)

        return {
            'Forecast_Period': next_ts,
            'Point_Forecast': round(point_forecast, 2),
            'Safety_Buffer_99': round(risk_buffer, 2),
            'Capacity_Target': int(np.ceil(risk_buffer)),
        }
