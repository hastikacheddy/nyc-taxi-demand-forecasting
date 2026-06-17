import os
import logging

import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.pyfunc

from src.features.feature_engineer import QuantFeatureEngineer
from src.inference.validation import validate_input_data
from src.inference.risk import monte_carlo_var, FALLBACK_SIGMA
from src.common.mlflow_config import resolve_tracking_uri

logger = logging.getLogger(__name__)

class DemandNowcastEngine:
    def __init__(self, granularity='D', stage='Production'):
        """
        Loads the serialized LightGBM model from MLflow Model Registry.
        """
        self.granularity = granularity.upper()
        self.config = self._load_config()

        mlflow.set_tracking_uri(resolve_tracking_uri(self.config))
        model_name = (self.config['mlflow']['daily_model_name'] if self.granularity == 'D'
                      else self.config['mlflow']['hourly_model_name'])

        # Use @champion alias (MLflow 3.x — replaces deprecated Production stage)
        model_uri = f"models:/{model_name}@champion"
        self.model = mlflow.pyfunc.load_model(model_uri)

        # GARCH-derived VaR sigma, stored on the model version at training time.
        self.error_sigma = self._load_error_sigma(model_name)

    def _load_error_sigma(self, model_name):
        """Read the garch_sigma tag from the @champion model version. Falls back
        to the notebook's MAE-based sigma if the tag is unavailable."""
        default = FALLBACK_SIGMA.get(self.granularity, 50.0)
        try:
            client = mlflow.MlflowClient()
            mv = client.get_model_version_by_alias(model_name, "champion")
            tag = mv.tags.get("garch_sigma")
            return float(tag) if tag else default
        except Exception as e:
            logger.warning("Could not load garch_sigma tag (%s); using fallback %.4f", e, default)
            return default

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
        with open(config_path, 'r') as f:
            raw = f.read()
        raw = os.path.expandvars(raw)
        return yaml.safe_load(raw)

    def prepare_inference_features(self, historical_data: pd.DataFrame):
        """
        Takes raw historical trip counts, validates them, and applies the production QuantFeatureEngineer logic.
        """
        # Data validation gate
        historical_data = validate_input_data(historical_data)
        
        fe = QuantFeatureEngineer(historical_data, self.granularity)
        features_df = fe.engineer_features()

        # Load pruned features from config
        key = 'daily' if self.granularity == 'D' else 'hourly'
        cols = self.config['features'][key]

        # Ensure all columns exist, fill with 0 if missing (e.g. regime cols)
        for col in cols:
            if col not in features_df.columns:
                features_df[col] = 0.0

        return features_df[cols].tail(1) # Return latest period for inference

    def predict_with_risk_buffer(self, X_input, confidence_level=0.99):
        """
        Generates a point forecast and a VaR-based safety buffer using the
        notebook's Monte Carlo VaR (GARCH sigma -> normal sims -> quantile).
        """
        point_forecast = max(0.0, float(self.model.predict(X_input)[0]))

        risk_buffer = monte_carlo_var(point_forecast, self.error_sigma, confidence_level)

        return {
            'Point_Forecast': round(point_forecast, 2),
            'Safety_Buffer_99': round(risk_buffer, 2),
            'Capacity_Target': int(np.ceil(risk_buffer))
        }
