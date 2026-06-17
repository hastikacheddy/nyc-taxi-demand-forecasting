"""
Tests for DemandNowcastEngine — model loading is mocked via MLflow.
Tests the feature preparation and risk buffer logic independently.
"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _make_engine(granularity):
    """Instantiate engine with a mocked MLflow model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([150.0])

    with patch('src.inference.engine.mlflow.pyfunc.load_model', return_value=mock_model):
        from src.inference.engine import DemandNowcastEngine
        engine = DemandNowcastEngine(granularity=granularity)
    engine.model = mock_model
    return engine


class TestPrepareInferenceFeatures:
    def test_daily_returns_single_row(self, daily_raw_df):
        engine = _make_engine('D')
        X = engine.prepare_inference_features(daily_raw_df)
        assert X.shape[0] == 1

    def test_hourly_returns_single_row(self, hourly_raw_df):
        engine = _make_engine('H')
        X = engine.prepare_inference_features(hourly_raw_df)
        assert X.shape[0] == 1

    def test_daily_feature_columns_match_config(self, daily_raw_df):
        import yaml
        with open(os.path.join(os.path.dirname(__file__), '..', 'config.yaml')) as f:
            config = yaml.safe_load(f)
        engine = _make_engine('D')
        X = engine.prepare_inference_features(daily_raw_df)
        assert list(X.columns) == config['features']['daily']

    def test_hourly_feature_columns_match_config(self, hourly_raw_df):
        import yaml
        with open(os.path.join(os.path.dirname(__file__), '..', 'config.yaml')) as f:
            config = yaml.safe_load(f)
        engine = _make_engine('H')
        X = engine.prepare_inference_features(hourly_raw_df)
        assert list(X.columns) == config['features']['hourly']

    def test_validation_gate_blocks_negative_counts(self, daily_raw_df):
        import pandera as pa
        engine = _make_engine('D')
        bad_df = daily_raw_df.copy()
        bad_df.loc[0, 'Volume'] = -999.0
        with pytest.raises(pa.errors.SchemaError):
            engine.prepare_inference_features(bad_df)

    def test_no_nulls_in_feature_output(self, daily_raw_df):
        engine = _make_engine('D')
        X = engine.prepare_inference_features(daily_raw_df)
        assert not X.isnull().values.any()


class TestPredictWithRiskBuffer:
    def test_returns_required_keys(self, daily_raw_df):
        engine = _make_engine('D')
        X = engine.prepare_inference_features(daily_raw_df)
        result = engine.predict_with_risk_buffer(X)
        assert 'Point_Forecast' in result
        assert 'Safety_Buffer_99' in result
        assert 'Capacity_Target' in result

    def test_safety_buffer_gte_point_forecast(self, daily_raw_df):
        engine = _make_engine('D')
        X = engine.prepare_inference_features(daily_raw_df)
        result = engine.predict_with_risk_buffer(X, confidence_level=0.99)
        assert result['Safety_Buffer_99'] >= result['Point_Forecast']

    def test_capacity_target_is_integer(self, daily_raw_df):
        engine = _make_engine('D')
        X = engine.prepare_inference_features(daily_raw_df)
        result = engine.predict_with_risk_buffer(X)
        assert isinstance(result['Capacity_Target'], int)

    def test_point_forecast_non_negative(self, daily_raw_df):
        """Even if model predicts a negative value, output must be clipped to 0."""
        engine = _make_engine('D')
        engine.model.predict.return_value = np.array([-50.0])
        X = engine.prepare_inference_features(daily_raw_df)
        result = engine.predict_with_risk_buffer(X)
        assert result['Point_Forecast'] >= 0

    def test_lower_confidence_gives_smaller_buffer(self, daily_raw_df):
        engine = _make_engine('D')
        X = engine.prepare_inference_features(daily_raw_df)
        r90 = engine.predict_with_risk_buffer(X, confidence_level=0.90)
        r99 = engine.predict_with_risk_buffer(X, confidence_level=0.99)
        assert r99['Safety_Buffer_99'] >= r90['Safety_Buffer_99']
