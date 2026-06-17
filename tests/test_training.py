"""
Tests for LightGBM training functions — MLflow is mocked so no server is needed.
"""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.train_lgbm import train_daily_model, train_hourly_model


def _mock_mlflow():
    """Return a context manager that patches all mlflow calls."""
    patches = [
        patch('src.training.train_lgbm.mlflow.set_tracking_uri'),
        patch('src.training.train_lgbm.mlflow.set_experiment'),
        patch('src.training.train_lgbm.mlflow.start_run'),
        patch('src.training.train_lgbm.mlflow.lightgbm.autolog'),
        patch('src.training.train_lgbm.mlflow.set_tag'),
        patch('src.training.train_lgbm.mlflow.log_metric'),
        patch('src.training.train_lgbm.mlflow.lightgbm.log_model'),
        # MlflowClient is used to set the @champion alias after registration —
        # must be mocked or the call hits the real registry and fails.
        patch('src.training.train_lgbm.mlflow.MlflowClient'),
    ]
    return patches


class TestTrainDailyModel:
    def test_runs_without_error(self, daily_raw_df):
        patches = _mock_mlflow()
        mocks = [p.start() for p in patches]
        # Mock the context manager returned by start_run
        mocks[2].return_value.__enter__ = MagicMock(return_value=MagicMock())
        mocks[2].return_value.__exit__ = MagicMock(return_value=False)
        try:
            train_daily_model(daily_raw_df)
        finally:
            for p in patches:
                p.stop()

    def test_mae_is_logged(self, daily_raw_df):
        patches = _mock_mlflow()
        mocks = [p.start() for p in patches]
        mocks[2].return_value.__enter__ = MagicMock(return_value=MagicMock())
        mocks[2].return_value.__exit__ = MagicMock(return_value=False)
        log_metric_mock = mocks[5]
        try:
            train_daily_model(daily_raw_df)
            # log_metric should be called with 'train_mae'
            call_args = [call[0][0] for call in log_metric_mock.call_args_list]
            assert 'train_mae' in call_args
        finally:
            for p in patches:
                p.stop()

    def test_feature_columns_match_config(self, daily_raw_df):
        """Ensure training doesn't crash when config features are present in data."""
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        from src.features.feature_engineer import QuantFeatureEngineer
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        df_fe = fe.engineer_features()
        for col in config['features']['daily']:
            assert col in df_fe.columns, f"Config feature missing from engineered DataFrame: {col}"


class TestTrainHourlyModel:
    def test_runs_without_error(self, hourly_raw_df):
        patches = _mock_mlflow()
        mocks = [p.start() for p in patches]
        mocks[2].return_value.__enter__ = MagicMock(return_value=MagicMock())
        mocks[2].return_value.__exit__ = MagicMock(return_value=False)
        try:
            train_hourly_model(hourly_raw_df)
        finally:
            for p in patches:
                p.stop()

    def test_feature_columns_match_config(self, hourly_raw_df):
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        from src.features.feature_engineer import QuantFeatureEngineer
        fe = QuantFeatureEngineer(hourly_raw_df, 'H')
        df_fe = fe.engineer_features()
        for col in config['features']['hourly']:
            assert col in df_fe.columns, f"Config feature missing from engineered DataFrame: {col}"
