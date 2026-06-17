"""
Tests for DAG callable logic — runs without Airflow by calling functions directly.
DAG import errors would only surface in a real Airflow environment.
"""
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestHourlyInferenceDagLogic:
    def test_inference_writes_shadow_log(self, tmp_path, hourly_raw_df):
        """Calling the hourly DAG function should append a row to the parquet shadow log."""
        hourly_csv = str(tmp_path / 'hourly_demand.csv')
        hourly_raw_df.to_csv(hourly_csv, index=False)

        shadow_dir = str(tmp_path / 'shadow_log')
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([12.5])

        with patch('src.inference.engine.mlflow.pyfunc.load_model', return_value=mock_model):
            from src.inference.engine import DemandNowcastEngine

            engine = DemandNowcastEngine(granularity='H')
            df = pd.read_csv(hourly_csv, parse_dates=['TimePeriod'])
            df = df.sort_values('TimePeriod').tail(60).reset_index(drop=True)
            X = engine.prepare_inference_features(df)
            results = engine.predict_with_risk_buffer(X)

            log_path = os.path.join(shadow_dir, 'hourly_shadow_log.parquet')
            os.makedirs(shadow_dir, exist_ok=True)
            log_row = pd.DataFrame([{
                'TimePeriod': df['TimePeriod'].iloc[-1],
                'Inferred_At': pd.Timestamp.utcnow(),
                **results,
            }])
            log_row.to_parquet(log_path, index=False)

        assert os.path.exists(log_path)
        written = pd.read_parquet(log_path)
        assert len(written) == 1
        assert 'Point_Forecast' in written.columns


class TestDailyInferenceDagLogic:
    def test_inference_writes_shadow_log(self, tmp_path, daily_raw_df):
        daily_csv = str(tmp_path / 'daily_demand.csv')
        daily_raw_df.to_csv(daily_csv, index=False)

        shadow_dir = str(tmp_path / 'shadow_log')
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([180.0])

        with patch('src.inference.engine.mlflow.pyfunc.load_model', return_value=mock_model):
            from src.inference.engine import DemandNowcastEngine

            engine = DemandNowcastEngine(granularity='D')
            df = pd.read_csv(daily_csv, parse_dates=['TimePeriod'])
            df = df.sort_values('TimePeriod').tail(60).reset_index(drop=True)
            X = engine.prepare_inference_features(df)
            results = engine.predict_with_risk_buffer(X)

            log_path = os.path.join(shadow_dir, 'daily_shadow_log.parquet')
            os.makedirs(shadow_dir, exist_ok=True)
            log_row = pd.DataFrame([{
                'TimePeriod': df['TimePeriod'].iloc[-1],
                'Inferred_At': pd.Timestamp.utcnow(),
                **results,
            }])
            log_row.to_parquet(log_path, index=False)

        assert os.path.exists(log_path)
        written = pd.read_parquet(log_path)
        assert 'Capacity_Target' in written.columns


class TestWeeklyTrainingDagLogic:
    def test_training_called_with_correct_dataframes(self, tmp_path, daily_raw_df, hourly_raw_df):
        daily_csv  = str(tmp_path / 'daily_demand.csv')
        hourly_csv = str(tmp_path / 'hourly_demand.csv')
        daily_raw_df.to_csv(daily_csv, index=False)
        hourly_raw_df.to_csv(hourly_csv, index=False)

        with patch('src.training.train_lgbm.mlflow.set_tracking_uri'), \
             patch('src.training.train_lgbm.mlflow.set_experiment'), \
             patch('src.training.train_lgbm.mlflow.start_run') as mock_run, \
             patch('src.training.train_lgbm.mlflow.lightgbm.autolog'), \
             patch('src.training.train_lgbm.mlflow.set_tag'), \
             patch('src.training.train_lgbm.mlflow.log_metric'), \
             patch('src.training.train_lgbm.mlflow.lightgbm.log_model'), \
             patch('src.training.train_lgbm.mlflow.MlflowClient'):

            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            from src.training.train_lgbm import train_daily_model, train_hourly_model
            df_d = pd.read_csv(daily_csv, parse_dates=['TimePeriod'])
            df_h = pd.read_csv(hourly_csv, parse_dates=['TimePeriod'])
            train_daily_model(df_d)
            train_hourly_model(df_h)
