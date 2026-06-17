"""
Tests for the leakage-free next-period forecaster.
"""
import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting.forecaster import (
    feature_columns, build_training_frame, build_next_step_features,
)
from src.inference.risk import estimate_conditional_sigmas


class TestConditionalSigma:
    def test_overnight_floored_peak_wider(self):
        # hour 2 = overnight (zero residuals -> floored); hour 9 = peak (spread)
        keys = np.array([2] * 30 + [9] * 30)
        resid = np.concatenate([np.zeros(30),
                                np.random.RandomState(0).normal(0, 20, 30)])
        sig = estimate_conditional_sigmas(resid, keys, fallback=8.0, floor=0.5)
        assert sig[2] == 0.5
        assert sig[9] > 5

    def test_sparse_bucket_uses_fallback(self):
        keys = np.array([5] * 3)          # too few samples
        resid = np.array([1.0, 2.0, 3.0])
        sig = estimate_conditional_sigmas(resid, keys, fallback=7.5, min_count=20)
        assert sig[5] == 7.5


class TestForecastFeatures:
    def test_training_frame_columns_and_no_nan(self, daily_raw_df):
        X, y = build_training_frame(daily_raw_df, 'D')
        assert list(X.columns) == feature_columns('D')
        assert not X.isnull().values.any()
        assert len(X) == len(y)

    def test_features_do_not_leak_target(self, daily_raw_df):
        """Mutating the last actual must NOT change the feature row that
        predicts it — proving features use only prior data."""
        X1, _ = build_training_frame(daily_raw_df, 'D')
        last_row_before = X1.iloc[-1].copy()

        mutated = daily_raw_df.copy()
        mutated.loc[mutated.index[-1], 'Volume'] += 5000.0
        X2, _ = build_training_frame(mutated, 'D')

        pd.testing.assert_series_equal(last_row_before, X2.iloc[-1], check_names=False)

    def test_next_step_is_future_single_row(self, daily_raw_df):
        X_row, next_ts = build_next_step_features(daily_raw_df, 'D')
        assert len(X_row) == 1
        assert not X_row.isnull().values.any()
        assert next_ts > daily_raw_df['TimePeriod'].max()

    def test_hourly_has_hour_features(self, hourly_raw_df):
        X, _ = build_training_frame(hourly_raw_df, 'H')
        assert 'Sin_Hour' in X.columns and 'Cos_Hour' in X.columns


class TestForecastEngine:
    def _engine(self, granularity):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([300.0])
        with patch('src.forecasting.engine.mlflow.pyfunc.load_model', return_value=mock_model):
            from src.forecasting.engine import DemandForecastEngine
            eng = DemandForecastEngine(granularity=granularity)
        eng.model = mock_model
        eng.error_sigma = 100.0
        return eng

    def test_forecast_next_returns_future_period(self, daily_raw_df):
        eng = self._engine('D')
        result = eng.forecast_next(daily_raw_df)
        assert result['Forecast_Period'] > daily_raw_df['TimePeriod'].max()
        assert result['Safety_Buffer_99'] >= result['Point_Forecast']
        assert isinstance(result['Capacity_Target'], int)

    def test_forecast_clips_negative(self, daily_raw_df):
        eng = self._engine('D')
        eng.model.predict.return_value = np.array([-20.0])
        result = eng.forecast_next(daily_raw_df)
        assert result['Point_Forecast'] >= 0

    def test_hourly_engine_picks_hour_sigma(self, hourly_raw_df):
        eng = self._engine('H')
        eng.hour_sigmas = {9: 15.0, 2: 0.5}
        assert eng._sigma_for(pd.Timestamp('2026-01-01 09:00')) == 15.0
        assert eng._sigma_for(pd.Timestamp('2026-01-01 02:00')) == 0.5
        # hour with no bucket falls back to the global sigma
        assert eng._sigma_for(pd.Timestamp('2026-01-01 13:00')) == eng.error_sigma
