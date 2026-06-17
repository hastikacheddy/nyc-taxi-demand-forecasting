"""
Tests for QuantFeatureEngineer — validates notebook-matching feature logic.
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.feature_engineer import QuantFeatureEngineer


# ── Constructor ────────────────────────────────────────────────

def test_missing_columns_raises(daily_raw_df):
    bad_df = daily_raw_df.rename(columns={'Volume': 'WrongName'})
    with pytest.raises(ValueError, match="must contain"):
        QuantFeatureEngineer(bad_df, 'D')


def test_invalid_granularity_raises(daily_raw_df):
    with pytest.raises(ValueError, match="Granularity"):
        QuantFeatureEngineer(daily_raw_df, 'X')


# ── Daily Feature Engineering ─────────────────────────────────

class TestDailyFeatures:
    def test_required_lag_columns_present(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        for lag in [1, 7, 14, 28]:
            assert f'Volume_Lag{lag}' in result.columns
            assert f'Volume_LagDelta{lag}' in result.columns

    def test_lag_delta_equals_current_minus_lag(self, daily_raw_df):
        """LagDelta must match notebook: Volume - Volume_Lag{n}."""
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        expected_delta1 = result['Volume'] - result['Volume_Lag1']
        pd.testing.assert_series_equal(
            result['Volume_LagDelta1'].reset_index(drop=True),
            expected_delta1.reset_index(drop=True),
            check_names=False,
        )

    def test_lag7_delta_correct(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        expected = result['Volume'] - result['Volume_Lag7']
        pd.testing.assert_series_equal(
            result['Volume_LagDelta7'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_rolling_features_present(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        for w in [7, 14, 28]:
            assert f'Volume_RollingMean{w}' in result.columns
            assert f'Volume_RollingStd{w}' in result.columns

    def test_cyclical_features_present(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        for col in ['Sin_DayOfWeek', 'Cos_DayOfWeek', 'Sin_WeekOfYear', 'Cos_WeekOfYear']:
            assert col in result.columns

    def test_no_hour_feature_in_daily(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        assert 'Sin_Hour' not in result.columns
        assert 'Cos_Hour' not in result.columns

    def test_cyclical_values_in_range(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        assert result['Sin_DayOfWeek'].between(-1, 1).all()
        assert result['Cos_DayOfWeek'].between(-1, 1).all()

    def test_regime_defaults(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        assert result['Regime_0'].eq(1.0).all()
        assert result['Regime_1'].eq(0.0).all()
        assert result['Regime_2'].eq(0.0).all()

    def test_no_nan_after_engineer(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        assert not result.isnull().values.any()

    def test_pruned_production_features_present(self, daily_raw_df):
        """The exact feature set used by the production model must exist."""
        production_features = [
            'Volume_Lag1', 'Volume_LagDelta1',
            'Volume_Lag7', 'Volume_LagDelta7',
            'Sin_DayOfWeek', 'Cos_DayOfWeek',
            'Sin_WeekOfYear', 'Cos_WeekOfYear',
            'Regime_0', 'Regime_1', 'Regime_2',
        ]
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        for col in production_features:
            assert col in result.columns, f"Missing production feature: {col}"

    def test_row_count_reduced_by_largest_lag(self, daily_raw_df):
        fe = QuantFeatureEngineer(daily_raw_df, 'D')
        result = fe.engineer_features()
        # Largest lag for daily is 28, plus rolling windows of 28
        assert len(result) < len(daily_raw_df)
        assert len(result) > 0


# ── Hourly Feature Engineering ────────────────────────────────

class TestHourlyFeatures:
    def test_required_lag_columns_present(self, hourly_raw_df):
        fe = QuantFeatureEngineer(hourly_raw_df, 'H')
        result = fe.engineer_features()
        for lag in [1, 6, 12, 24]:
            assert f'Volume_Lag{lag}' in result.columns
            assert f'Volume_LagDelta{lag}' in result.columns

    def test_lag_delta_equals_current_minus_lag(self, hourly_raw_df):
        fe = QuantFeatureEngineer(hourly_raw_df, 'H')
        result = fe.engineer_features()
        expected = result['Volume'] - result['Volume_Lag1']
        pd.testing.assert_series_equal(
            result['Volume_LagDelta1'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_hour_cyclical_features_present(self, hourly_raw_df):
        fe = QuantFeatureEngineer(hourly_raw_df, 'H')
        result = fe.engineer_features()
        assert 'Sin_Hour' in result.columns
        assert 'Cos_Hour' in result.columns

    def test_sin_hour_values_in_range(self, hourly_raw_df):
        fe = QuantFeatureEngineer(hourly_raw_df, 'H')
        result = fe.engineer_features()
        assert result['Sin_Hour'].between(-1, 1).all()
        assert result['Cos_Hour'].between(-1, 1).all()

    def test_no_nan_after_engineer(self, hourly_raw_df):
        fe = QuantFeatureEngineer(hourly_raw_df, 'H')
        result = fe.engineer_features()
        assert not result.isnull().values.any()

    def test_pruned_production_features_present(self, hourly_raw_df):
        production_features = [
            'Volume_Lag1', 'Volume_LagDelta1',
            'Volume_Lag6', 'Volume_LagDelta6',
            'Volume_Lag12', 'Volume_LagDelta12',
            'Volume_Lag24', 'Volume_LagDelta24',
            'Sin_Hour', 'Cos_Hour',
            'Regime_0', 'Regime_1', 'Regime_2',
        ]
        fe = QuantFeatureEngineer(hourly_raw_df, 'H')
        result = fe.engineer_features()
        for col in production_features:
            assert col in result.columns, f"Missing production feature: {col}"

    def test_volume_not_negative_in_output(self, hourly_raw_df):
        fe = QuantFeatureEngineer(hourly_raw_df, 'H')
        result = fe.engineer_features()
        assert (result['Volume'] >= 0).all()
