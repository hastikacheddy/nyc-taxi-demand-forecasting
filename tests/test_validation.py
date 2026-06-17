"""
Tests for the data validation gate (pandera schema).
These tests guard the circuit breaker logic — bad data must never reach the model.
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.validation import validate_input_data
import pandera as pa


def _valid_df():
    return pd.DataFrame({
        'TimePeriod': pd.date_range('2024-01-01', periods=5, freq='D'),
        'Volume': [100.0, 200.0, 150.0, 175.0, 90.0],
    })


def test_valid_data_passes():
    df = _valid_df()
    result = validate_input_data(df)
    assert len(result) == 5


def test_negative_volume_raises():
    df = _valid_df()
    df.loc[2, 'Volume'] = -10.0
    with pytest.raises(pa.errors.SchemaError):
        validate_input_data(df)


def test_null_volume_raises():
    df = _valid_df()
    df.loc[1, 'Volume'] = None
    with pytest.raises(pa.errors.SchemaError):
        validate_input_data(df)


def test_null_timestamp_raises():
    df = _valid_df()
    df.loc[0, 'TimePeriod'] = None
    with pytest.raises(pa.errors.SchemaError):
        validate_input_data(df)


def test_zero_volume_passes():
    """Zero is a valid trip count (off-hours with no traffic)."""
    df = _valid_df()
    df.loc[0, 'Volume'] = 0.0
    result = validate_input_data(df)
    assert result['Volume'].iloc[0] == 0.0


def test_non_datetime_timestamp_raises():
    df = _valid_df()
    df['TimePeriod'] = df['TimePeriod'].astype(str)  # Break the type
    with pytest.raises(pa.errors.SchemaError):
        validate_input_data(df)


def test_extra_columns_allowed():
    """Schema should not fail if extra columns are present."""
    df = _valid_df()
    df['ExtraCol'] = 999
    result = validate_input_data(df)
    assert 'TimePeriod' in result.columns
