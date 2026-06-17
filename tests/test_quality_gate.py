"""Tests for the DAG data-quality gate."""
import pandas as pd
import pandera as pa
import pytest

from src.data.quality_gate import assert_data_quality


def _write(tmp_path, df):
    p = tmp_path / "agg.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_passes_on_clean_data(tmp_path):
    df = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=40, freq="D"),
        "Volume": range(40),
    })
    assert assert_data_quality(_write(tmp_path, df)) == 40


def test_halts_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        assert_data_quality(str(tmp_path / "nope.csv"))


def test_halts_on_too_few_rows(tmp_path):
    df = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Volume": range(5),
    })
    with pytest.raises(ValueError):
        assert_data_quality(_write(tmp_path, df), min_rows=30)


def test_halts_on_schema_violation(tmp_path):
    df = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=40, freq="D"),
        "Volume": [-1] * 40,           # negative -> schema reject
    })
    with pytest.raises(pa.errors.SchemaError):
        assert_data_quality(_write(tmp_path, df))
