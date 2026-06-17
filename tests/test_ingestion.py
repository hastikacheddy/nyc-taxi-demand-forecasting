"""
Tests for the data ingestion and cleaning pipeline.
Uses the raw_trip_log_df fixture (no real database or CSV needed).
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _build_raw_df(n=500, add_duplicates=True, add_nat=True):
    """Create a synthetic raw trip DataFrame of raw taxi trips."""
    np.random.seed(0)
    timestamps = pd.date_range('2024-01-01', periods=n, freq='15min')
    rows = list(timestamps)
    if add_duplicates:
        rows += list(timestamps[:20])
    if add_nat:
        rows += [pd.NaT] * 5
    np.random.shuffle(rows)
    return pd.DataFrame({'CreatedStamp': rows})


def _run_aggregate(raw_df, granularity):
    """Helper that runs the notebook's aggregate_data logic inline."""
    df = raw_df.copy()
    df['CreatedStamp'] = pd.to_datetime(df['CreatedStamp'], errors='coerce')
    df.dropna(subset=['CreatedStamp'], inplace=True)
    df.drop_duplicates(inplace=True)
    if df['CreatedStamp'].dt.tz is None:
        df['CreatedStamp'] = df['CreatedStamp'].dt.tz_localize('UTC')
    else:
        df['CreatedStamp'] = df['CreatedStamp'].dt.tz_convert('UTC')
    df = df.set_index('CreatedStamp')
    agg = df.resample(granularity).size().to_frame(name='Volume')
    agg.reset_index(inplace=True)
    agg.rename(columns={'CreatedStamp': 'TimePeriod'}, inplace=True)
    return agg


class TestAggregation:
    def test_daily_output_has_required_columns(self):
        raw = _build_raw_df()
        result = _run_aggregate(raw, 'D')
        assert 'TimePeriod' in result.columns
        assert 'Volume' in result.columns

    def test_hourly_output_has_required_columns(self):
        raw = _build_raw_df()
        result = _run_aggregate(raw, 'h')
        assert 'TimePeriod' in result.columns
        assert 'Volume' in result.columns

    def test_no_negative_volume(self):
        raw = _build_raw_df()
        for gran in ['D', 'h']:
            result = _run_aggregate(raw, gran)
            assert (result['Volume'] >= 0).all()

    def test_nat_rows_are_dropped(self):
        raw = _build_raw_df(add_nat=True)
        result = _run_aggregate(raw, 'D')
        assert not result['TimePeriod'].isnull().any()

    def test_duplicates_are_removed(self):
        raw = _build_raw_df(add_duplicates=True)
        result_with = _run_aggregate(raw, 'D')
        # Totals from deduped version should be <= original (duplicates removed)
        assert result_with['Volume'].sum() <= len(pd.date_range('2024-01-01', periods=500, freq='15min'))

    def test_output_timestamps_are_monotonic(self):
        raw = _build_raw_df()
        result = _run_aggregate(raw, 'D')
        assert result['TimePeriod'].is_monotonic_increasing

    def test_daily_periods_are_date_only(self):
        raw = _build_raw_df()
        result = _run_aggregate(raw, 'D')
        # Daily aggregation — all hours should be midnight
        assert (result['TimePeriod'].dt.hour == 0).all()

    def test_hourly_volume_sums_to_daily(self):
        """Hourly totals for a day must equal the daily count for that day."""
        raw = _build_raw_df(add_duplicates=False, add_nat=False)
        daily = _run_aggregate(raw, 'D')
        hourly = _run_aggregate(raw, 'h')

        # Tz-aware comparison — normalize both
        hourly['Date'] = hourly['TimePeriod'].dt.date
        hourly_agg = hourly.groupby('Date')['Volume'].sum().reset_index()

        # TimePeriod from _run_aggregate is already tz-aware (UTC)
        daily['Date'] = daily['TimePeriod'].dt.date
        merged = daily.merge(hourly_agg, on='Date', suffixes=('_daily', '_hourly_sum'))
        # Sums should match exactly
        assert (merged['Volume_daily'] == merged['Volume_hourly_sum']).all()


class TestIngestionPipeline:
    def test_pipeline_writes_output_files(self, tmp_path, monkeypatch):
        """run_ingestion_pipeline writes both CSV files when given a valid CSV."""
        from src.data.ingestion_cleaning import run_ingestion_pipeline

        raw = _build_raw_df(add_duplicates=True, add_nat=True)
        raw_csv = str(tmp_path / 'raw_trips.csv')
        raw.to_csv(raw_csv, index=False)

        daily_out  = str(tmp_path / 'daily_demand.csv')
        hourly_out = str(tmp_path / 'hourly_demand.csv')

        # Force mock fallback so it reads from our temp CSV
        monkeypatch.setenv('MOCK_DATABASE', 'True')
        run_ingestion_pipeline(daily_out, hourly_out, raw_csv_path=raw_csv)

        assert os.path.exists(daily_out)
        assert os.path.exists(hourly_out)

        daily_df  = pd.read_csv(daily_out)
        hourly_df = pd.read_csv(hourly_out)
        assert 'TimePeriod' in daily_df.columns
        assert 'Volume'   in daily_df.columns
        assert 'TimePeriod' in hourly_df.columns
        assert 'Volume'   in hourly_df.columns

    def test_pipeline_raises_if_csv_missing(self, tmp_path, monkeypatch):
        from src.data.ingestion_cleaning import run_ingestion_pipeline
        monkeypatch.setenv('MOCK_DATABASE', 'True')
        with pytest.raises(FileNotFoundError):
            run_ingestion_pipeline(
                str(tmp_path / 'daily.csv'),
                str(tmp_path / 'hourly.csv'),
                raw_csv_path='/nonexistent/path.csv',
            )
