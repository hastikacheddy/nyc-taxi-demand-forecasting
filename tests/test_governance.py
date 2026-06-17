"""Tests for Model Card / Data Card generation."""
import pandas as pd

from src.governance.cards import render_model_card, render_data_card, build_data_card


def test_model_card_contains_key_sections():
    card = render_model_card({
        "name": "TaxiDemand_Daily_Forecast", "granularity": "D", "version": "7",
        "git_commit": "abc123", "sha256": "deadbeef",
        "holdout_mae": 101.5, "garch_sigma": 126.4,
        "features": ["Lag1", "LagS"],
    })
    for section in ("Model Card", "Intended use", "Caveats", "Holdout MAE", "PII"):
        assert section in card
    assert "abc123" in card and "101.5" in card and "Lag1" in card


def test_data_card_documents_pii_and_schema():
    card = render_data_card({"name": "x", "granularity": "Daily",
                             "start": "a", "end": "b", "rows": 10})
    assert "PII handling" in card
    assert "Volume" in card and "Schema" in card


def test_build_data_card_from_frame():
    df = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Volume": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    card = build_data_card("D", df, dvc_remote="s3://bucket")
    assert "Rows: 5" in card
    assert "s3://bucket" in card
