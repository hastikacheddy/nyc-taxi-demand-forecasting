"""Tests for the ingestion PII gate."""
import pandas as pd
import pytest

from src.data.pii import scan_dataframe, redact_pii_columns, assert_no_pii, PIIDetectedError


def test_detects_email_and_ssn():
    df = pd.DataFrame({
        "note": ["ping me at jane.doe@example.com", "nothing here"],
        "ref": ["ssn 123-45-6789", "ok"],
        "Volume": [1.0, 2.0],
    })
    report = scan_dataframe(df)
    assert "email" in report["note"]
    assert "ssn" in report["ref"]


def test_redact_drops_pii_text_columns_keeps_counts():
    df = pd.DataFrame({
        "customer_email": ["a@b.com", "c@d.com"],
        "Volume": [1.0, 2.0],
    })
    clean, report = redact_pii_columns(df)
    assert "customer_email" not in clean.columns
    assert "Volume" in clean.columns
    assert "customer_email" in report


def test_assert_no_pii_passes_on_aggregates():
    agg = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=3, freq="D"),
        "Volume": [10.0, 20.0, 30.0],
    })
    assert_no_pii(agg)   # must not raise


def test_assert_no_pii_raises_when_present():
    bad = pd.DataFrame({"leak": ["call 415-555-0172 now"], "Volume": [1.0]})
    with pytest.raises(PIIDetectedError):
        assert_no_pii(bad)
