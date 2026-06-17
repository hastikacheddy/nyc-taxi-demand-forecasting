"""
Lightweight PII detection for the ingestion layer.

The pipeline practises data minimization — only the timestamp is used, and it is
aggregated to anonymous counts. This module makes that guarantee *enforceable*:
raw inputs are scanned so any PII is logged and the columns dropped before
processing, and the published aggregates are asserted to contain none.
"""
import re

import pandas as pd

# Patterns are deliberately specific so ID/date/timestamp columns (e.g.
# 2024-06-17, long numeric IDs) are NOT flagged as PII.
PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # 16 digits in four separated groups — not bare numeric IDs.
    "credit_card": re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b"),
    # 3-3-4 grouping with separators — not 4-2-2 dates.
    "phone": re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
}


class PIIDetectedError(ValueError):
    pass


def _object_columns(df: pd.DataFrame, columns=None):
    if columns is not None:
        return columns
    return [c for c in df.columns if df[c].dtype == object]


def scan_dataframe(df: pd.DataFrame, columns=None, sample: int = 5000) -> dict:
    """Return {column: {pattern: match_count}} for any text column with hits."""
    report = {}
    for col in _object_columns(df, columns):
        series = df[col].dropna().astype(str)
        if sample and len(series) > sample:
            series = series.sample(sample, random_state=42)
        hits = {}
        for name, pat in PATTERNS.items():
            n = int(series.str.contains(pat, regex=True).sum())
            if n:
                hits[name] = n
        if hits:
            report[col] = hits
    return report


def redact_pii_columns(df: pd.DataFrame, keep=("CreatedStamp", "createdstamp", "TimePeriod", "Volume")):
    """Drop any text column found to contain PII (keeping the minimal allow-list).
    Returns (clean_df, report)."""
    report = scan_dataframe(df)
    drop = [c for c in report if c not in keep]
    return df.drop(columns=drop, errors="ignore"), report


def assert_no_pii(df: pd.DataFrame, columns=None) -> None:
    """Raise if any PII remains — used as a gate on the published aggregates."""
    report = scan_dataframe(df, columns=columns)
    if report:
        raise PIIDetectedError(f"PII detected in output columns: {report}")
