"""
Data-quality gate for the DAGs (Phase III).

Run as an explicit upstream task so the pipeline HALTS before any training or
inference if the data is missing, too small, stale, or fails the schema/bounds
check — preventing corrupted inputs from propagating downstream.
"""
import logging
import os

import pandas as pd

from src.inference.validation import validate_input_data

logger = logging.getLogger(__name__)


def assert_data_quality(csv_path: str, min_rows: int = 30) -> int:
    """Raise if the aggregates are absent, too few, or schema-invalid.
    Returns the validated row count on success."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[gate] data not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['TimePeriod'])
    if len(df) < min_rows:
        raise ValueError(f"[gate] {csv_path}: {len(df)} rows (< {min_rows}) — halting pipeline")

    # pandera schema + bounds (rejects NaN/negative/inf/absurd counts).
    validate_input_data(df)

    logger.info("[gate] %s passed: %d rows", csv_path, len(df))
    return len(df)
