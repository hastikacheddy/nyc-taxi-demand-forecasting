import os
import logging

import pandas as pd
import yaml
from sqlalchemy import create_engine, text

from src.data.pii import redact_pii_columns, assert_no_pii

logger = logging.getLogger(__name__)

# Timestamp columns we know how to use, in priority order (NYC TLC + generic).
_TS_CANDIDATES = ['tpep_pickup_datetime', 'lpep_pickup_datetime', 'pickup_datetime',
                  'CreatedStamp', 'createdstamp', 'TimePeriod']


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    with open(config_path, 'r') as f:
        raw = f.read()
    # Expand environment variables (${VAR}) in config values
    raw = os.path.expandvars(raw)
    return yaml.safe_load(raw)


def _read_raw(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def run_ingestion_pipeline(daily_output_path: str, hourly_output_path: str, raw_csv_path: str = None):
    """
    Ingest raw NYC-taxi trip records, clean them (timezone, invalid/out-of-range
    timestamps), aggregate pickups into daily and hourly TRIP COUNTS, and write
    the aggregates.

    Primary source is a warehouse table (read-only); it falls back to a local
    raw file (parquet/CSV) produced by scripts/download_data.py for local runs.
    """
    config = load_config()
    db_config = config['database']['crm_replica']

    connection_uri = (
        f"postgresql://{db_config['username']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    df = None
    use_mock_fallback = False
    if db_config['host'] == "crm-replica.example.com" or os.environ.get("MOCK_DATABASE") == "True":
        use_mock_fallback = True
        logger.info("[Ingestion] Placeholder host or MOCK_DATABASE=True — using local raw file.")

    if not use_mock_fallback:
        try:
            logger.info("[Ingestion] Connecting to %s:%s...", db_config['host'], db_config['port'])
            engine = create_engine(connection_uri, pool_pre_ping=True)
            with engine.connect() as conn:                      # text() guards against injection
                df = pd.read_sql(text(db_config['query']), conn)
            logger.info("[Ingestion] Loaded %d rows from the warehouse.", len(df))
        except Exception as e:
            logger.info("[Ingestion] DB unavailable (%s); falling back to local raw file.", e)
            use_mock_fallback = True

    if use_mock_fallback:
        raw_path = raw_csv_path or os.environ.get('RAW_DATA_PATH') or os.environ.get('RAW_CSV_PATH', '')
        if not raw_path or not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Raw trip data not found at '{raw_path}'. Run: python scripts/download_data.py")
        logger.info("[Ingestion] Reading raw trip data: %s", raw_path)
        df = _read_raw(raw_path)
        logger.info("[Ingestion] Loaded raw trips. Shape: %s", df.shape)

    # PII gate (taxi data carries none, but the contract is enforced anyway).
    df, pii_report = redact_pii_columns(df)
    if pii_report:
        logger.warning("[Ingestion] PII detected in raw input and dropped: %s", pii_report)

    # Pick the pickup-timestamp column and normalize to UTC.
    stamp_col = next((c for c in _TS_CANDIDATES if c in df.columns), df.columns[0])
    df[stamp_col] = pd.to_datetime(df[stamp_col], errors='coerce')
    df.dropna(subset=[stamp_col], inplace=True)
    if df[stamp_col].dt.tz is None:
        df[stamp_col] = df[stamp_col].dt.tz_localize('UTC')
    else:
        df[stamp_col] = df[stamp_col].dt.tz_convert('UTC')

    # Each trip is a distinct event (no dedupe). Drop stray timestamps — the TLC
    # files contain a handful of records dated outside the trip month.
    before = len(df)
    df = df[(df[stamp_col].dt.year >= 2009) & (df[stamp_col] <= pd.Timestamp.utcnow())]
    logger.info("[Ingestion] Range-filtered timestamps: %d -> %d rows", before, len(df))

    df.set_index(stamp_col, inplace=True)

    logger.info("[Ingestion] Aggregating daily trip counts...")
    daily_demand = df.resample('D').size().to_frame(name='Volume').reset_index()
    daily_demand.rename(columns={stamp_col: 'TimePeriod'}, inplace=True)

    logger.info("[Ingestion] Aggregating hourly trip counts...")
    hourly_demand = df.resample('h').size().to_frame(name='Volume').reset_index()
    hourly_demand.rename(columns={stamp_col: 'TimePeriod'}, inplace=True)

    # Output gate: published aggregates must be PII-free (anonymous counts only).
    assert_no_pii(daily_demand)
    assert_no_pii(hourly_demand)

    os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(hourly_output_path), exist_ok=True)
    daily_demand.to_csv(daily_output_path, index=False)
    hourly_demand.to_csv(hourly_output_path, index=False)
    logger.info("[Ingestion] Wrote %s (%d) and %s (%d)",
                daily_output_path, len(daily_demand), hourly_output_path, len(hourly_demand))


if __name__ == "__main__":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    run_ingestion_pipeline(
        os.path.join(_root, 'data', 'daily_demand.csv'),
        os.path.join(_root, 'data', 'hourly_demand.csv'),
    )
