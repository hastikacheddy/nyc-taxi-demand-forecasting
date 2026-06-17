import os
import pandas as pd

from src.features.feature_engineer import QuantFeatureEngineer

def materialize_features(df_raw: pd.DataFrame, granularity: str, output_dir: str = "data/features"):
    """
    Batch job that computes features via QuantFeatureEngineer and persists them
    as a Parquet file for the Feast offline store to ingest.

    This is the ONLY place features are computed. Both training and inference
    pipelines READ from this store, guaranteeing zero training-serving skew.

    Args:
        df_raw: Raw historical trip data with 'TimePeriod' and 'Volume' columns.
        granularity: 'D' for Daily or 'H' for Hourly.
        output_dir: Path to directory where Parquet files will be saved.
    """
    gran_key = granularity.upper()
    print(f"[Materializer] Computing {gran_key} features for {len(df_raw)} rows...")

    fe = QuantFeatureEngineer(df_raw, gran_key)
    features_df = fe.engineer_features()

    # Add entity key expected by Feast
    if gran_key == 'D':
        features_df['time_period'] = features_df.index.strftime('%Y-%m-%d')
    else:
        features_df['time_period'] = features_df.index.strftime('%Y-%m-%dT%H')

    # Reset index so TimePeriod is a column (required by Feast FileSource)
    features_df = features_df.reset_index()

    os.makedirs(output_dir, exist_ok=True)
    filename = 'daily_features.parquet' if gran_key == 'D' else 'hourly_features.parquet'
    output_path = os.path.join(output_dir, filename)
    features_df.to_parquet(output_path, index=False)

    print(f"[Materializer] Saved {len(features_df)} rows of {gran_key} features to {output_path}")
    return output_path
