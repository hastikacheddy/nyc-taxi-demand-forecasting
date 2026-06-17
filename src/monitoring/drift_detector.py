import pandas as pd
import os
import json
from datetime import datetime
from evidently.report import Report
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
    RegressionQualityMetric,
)

# Thresholds — if drift score exceeds these, event-driven CT is triggered
DATA_DRIFT_THRESHOLD    = 0.10  # >10% share of drifted features
CONCEPT_DRIFT_THRESHOLD = 0.15  # >15% relative MAE degradation vs. baseline

def load_reference_data(granularity: str) -> pd.DataFrame:
    """
    Loads the training baseline data (X + y) used when the Champion model was trained.
    In production this is fetched from the Feast offline store or MLflow artifact.
    """
    gran_key = 'daily' if granularity.upper() == 'D' else 'hourly'
    path = os.path.join("data", "reference", f"{gran_key}_reference.parquet")
    return pd.read_parquet(path)

def load_current_window(granularity: str, window_days: int = 7) -> pd.DataFrame:
    """
    Loads recent shadow-log predictions and actuals for comparison.
    In production this queries the shadow log database.
    """
    gran_key = 'daily' if granularity.upper() == 'D' else 'hourly'
    path = os.path.join("data", "shadow_log", f"{gran_key}_shadow_log.parquet")
    df = pd.read_parquet(path)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days)
    return df[df['TimePeriod'] >= cutoff]

def run_drift_report(granularity: str, report_output_dir: str = "data/drift_reports") -> dict:
    """
    Runs EvidentlyAI Data Drift + Regression Quality reports.
    Returns a summary dict with drift detected flag and scores.
    """
    gran_key = granularity.upper()
    print(f"[DriftDetector] Running drift analysis for granularity={gran_key}...")

    reference = load_reference_data(gran_key)
    current   = load_current_window(gran_key)

    if current.empty:
        print("[DriftDetector] No current window data available. Skipping.")
        return {"drift_detected": False, "reason": "no_current_data"}

    # --- Data Drift Report ---
    data_drift_report = Report(metrics=[
        DatasetDriftMetric(drift_share_threshold=DATA_DRIFT_THRESHOLD),
        ColumnDriftMetric(column_name="Volume"),
    ])
    data_drift_report.run(reference_data=reference, current_data=current)

    # --- Regression Quality Report (Concept Drift) ---
    regression_report = Report(metrics=[RegressionQualityMetric()])
    regression_report.run(reference_data=reference, current_data=current)

    # --- Parse Results ---
    data_drift_result     = data_drift_report.as_dict()
    regression_result     = regression_report.as_dict()

    dataset_drift_detected = data_drift_result["metrics"][0]["result"]["dataset_drift"]
    drift_share            = data_drift_result["metrics"][0]["result"]["share_of_drifted_columns"]
    current_mae            = regression_result["metrics"][0]["result"]["current"]["mean_abs_error"]
    reference_mae          = regression_result["metrics"][0]["result"]["reference"]["mean_abs_error"]
    mae_degradation        = (current_mae - reference_mae) / (reference_mae + 1e-9)
    concept_drift_detected = mae_degradation > CONCEPT_DRIFT_THRESHOLD

    # --- Save HTML Report ---
    os.makedirs(report_output_dir, exist_ok=True)
    timestamp    = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    report_path  = os.path.join(report_output_dir, f"{gran_key.lower()}_{timestamp}_drift_report.html")
    data_drift_report.save_html(report_path)
    print(f"[DriftDetector] Report saved to {report_path}")

    summary = {
        "granularity":            gran_key,
        "data_drift_detected":    dataset_drift_detected,
        "drift_share":            round(drift_share, 4),
        "current_mae":            round(current_mae, 4),
        "reference_mae":          round(reference_mae, 4),
        "mae_degradation_pct":    round(mae_degradation * 100, 2),
        "concept_drift_detected": concept_drift_detected,
        "drift_detected":         dataset_drift_detected or concept_drift_detected,
        "report_path":            report_path,
        "evaluated_at":           timestamp,
    }

    print(f"[DriftDetector] Summary: {json.dumps(summary, indent=2)}")
    return summary
