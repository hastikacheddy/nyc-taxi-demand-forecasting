"""
Central resolution of the MLflow tracking URI.

In production (Kubernetes) MLFLOW_TRACKING_URI points at the remote tracking
server, which is backed by Postgres for the registry and MinIO (S3) for model
artifacts. Clients additionally need the S3 env vars to read/write artifacts:

    MLFLOW_TRACKING_URI=http://mlflow-server:5000
    MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...

Locally, with none of these set, it falls back to the config value (sqlite),
so development keeps working unchanged.
"""
import os


def resolve_tracking_uri(config) -> str:
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    return os.path.expandvars(config['mlflow']['tracking_uri'])
