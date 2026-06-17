#!/usr/bin/env bash
#
# Nightly backup of pipeline STATE — the things you cannot regenerate:
#   - Airflow metadata DB        (DAG/run history, connections, pools)
#   - MLflow registry DB         (model versions, @champion, metrics)
#   - Shadow-log history         (the pilot's forecast-vs-actual evidence)
#   - Cleaned aggregates         (daily/hourly trip counts)
#
# Postgres (prod) is dumped with pg_dump; the local sqlite files are copied.
# The bundle is checksummed and pushed to an isolated bucket if configured.
#
# Env:
#   AIRFLOW_DB_URL   postgres URL for the Airflow metadata DB (optional)
#   MLFLOW_DB_URL    postgres URL for the MLflow registry DB  (optional)
#   BACKUP_S3_URI    e.g. s3://mlops-backups/airflow  (optional; needs mc or aws)
#   BACKUP_DIR       local staging dir (default ./backups)
#   RETENTION        local copies to keep (default 14)
#
# Schedule (cron):   0 2 * * *  /path/to/scripts/backup_infra_state.sh
# Or as a Kubernetes CronJob in production.
set -euo pipefail

TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION="${RETENTION:-14}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$BACKUP_DIR"

echo "[backup] staging $TS"

# 1. Metadata databases -------------------------------------------------------
if [[ -n "${AIRFLOW_DB_URL:-}" ]]; then
  echo "[backup] pg_dump airflow metadata"
  pg_dump "$AIRFLOW_DB_URL" > "$STAGE/airflow_metadata.sql"
fi
if [[ -n "${MLFLOW_DB_URL:-}" ]]; then
  echo "[backup] pg_dump mlflow registry"
  pg_dump "$MLFLOW_DB_URL" > "$STAGE/mlflow_registry.sql"
fi
# Local-dev fallback: copy sqlite registries if present.
for db in data/mlflow.db data/mlflow_airflow.db; do
  [[ -f "$db" ]] && cp "$db" "$STAGE/" || true
done

# 2. Irreplaceable state: shadow log + cleaned aggregates ---------------------
if [[ -d data/shadow_log || -f data/daily_demand.csv ]]; then
  tar -czf "$STAGE/state.tar.gz" \
    $( [[ -d data/shadow_log ]] && echo data/shadow_log ) \
    $( [[ -f data/daily_demand.csv ]] && echo data/daily_demand.csv ) \
    $( [[ -f data/hourly_demand.csv ]] && echo data/hourly_demand.csv ) 2>/dev/null || true
fi

# 3. Bundle + checksum --------------------------------------------------------
ARCHIVE="$BACKUP_DIR/infra_state_$TS.tar.gz"
tar -czf "$ARCHIVE" -C "$STAGE" .
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo "[backup] wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

# 4. Push to an isolated bucket ----------------------------------------------
if [[ -n "${BACKUP_S3_URI:-}" ]]; then
  if command -v mc >/dev/null 2>&1; then
    mc cp "$ARCHIVE" "$BACKUP_S3_URI/" && mc cp "$ARCHIVE.sha256" "$BACKUP_S3_URI/"
  elif command -v aws >/dev/null 2>&1; then
    aws s3 cp "$ARCHIVE" "$BACKUP_S3_URI/" && aws s3 cp "$ARCHIVE.sha256" "$BACKUP_S3_URI/"
  else
    echo "[backup] WARN: BACKUP_S3_URI set but neither 'mc' nor 'aws' found — kept local only"
  fi
fi

# 5. Local retention ----------------------------------------------------------
ls -1t "$BACKUP_DIR"/infra_state_*.tar.gz 2>/dev/null | tail -n "+$((RETENTION + 1))" | while read -r old; do
  rm -f "$old" "$old.sha256"
done

echo "[backup] done"
