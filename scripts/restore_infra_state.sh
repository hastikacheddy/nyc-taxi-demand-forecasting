#!/usr/bin/env bash
#
# Restore pipeline state from a backup made by backup_infra_state.sh.
# A backup you have never restored is not a backup — run this against a staging
# environment periodically to prove it works.
#
#   scripts/restore_infra_state.sh backups/infra_state_<TS>.tar.gz
#
# Env (same as backup): AIRFLOW_DB_URL, MLFLOW_DB_URL
set -euo pipefail

ARCHIVE="${1:?usage: restore_infra_state.sh <infra_state_*.tar.gz>}"
[[ -f "$ARCHIVE" ]] || { echo "not found: $ARCHIVE" >&2; exit 1; }

# Verify integrity first.
if [[ -f "$ARCHIVE.sha256" ]]; then
  echo "[restore] verifying checksum"
  sha256sum -c "$ARCHIVE.sha256"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
tar -xzf "$ARCHIVE" -C "$WORK"

# 1. Databases
if [[ -f "$WORK/airflow_metadata.sql" && -n "${AIRFLOW_DB_URL:-}" ]]; then
  echo "[restore] airflow metadata -> \$AIRFLOW_DB_URL"
  psql "$AIRFLOW_DB_URL" < "$WORK/airflow_metadata.sql"
fi
if [[ -f "$WORK/mlflow_registry.sql" && -n "${MLFLOW_DB_URL:-}" ]]; then
  echo "[restore] mlflow registry -> \$MLFLOW_DB_URL"
  psql "$MLFLOW_DB_URL" < "$WORK/mlflow_registry.sql"
fi
for db in mlflow.db mlflow_airflow.db; do
  [[ -f "$WORK/$db" ]] && { mkdir -p data; cp "$WORK/$db" "data/$db"; echo "[restore] data/$db"; }
done

# 2. Shadow log + aggregates
if [[ -f "$WORK/state.tar.gz" ]]; then
  echo "[restore] shadow log + aggregates"
  tar -xzf "$WORK/state.tar.gz" -C .
fi

echo "[restore] complete from $ARCHIVE"
