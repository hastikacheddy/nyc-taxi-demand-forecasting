# 🏛️ NYC Taxi Demand Forecasting — Architecture & Handover

A batch MLOps platform that forecasts customer-taxi trip volume (daily + hourly) to
plan fleet capacity, running as a **shadow pilot** (predictions are logged, never
acted on automatically). One mental model carries the whole repo:

> **Airflow conducts; `src/` plays.** The DAGs only say *when* and *in what
> order*. The actual work lives in `src/` and runs identically with or without
> Airflow.

---

## 1. System topology (how data moves)

```
 CRM replica / raw CSV
        │
        ▼   data_ingestion_and_cleaning  (@daily)
   sensor → ingest+clean (dedupe, UTC, PII drop) → validate
        │
        ▼
   data/{daily,hourly}_demand.csv   (DVC-tracked aggregates)
        │
        ├─► weekly_continuous_training (@weekly)
        │      validate → train forecaster → CHAMPION-CHALLENGER gate
        │      → MLflow registry  (models:/...@champion, artifacts in MinIO)
        │
        ├─► daily_demand_inference (@daily) / hourly_demand_inference (@hourly)
        │      sensor → validate → forecast_next → shadow_log (parquet, upsert)
        │
        └─► shadow_monitoring (@daily)
               score shadow log vs actuals → if concept drift, TRIGGER training
 ​
 Serving API (FastAPI)  loads models:/...@champion, exposes /predict + /metrics
```

**Where the real code is** (not in the DAGs):
| Concern | File |
|---|---|
| Ingestion / cleaning / dedupe | `src/data/ingestion_cleaning.py` |
| PII detection | `src/data/pii.py` |
| Data-quality gate | `src/data/quality_gate.py` |
| Feature engineering | `src/features/feature_engineer.py`, `src/forecasting/forecaster.py` |
| Training + promotion gate | `src/forecasting/train_forecaster.py`, `src/common/promotion.py` |
| Forecast engine (VaR buffer) | `src/forecasting/engine.py`, `src/inference/risk.py` |
| Monitoring / drift | `src/monitoring/scoring.py`, `src/monitoring/drift_detector.py` |
| Serving API | `src/serving/api.py` |
| DAG task logic (idempotent) | `src/pipelines/tasks.py` |
| Airflow operator factory | `src/orchestration/airflow_helpers.py` |

---

## 2. Infrastructure guardrails (locked — do not loosen without reason)

These are deliberate. Removing them reintroduces the exact failure each prevents:

- **`max_active_runs=1`** on every DAG — stops *task piling*: a delayed run and
  the next scheduled run executing at once and fighting over the DB/model.
- **Sensors `mode='reschedule'`** (never `poke`) — a waiting sensor *releases*
  its worker slot instead of holding it, preventing worker starvation.
- **`shadow_db` pool (2 slots)** — caps concurrent connections to the shadow
  store. Create it once: `airflow pools set shadow_db 2 "shadow DB slots"`.
- **Champion-challenger gate** (`src/common/promotion.py`) — a retrain is only
  aliased `@champion` if its out-of-sample holdout MAE beats the incumbent;
  otherwise it is parked `@challenger`. Never blind-promote.
- **Idempotency** — task inputs are bounded by `{{ data_interval_end }}` and
  shadow-log writes upsert on the forecast period, so re-running a date never
  duplicates rows. Enforced by the Semgrep rule `no-wallclock-time-in-dags`.
- **Retries with exponential backoff** (`DEFAULT_ARGS`) — absorbs transient
  network/DB blips without manual intervention.
- **Compute isolation** — set `COMPUTE_BACKEND=kubernetes` in prod and heavy
  tasks run as their own pods (`KubernetesPodOperator`); Airflow only
  orchestrates. Locally it falls back to in-process `PythonOperator`.

---

## 3. Local verification execution loop

```bash
# Unit + DAG tests (DAG suite runs on Linux/CI; skips on Windows)
pytest tests/ testing/

# Full pipeline end-to-end, no Airflow needed
python run_pipeline.py

# Static security + quality review (flake8, bandit, semgrep, radon)
pip install -r requirements-dev.txt
bash scripts/run_automated_review.sh

# Inspect the DAGs in a real Airflow UI (Docker)
docker compose -f docker-compose.airflow.yml up -d --build   # http://localhost:8088

# Manual task override (run one task in isolation)
python -m src.pipelines.cli daily-inference --as-of 2026-04-01
```

---

## 4. Operations

- **Backups:** `scripts/backup_infra_state.sh` (cron `0 2 * * *`) dumps the
  Airflow + MLflow DBs, shadow log, and aggregates to an isolated bucket.
  Restore with `scripts/restore_infra_state.sh <archive>` — **test it on
  staging periodically**.
- **CI (GitHub Actions):** `.github/workflows/mlops_pipeline.yaml` — lint, code
  review (bandit/semgrep/radon), tests, data-quality, model smoke test, SBOM +
  SCA, image build/scan/sign.
- **Observability:** `docker-compose.observability.yml` + `observability/`
  (Prometheus + Grafana). Dashboard: *MLOps — NYC Taxi Demand Forecasting*.
- **Deploy:** images at `ghcr.io/your-username/demand-forecast-{inference,serving}`;
  manifests in `kubernetes/`. Required cluster config: ConfigMap `mlops-config`,
  Secrets `mlops-api-secrets`/`mlops-db-secrets`/`mlops-minio-secrets`, PVC
  `mlops-data`, and the `shadow_db` Airflow pool.

---

## 5. Secrets & config
No credentials live in the repo. `config.yaml` uses `${ENV_VAR}` placeholders;
prod injects via Kubernetes Secrets / env. `MLFLOW_TRACKING_URI`,
`MLFLOW_S3_ENDPOINT_URL`, `AWS_*`, `API_KEY`, `COMPUTE_BACKEND` are environment
driven (see `.env.example`).
