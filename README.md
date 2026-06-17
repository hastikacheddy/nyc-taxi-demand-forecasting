# NYC Taxi Demand Forecasting — End-to-End MLOps Platform

A production-grade MLOps platform that forecasts **NYC yellow-taxi trip demand**
(daily and hourly) from the public [TLC trip-record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
It pairs a calibrated LightGBM time-series model with the full operational stack —
orchestration, a model registry with automated promotion gating, drift-triggered
retraining, serving, CI/CD, and observability.

> One mental model carries the whole repo: **Airflow conducts; `src/` plays.**
> The DAGs only decide *when* and *in what order*; the real work lives in `src/`
> and runs identically with or without Airflow.

## Architecture

```
 TLC trip records (Parquet)
      │  data_ingestion_and_cleaning (@daily)
      ▼  sensor → ingest+clean (UTC, drop invalid/out-of-range) → validate
 data/{daily,hourly}_demand.csv   (per-period trip counts)
      │
      ├─► weekly_continuous_training (@weekly)
      │     validate → train → CHAMPION-CHALLENGER gate → MLflow @champion
      ├─► daily/hourly_demand_inference
      │     sensor → validate → forecast_next → shadow_log (idempotent upsert)
      └─► shadow_monitoring (@daily)
            score forecasts vs actuals → on drift, TRIGGER retraining

 Serving API (FastAPI)  loads @champion model, exposes /predict + /metrics
```

## Highlights

- **Forecasting:** leakage-free time-series features + LightGBM; calibrated 99%
  buffers via GARCH volatility + Monte-Carlo Value-at-Risk (hour-of-day
  conditional band for the hourly model). On 4 months of 2024 data the daily
  model forecasts ~117k trips/day with calibrated intervals.
- **Orchestration (Airflow):** five DAGs with enterprise guardrails — idempotent
  tasks bounded by the execution date, `KubernetesPodOperator` compute
  isolation, reschedule-mode sensors, concurrency pools, exponential-backoff
  retries.
- **MLOps:** MLflow registry + automated **champion-challenger** promotion gate;
  a closed **drift-monitoring loop** (Evidently) that auto-triggers retraining.
- **Serving & deploy:** FastAPI (API-key auth, rate limiting, Prometheus
  metrics), Docker, hardened Kubernetes manifests (non-root, HPA), MinIO S3
  artifact store.
- **CI/CD & DevSecOps:** lint, 110 tests, Bandit + Semgrep scans, SBOM +
  dependency audit, container scan, image signing.
- **Observability & ops:** Prometheus + Grafana, automated backup/restore,
  data-quality gates (pandera), full handover docs.

## Tech stack
`Python · LightGBM · Apache Airflow · MLflow · FastAPI · Docker · Kubernetes ·
MinIO/S3 · Prometheus/Grafana · pandera · Evidently · DVC · GitHub Actions ·
Bandit/Semgrep`

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Fetch public TLC data and build the raw events file (~200 MB download)
python scripts/download_data.py --start 2024-01 --end 2024-04

# Run the whole pipeline end-to-end (ingest -> train -> forecast), no Airflow
python run_pipeline.py

# Tests
pytest tests/

# Inspect the DAGs in a real Airflow UI (Docker)
docker compose -f docker-compose.airflow.yml up -d --build   # http://localhost:8088
```
> The small aggregated `data/*_demand.csv` are committed so the repo runs out of
> the box; the large raw Parquet is fetched by the download script (git-ignored).

## Repository layout
| Path | What |
|---|---|
| `src/forecasting/` | leakage-free forecaster, training, engine, VaR |
| `src/pipelines/` | idempotent DAG task logic (Airflow-free) |
| `src/orchestration/` | Airflow operator factory (compute isolation) |
| `src/data/`, `src/inference/`, `src/monitoring/`, `src/serving/` | ingestion, validation/risk, drift, API |
| `dags/` | thin Airflow DAGs |
| `kubernetes/`, `docker/` | deploy manifests + images |
| `observability/` | Prometheus + Grafana |
| `tests/`, `testing/` | unit + DagBag validation suites |

See **[ARCHITECTURE_AND_HANDOVER.md](ARCHITECTURE_AND_HANDOVER.md)** for the full
design, the safety guardrails (and *why* each exists), and the ops runbook.

## Data & license
Trip data: NYC TLC open data (public). Image paths (`ghcr.io/your-username/…`)
and infra endpoints are placeholders — set them for your own environment.
Code licensed under MIT — see [LICENSE](LICENSE).
