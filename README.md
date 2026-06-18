# NYC Taxi Demand Forecasting — End-to-End MLOps Platform

> A production-grade system that forecasts **NYC yellow-taxi trip demand** (daily &
> hourly) and ships it the way a real ML team would: leakage-free modelling,
> calibrated risk intervals, a model registry with an automated promotion gate, a
> closed drift-retraining loop, a hardened serving API, Kubernetes deploy, and a
> six-stage CI/CD pipeline with security scanning and image signing.

<p>
<a href="https://github.com/hastikacheddy/nyc-taxi-demand-forecasting/actions/workflows/mlops_pipeline.yaml"><img alt="CI" src="https://github.com/hastikacheddy/nyc-taxi-demand-forecasting/actions/workflows/mlops_pipeline.yaml/badge.svg"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11-blue">
<img alt="Tests" src="https://img.shields.io/badge/tests-109%20passing-brightgreen">
<img alt="Coverage gate" src="https://img.shields.io/badge/coverage%20gate-%E2%89%A570%25-green">
<img alt="Security" src="https://img.shields.io/badge/security-Bandit%20%7C%20Semgrep%20%7C%20Trivy-orange">
<img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

This is not a notebook with a `model.pkl`. It is the surrounding **operational
system** — the 90% of ML work that happens after the model trains — built on real
data and graded against an MLOps maturity framework.

---

## Contents
- [Results](#results) · [What makes it production-grade](#what-makes-this-production-grade-not-a-notebook) · [Architecture](#architecture)
- [Capabilities](#capabilities) · [Engineering decisions](#engineering-decisions-worth-calling-out) · [Quickstart](#quickstart) · [Repo layout](#repository-layout)

---

## Results

Out-of-sample evaluation on **~13M real trips** across 4 months of 2024 TLC
yellow-taxi data, aggregated to per-period demand. Scoring uses a strict
chronological **80/20 split** — the model never sees the held-out tail it is
graded on. Reproduce end to end with `python scripts/evaluate.py`.

| Model  | Test window | MAE | MAPE | 99% interval coverage |
|--------|-------------|----:|-----:|----------------------:|
| **Daily**  | 23 days        | ~6,200 trips | **5.4 %** | 95.7 % |
| **Hourly** | 24 days (576 h) | ~260 trips   | **8.1 %** | 100 % |

The 99% Value-at-Risk band is **calibrated** — empirical coverage lands at/above
the 99% target without being absurdly wide (the daily band sits ~12–15% above the
point forecast). Next-step production forecast for `2024-05-01`: **117,078
trips/day** (99% capacity target 134,564); peak-hour target **2,592 trips/h**.

![Forecast vs actual](docs/forecast_vs_actual.png)

*Top — daily demand: the point forecast (dashed) tracks actuals inside the 99% VaR
band. Bottom — hourly demand over the last 7 test days: the model captures the full
commuter rhythm (overnight trough, AM/PM peaks) with ~8% error.*

---

## What makes this *production-grade*, not a notebook

The single most important engineering decision in the repo, and the one I'd lead
with in an interview:

> **The original notebook model was a *nowcast*, not a forecast — and I caught it.**

The notebook's features include `LagDelta = Volume − Lag_n`, which contains the
*current* period's value. That produces a beautiful in-sample MAE (~13) but is
**data leakage**: the model can only "predict" a period whose actual is already
known. A naïve port would have shipped a model that looks excellent in a demo and
is useless in production.

Instead I built a separate, **leakage-free forecaster**
([`src/forecasting/forecaster.py`](src/forecasting/forecaster.py)) whose every
feature for target period *t* is computed strictly from data **before** *t*
(`Lag1`, seasonal lag, *past-only* momentum and rolling stats, plus calendar
features known in advance). Re-scored honestly out-of-sample, that's the **5.4%
daily MAPE** above — a number you can actually trust in production. The original
notebook logic is preserved untouched; the forecaster is additive.

This is the difference between *"I trained a model"* and *"I understand why a
model fails in production."*

---

## Architecture

```
 TLC trip records (public Parquet, ~13M rows)
      │  data_cleaning_dag  (@daily)
      ▼  sensor → ingest+clean (UTC, drop invalid/out-of-range) → pandera gate
 data/{daily,hourly}_demand.csv   (per-period trip counts, DVC-tracked)
      │
      ├─►  weekly_training_dag  (@weekly)
      │      validate → train LightGBM → GARCH σ → CHAMPION-CHALLENGER gate
      │      → MLflow registry @champion   (+ model card, SHA-256 integrity tag)
      │
      ├─►  daily / hourly_inference_dag
      │      sensor → forecast_next → 99% VaR band → shadow_log (idempotent upsert)
      │
      └─►  monitoring_dag  (@daily)
             join matured forecasts to actuals → realised MAE & coverage
             → on concept drift, TRIGGER retraining          ← closed loop
                                   │
 Serving API (FastAPI) ───────────┘  loads @champion, /predict + /metrics
      │   API-key auth · rate-limited · Prometheus telemetry · anomaly guard
      ▼
 Kubernetes (non-root, read-only FS, HPA 2→6) · MinIO/S3 artifacts · Prometheus+Grafana
```

> **One mental model carries the whole repo: _Airflow conducts; `src/` plays._**
> The DAGs only decide *when* and *in what order*. The real logic lives in `src/`
> and runs identically with or without Airflow — which is exactly why **109 tests**
> can exercise it directly.

See **[ARCHITECTURE_AND_HANDOVER.md](ARCHITECTURE_AND_HANDOVER.md)** for the full
design, every safety guardrail (and *why* it exists), and the ops runbook.

---

## Capabilities

### 🧠 Modelling & uncertainty
| What | Where |
|---|---|
| Leakage-free 1-step LightGBM forecaster (daily + hourly) | [`forecasting/forecaster.py`](src/forecasting/forecaster.py) |
| **GARCH(1,1) + 10k-sim Monte-Carlo Value-at-Risk** for calibrated 99% buffers | [`inference/risk.py`](src/inference/risk.py) |
| **Hour-of-day conditional σ** — wide at rush hour, floored overnight (heteroscedasticity the single GARCH σ misses) | [`inference/risk.py`](src/inference/risk.py) |
| Reproducible out-of-sample evaluation harness + chart | [`scripts/evaluate.py`](scripts/evaluate.py) |

### 🔁 MLOps lifecycle
| What | Where |
|---|---|
| MLflow registry with **champion-challenger promotion gate** — a retrain only goes live if its holdout MAE *beats* the incumbent; otherwise it's parked as `@challenger` | [`common/promotion.py`](src/common/promotion.py) |
| **Closed drift loop** — joins matured forecasts to actuals, flags concept drift at +15% realised MAE, auto-triggers retraining | [`monitoring/scoring.py`](src/monitoring/scoring.py) |
| Evidently drift detection on feature distributions | [`monitoring/drift_detector.py`](src/monitoring/drift_detector.py) |
| Feast feature store (offline → online materialization) | [`feature_repo/`](feature_repo/) |
| Model cards / governance metadata logged per version | [`governance/cards.py`](src/governance/cards.py) |
| DVC data versioning | [`.dvc/`](.dvc/) |

### 🛰️ Orchestration (Airflow)
Five DAGs with **enterprise guardrails**: idempotent tasks bounded by the
execution date (safe backfills/retries), `KubernetesPodOperator` compute
isolation, **reschedule-mode sensors** (no held worker slots), concurrency pools +
`max_active_runs=1`, and exponential-backoff retries. The orchestration-free task
logic lives in [`src/pipelines/`](src/pipelines/) and is validated by a DagBag
suite in [`testing/`](testing/).

### 🚀 Serving & deployment
| What | Where |
|---|---|
| FastAPI service sharing the **exact** batch forecast code path (API & DAGs give identical forecasts) | [`serving/api.py`](src/serving/api.py) |
| API-key auth (constant-time compare), **60 req/min rate limit** (model-extraction defence), audit log with hashed request IDs, OpenAPI disabled in prod, anomaly guard on outputs | [`serving/api.py`](src/serving/api.py) |
| Hardened Kubernetes: non-root, `readOnlyRootFilesystem`, drop **ALL** caps, seccomp `RuntimeDefault`, no SA token, **HPA 2→6 @70% CPU** | [`kubernetes/`](kubernetes/) |
| OPA/Rego admission policy enforcing the pod least-privilege baseline | [`kubernetes/opa-policy.rego`](kubernetes/opa-policy.rego) |
| Locust load-test profile | [`loadtest/`](loadtest/) |

### 🔐 Security & supply chain
| What | Where |
|---|---|
| **6-stage CI/CD**: lint → code-review → tests(+cov gate) → data-quality → smoke-train → supply-chain → image build | [`.github/workflows/`](.github/workflows/mlops_pipeline.yaml) |
| Static analysis: **Bandit + Semgrep** (custom architecture rules) gate the build | [`.semgrep/rules.yml`](.semgrep/rules.yml) |
| **SBOM/AI-BOM** (CycloneDX), dependency CVE scan (pip-audit), **Trivy** image scan (fails on CRITICAL), **cosign** image signing | CI workflow |
| **Model-artifact integrity** — SHA-256 verified before load (defends against tampered pickles executing on deserialization) | [`inference/model_integrity.py`](src/inference/model_integrity.py) |
| PII scanner + pandera data-quality gates on every batch | [`data/pii.py`](src/data/pii.py), [`data/quality_gate.py`](src/data/quality_gate.py) |

### 📈 Observability & ops
Prometheus + Grafana dashboards (request rate, latency histograms, drift metrics
via statsd-exporter), automated infra-state **backup/restore** scripts, and a full
handover runbook. See [`observability/`](observability/).

---

## Engineering decisions worth calling out

- **Caught data leakage** that would have shipped a fake-good model, and fixed it
  without discarding the original analysis ([story above](#what-makes-this-production-grade-not-a-notebook)).
- **Risk as a first-class output, not an afterthought.** Capacity planning needs a
  *bound*, not a point estimate — so the system serves a calibrated 99% VaR buffer,
  validated by empirical coverage, with per-hour volatility.
- **A promotion gate that can say no.** Continuous training is dangerous without
  one: a bad retrain silently degrades prod. Here a new model must *earn* `@champion`.
- **The serving API and the batch DAGs run the same forecasting code** — no
  train/serve skew by construction.
- **Security treated as part of "done"**: least-privilege pods, signed & scanned
  images, artifact-integrity checks, rate limiting, secrets via env/secret-refs only.

---

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Fetch public TLC data and build the raw events file (~200 MB download)
python scripts/download_data.py --start 2024-01 --end 2024-04

# Run the whole pipeline end-to-end (ingest → train → forecast), no Airflow needed
python run_pipeline.py

# Reproduce the out-of-sample evaluation + chart
python scripts/evaluate.py

# Tests (109 unit + integration)
pytest tests/ testing/

# Inspect the DAGs in a real Airflow UI (Docker)
docker compose -f docker-compose.airflow.yml up -d --build   # http://localhost:8088
```
> The small aggregated `data/*_demand.csv` are committed so the repo runs out of
> the box; the large raw Parquet is fetched by the download script (git-ignored).

---

## Repository layout
| Path | What |
|---|---|
| [`src/forecasting/`](src/forecasting/) | leakage-free forecaster, training, serving engine, VaR |
| [`src/inference/`](src/inference/) | risk/VaR, input validation, model-integrity guard |
| [`src/pipelines/`](src/pipelines/) | idempotent DAG task logic (Airflow-free, fully testable) |
| [`src/orchestration/`](src/orchestration/) | Airflow operator factory (compute isolation, retries) |
| [`src/monitoring/`](src/monitoring/) | drift loop + scoring against actuals |
| [`src/serving/`](src/serving/) | FastAPI app (auth, rate limit, metrics) |
| [`src/data/`](src/data/), [`src/features/`](src/features/), [`src/governance/`](src/governance/) | ingestion, Feast features, model cards |
| [`dags/`](dags/), [`testing/`](testing/) | thin Airflow DAGs + DagBag validation suite |
| [`kubernetes/`](kubernetes/), [`docker/`](docker/) | hardened deploy manifests + images |
| [`observability/`](observability/) | Prometheus + Grafana stack |
| [`tests/`](tests/) | 109 unit/integration tests incl. adversarial cases |

---

## Tech stack
`Python 3.11 · LightGBM · GARCH/arch · Apache Airflow · MLflow · Feast · Evidently ·
FastAPI · Docker · Kubernetes · MinIO/S3 · Prometheus · Grafana · pandera · DVC ·
GitHub Actions · Bandit · Semgrep · Trivy · cosign · CycloneDX`

## Data & license
Trip data: **NYC TLC open data** (public). Image paths (`ghcr.io/your-username/…`)
and infra endpoints are placeholders — set them for your own environment. Code
licensed under **MIT** — see [LICENSE](LICENSE).
