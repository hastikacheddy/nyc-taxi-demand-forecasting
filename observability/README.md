# Observability stack (Prometheus + Grafana)

Continuous metrics for the pilot — model latency, request outcomes, Airflow
task health, and shadow-DB pool saturation.

## What's wired
- The serving API already exposes `/metrics` (request counts + a latency
  histogram); the K8s manifests carry Prometheus scrape annotations.
- Airflow emits statsd metrics (`AIRFLOW__METRICS__STATSD_*` in
  `docker-compose.airflow.yml`) → **statsd-exporter** → **Prometheus**.
- **Grafana** auto-provisions the Prometheus datasource and the
  *MLOps — NYC Taxi Demand Forecasting* dashboard.

## Run it locally
```bash
# 1. Airflow first (creates the shared network + emits metrics)
docker compose -f docker-compose.airflow.yml up -d --build   # --build picks up statsd

# 2. The observability stack
docker compose -f docker-compose.observability.yml up -d
```
- Grafana:    http://localhost:3000  (admin / admin)
- Prometheus: http://localhost:9090

Trigger a few DAGs (or hit the serving API) to populate the panels.

## Production
Point a real Prometheus at the serving Pods (the scrape annotations are already
on `kubernetes/serving-deployment.yaml`) and deploy the Airflow statsd-exporter
sidecar; import `grafana/dashboards/mlops_overview.json`.
