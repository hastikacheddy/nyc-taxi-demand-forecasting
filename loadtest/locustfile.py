"""
Load / stress profile for the forecast serving API (Phase III).

Run against a deployed instance to validate latency and QPS thresholds before
exposing production traffic:

    pip install locust
    API_KEY=... locust -f loadtest/locustfile.py --host http://localhost:8080 \
        --users 100 --spawn-rate 10 --run-time 2m --headless

Acceptance (from the pilot plan): P95 latency < 1 min (the batch guardrail);
for the online API, target P95 well under 250 ms at the expected QPS.
"""
import os

from locust import HttpUser, task, between

_API_KEY = os.environ.get("API_KEY", "")
_HEADERS = {"X-API-Key": _API_KEY}


class ForecastUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict_daily(self):
        self.client.post("/predict", headers=_HEADERS, name="/predict[D]",
                         json={"time_period": "2026-03-31", "granularity": "D"})

    @task(3)
    def predict_hourly(self):
        self.client.post("/predict", headers=_HEADERS, name="/predict[H]",
                         json={"time_period": "2026-03-31", "granularity": "H"})

    @task(1)
    def health(self):
        self.client.get("/health")
