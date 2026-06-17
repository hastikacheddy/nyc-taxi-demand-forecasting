# Local Airflow image for inspecting/running the DAGs (development).
# Adds libgomp1 (LightGBM's OpenMP runtime) and the project's Python deps so all
# DAGs import cleanly, baked into the image for fast restarts.
FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    pandas==2.2.1 numpy==1.26.4 lightgbm==4.6.0 scikit-learn==1.9.0 scipy==1.14.0 \
    mlflow==3.13.0 pandera==0.31.1 arch==8.0.0 pyarrow==17.0.0 joblib==1.5.3 \
    statsd==4.0.1
