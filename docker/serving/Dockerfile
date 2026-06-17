FROM python:3.11-slim

WORKDIR /opt/app

# Install dependencies as root, then drop privileges. UID 1000 matches the
# Kubernetes Deployment's runAsUser so file ownership lines up.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --no-create-home --shell /bin/false --uid 1000 mlops \
    && chown -R mlops:mlops /opt/app

# Copy source code
COPY src/ /opt/app/src/
COPY feature_repo/ /opt/app/feature_repo/
COPY config.yaml /opt/app/config.yaml

EXPOSE 8080

# Drop to non-root user — container will not run as root in production
USER mlops

# Secrets (API_KEY, DB passwords) must be injected via Kubernetes Secrets, not baked in
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
