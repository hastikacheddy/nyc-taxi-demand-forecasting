FROM python:3.11-slim

WORKDIR /opt/app

# Install dependencies as root, then drop privileges
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --no-create-home --shell /bin/false mlops \
    && chown -R mlops:mlops /opt/app

# Copy source code
COPY src/ /opt/app/src/
COPY config.yaml /opt/app/config.yaml

# Drop to non-root user — container will not run as root in production
USER mlops

# Secrets (API_KEY, DB passwords) must be injected via Kubernetes Secrets, not baked in
