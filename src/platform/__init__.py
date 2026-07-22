"""
Internal AI Platform layer.

This is the *control plane* that sits above the raw forecasting model and turns
it into a multi-tenant, self-service platform — the same abstraction a team runs
internally on top of Vertex AI / SageMaker / Databricks Model Serving:

    developers → internal AI APIs → platform layer → compute / serving / data

The public surface is a small, framework-agnostic API:

    POST /v1/models            register a model version (any framework)
    GET  /v1/models            list registered models
    POST /v1/deployments       create a deployment (picks CPU vs GPU pool)
    GET  /v1/deployments/{id}  deployment status / health
    POST /v1/inference         route a request to a live deployment
    GET  /v1/metrics           Prometheus scrape

The point of the layer is *abstraction*: a caller asks for "model X, latest" and
the platform handles registry lookup, compute-pool placement, canary traffic
splitting, backend selection (LightGBM vs vLLM vs Transformers), and failure
fallback — without the caller knowing any of it.
"""

from src.platform.registry import ModelRegistry, ModelVersion, Framework
from src.platform.deployments import Deployment, DeploymentManager, ComputePool
from src.platform import backends

__all__ = [
    "ModelRegistry",
    "ModelVersion",
    "Framework",
    "Deployment",
    "DeploymentManager",
    "ComputePool",
    "backends",
]
