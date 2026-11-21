# `kubernetes/serving/` — GPU-aware, KServe-native serving

The cluster-side actuation of the platform control plane (`src/platform/`). Where
the in-process `DeploymentManager` *simulates* placement and canary routing, these
manifests are how it happens on a real cluster.

| File | Kind | Purpose |
|---|---|---|
| `gpu-priorityclasses.yaml` | PriorityClass ×3 | serving > batch > training; serving may evict training |
| `gpu-resourcequota.yaml` | ResourceQuota + LimitRange | cap GPUs per tenant namespace (multi-tenancy guardrail) |
| `kserve-lightgbm.yaml` | InferenceService | forecaster on the **CPU** pool, canary 20% |
| `kserve-llm-vllm.yaml` | InferenceService | LLM via **vLLM** on the **GPU** pool, scale-on-concurrency |

The parent [`kubernetes/`](../) folder holds the already-hardened plain-Deployment
serving path (`serving-deployment.yaml`) and the OPA admission policy. This folder
is the *next tier*: KServe/Knative-managed, GPU-scheduled, autoscaled on
concurrency.

## Apply order (on a cluster with KServe + a GPU node pool)

```bash
kubectl apply -f gpu-priorityclasses.yaml      # cluster-scoped
kubectl apply -f gpu-resourcequota.yaml        # namespace mlops
kubectl apply -f kserve-lightgbm.yaml
kubectl apply -f kserve-llm-vllm.yaml
kubectl get inferenceservices -n mlops
```

## Prerequisites (documented, not assumed)

- KServe + Knative Serving installed.
- A GPU node pool tainted `nvidia.com/gpu=present:NoSchedule` and labeled
  `accelerator=nvidia-a100` (and/or `nvidia-l4`).
- NVIDIA device plugin + DCGM exporter (GPU metrics → Prometheus).
- Secrets `mlops-hf-secrets` (HF token) and object-storage creds for `storageUri`.

Design rationale for every choice here: [docs/platform/GPU_DESIGN.md](../../docs/platform/GPU_DESIGN.md).
