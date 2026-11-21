# AI Platform — Security

> The platform inherits a hardened baseline from the existing repo and extends it
> to the multi-tenant, GPU, and LLM surfaces. This doc is the consolidated view;
> it does not re-derive what the code already enforces, it *maps* it.

---

## 1. Inherited baseline (already enforced in code)

| Control | Where |
|---|---|
| Non-root pods, `readOnlyRootFilesystem`, drop ALL caps, seccomp `RuntimeDefault`, no SA token | `kubernetes/serving-deployment.yaml` |
| OPA/Rego admission policy enforcing the least-privilege pod baseline | `kubernetes/opa-policy.rego` |
| API-key auth (constant-time compare), 60 req/min rate limit, audit log, OpenAPI off in prod | `src/serving/api.py` |
| Model-artifact SHA-256 integrity check before load (defends tampered pickles) | `src/inference/model_integrity.py` |
| SBOM/AI-BOM, pip-audit, Trivy (fail on CRITICAL), cosign image signing | `.github/workflows/` |
| PII scanner + pandera data-quality gates | `src/data/pii.py`, `src/data/quality_gate.py` |

The platform gateway (`src/platform/gateway.py`) reuses the same constant-time
API-key pattern; the KServe manifests carry the same pod hardening.

---

## 2. New surfaces the platform introduces

### 2.1 Multi-tenancy (the big new risk)
Many teams, one cluster → isolation is now a security property, not just a cost one.
- **GPU `ResourceQuota`** per namespace — a tenant can't exhaust the accelerator pool (DoS-by-greed).
- **PriorityClasses** — one team's training can't preempt another's serving.
- **NetworkPolicies** (recommended next) — default-deny between tenant namespaces;
  the gateway is the only ingress to model pods.
- **Per-model API scoping** — the platform key model should evolve from one shared
  key to per-tenant keys / OIDC so inference calls are attributable and revocable.

### 2.2 LLM-specific threats (OWASP LLM Top 10, the relevant subset)
| Threat | Mitigation posture |
|---|---|
| Prompt injection | input/output guardrails at the LLM gateway (planned LLMOps layer); treat model output as untrusted |
| Sensitive-data disclosure | PII scanning on prompts/completions; no secrets in system prompts |
| Model DoS (expensive prompts) | `max_tokens` cap + rate limit + concurrency bound in `kserve-llm-vllm.yaml` |
| Supply chain (model weights) | pull from trusted registry; checksum weights; the SHA-256 discipline extends to LLM artifacts |
| Insecure output handling | never eval/exec model output; schema-validate structured outputs |

### 2.3 Secrets & keys
Existing pattern (env/secretRef only, Key Vault-backed on Azure) extends to the HF
token (`mlops-hf-secrets` in the vLLM manifest) and object-storage creds. No
secret is ever baked into an image or a model artifact.

---

## 3. Threat model in one paragraph

The adversaries that matter for an internal AI platform are (1) a **compromised or
greedy tenant** — contained by quotas, priorities, network policy, and
attributable keys; (2) a **tampered model artifact** — contained by fail-closed
SHA-256 verification and signed images; and (3) **malicious input to an LLM** —
contained by guardrails, output distrust, and token/rate caps. Physical/cluster
compromise is delegated to the cloud provider's shared-responsibility boundary
([ADR-001](../adr/001-why-kubernetes.md)).
