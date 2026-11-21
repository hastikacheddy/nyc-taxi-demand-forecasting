# AI Platform — Disaster Recovery

> Recovery objectives and procedures for the states that
> [RELIABILITY](RELIABILITY.md) classes as beyond graceful degradation — data
> loss, control-plane loss, region loss.

---

## 1. What we're protecting, and the targets

| Asset | Store | RPO (max loss) | RTO (max downtime) |
|---|---|---|---|
| Model artifacts | object storage (MinIO/S3/Blob), versioned + SHA-256 | 0 (immutable, versioned) | minutes (re-pull) |
| Model registry / metadata | MLflow DB | ≤ 24 h (daily backup) | < 1 h (restore) |
| Feature data / aggregates | DVC-tracked + object storage | ≤ 24 h | < 1 h |
| Serving config (manifests) | Git (this repo) | 0 (source of truth) | minutes (re-apply) |
| In-flight requests | none | n/a (stateless, at-least-once) | 0 (retry) |

The platform is **stateless in the serving path** — every durable thing is either
in Git or in versioned object storage. That is the property that makes recovery a
*redeploy*, not a *rebuild*.

---

## 2. Backup & restore (what exists)

The repo already ships infra-state **backup/restore scripts** (`observability/` /
`scripts/`). DR extends them with a fixed cadence and a *tested* restore:

- **Model registry** — nightly `mlflow` DB dump to object storage; restore =
  reload dump, re-point `MLFLOW_TRACKING_URI`.
- **Artifacts** — already immutable + versioned; bucket replication (cross-region)
  is the only add for regional durability.
- **Config** — Git *is* the backup; `kubectl apply -f kubernetes/` rebuilds the
  serving topology from scratch.

> **The untested backup is a liability, not an asset.** DR drills (restore into a
> scratch namespace quarterly) are part of the definition of done here.

---

## 3. Scenario runbooks

### 3.1 Control plane lost (MLflow / registry DB gone)
1. Serving continues — pods hold warmed weights (RELIABILITY row 5). No customer
   impact yet.
2. Restore MLflow DB from last nightly dump.
3. Reconcile: any model registered since the dump is re-registered from the
   artifact bucket (artifacts are the source of truth; the registry is an index).
4. Resume deploys.
**RTO < 1 h, RPO ≤ 24 h.**

### 3.2 Cluster lost (AKS/EKS control plane failure)
1. Re-create the cluster (Terraform, `infra/azure/` — planned) or fail to a warm
   standby cluster.
2. `kubectl apply -f kubernetes/` + `kubernetes/serving/` — topology is fully
   declarative.
3. Models re-pull from object storage on pod start.
**RTO tied to cluster provision time (~15–30 min with Terraform).**

### 3.3 Region lost
Out of scope for the single-region MVP; the designed path:
- Active/passive: cross-region bucket replication + a standby cluster + DNS/global
  LB failover.
- Cost/complexity trade stated explicitly — most internal platforms start
  single-region with tested backups and add region-DR when an SLA demands it.
This is documented as a **known gap with a plan**, not a silent omission.

### 3.4 Bad deploy corrupts production
- Rollback = re-point the KServe `storageUri`/revision to the previous version
  (RELIABILITY row 3). Because versions are immutable, rollback is instant and
  deterministic — there is no "rebuild the old model", it still exists.

---

## 4. The DR posture in one sentence

**Durable state lives in Git and versioned object storage; everything else is
cattle** — so disaster recovery is "re-apply declarative config and re-pull
immutable artifacts", and the only genuinely hard problem left (region failover)
is named, scoped, and deferred on purpose rather than by accident.
