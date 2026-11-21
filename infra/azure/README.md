# `infra/azure/` — AI Platform on Azure (Terraform)

The cloud substrate behind [ADR-001](../../docs/adr/001-why-kubernetes.md) and
[GPU_DESIGN](../../docs/platform/GPU_DESIGN.md), as infrastructure-as-code. "Runs
locally" becomes "provisioned, private, and GPU-scheduled" with `terraform apply`.

```
Azure Resource Group
├── VNet + subnet ............ private networking (Calico network policy)
├── AKS cluster
│   ├── system node pool ..... CPU: gateway, LightGBM serving, Airflow, Prometheus
│   └── gpu node pool ........ A100, TAINTED + labeled, scale-to-zero (min=0)
├── ACR (Premium) ............ images; AKS pulls via managed identity (no passwords)
├── Storage (Blob, GRS) ...... models, DVC remote, MLflow artifacts (geo-redundant)
├── Key Vault (RBAC) ......... secrets via CSI driver — none in env/images
└── Log Analytics ............ Azure Monitor sink alongside in-cluster Prometheus
```

## What maps to what

| Design doc says | This Terraform does |
|---|---|
| GPU pool repels non-GPU pods (GPU_DESIGN §1) | `node_taints = ["nvidia.com/gpu=present:NoSchedule"]` |
| KServe `nodeSelector: {accelerator: nvidia-a100}` | `node_labels = { accelerator = "nvidia-a100" }` |
| Scale-to-zero idle A100s (COST_MODEL §4) | GPU pool `min_count = 0`, autoscaling on |
| No secrets in images (SECURITY.md) | Key Vault + CSI secrets provider, RBAC auth |
| Regional durability (DISASTER_RECOVERY) | Blob `GRS`, ACR Premium geo-replication |
| Tenant isolation (SECURITY.md) | `network_policy = "calico"` |

## Use

```bash
az login
cd infra/azure
terraform init
terraform plan  -var 'prefix=myplatform'          # review
terraform apply -var 'prefix=myplatform'
$(terraform output -raw get_credentials_command)   # kubectl → new cluster

# then deploy the platform onto it:
kubectl apply -f ../../kubernetes/serving/gpu-priorityclasses.yaml
kubectl apply -f ../../kubernetes/serving/
```

## Status & honesty

- `terraform validate` **passes** against `azurerm ~> 3.116` (CI-checkable).
- Not applied to a live subscription here — `apply` needs your Azure credentials
  and, critically, **A100 (NC-A100-v4) quota** in the target region (request it
  first; it is not granted by default).
- Estimated cost is dominated by the GPU pool; with `gpu_node_min = 0` the idle
  cost is the CPU pool only (~2× `Standard_D4s_v5`). See
  [COST_MODEL](../../docs/platform/COST_MODEL.md).
- Remote state backend is stubbed in `versions.tf` — enable it before team use so
  state is shared and locked.
