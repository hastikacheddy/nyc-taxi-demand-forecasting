# ─────────────────────────────────────────────────────────────────
# AI Platform on Azure — the concrete provisioning behind ADR-001.
#
#   AKS (system CPU pool + tainted GPU pool) · ACR · Blob · Key Vault · Monitor
#   · private VNet. This is the "runs locally is not enough" cloud story: the
#   GPU/CPU pool split, taints, and scale-to-zero from docs/platform/GPU_DESIGN.md
#   expressed as infrastructure-as-code.
# ─────────────────────────────────────────────────────────────────
data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "main" {
  name     = "${var.prefix}-rg"
  location = var.location
  tags     = var.tags
}

# ── Networking (private by default) ───────────────────────────────
resource "azurerm_virtual_network" "main" {
  name                = "${var.prefix}-vnet"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.20.0.0/16"]
  tags                = var.tags
}

resource "azurerm_subnet" "aks" {
  name                 = "aks"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.20.0.0/20"]
}

# ── Observability sink (Azure Monitor / Log Analytics) ────────────
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.prefix}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

# ── Container registry (ACR) ──────────────────────────────────────
resource "azurerm_container_registry" "main" {
  name                = replace("${var.prefix}acr", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Premium" # Premium = private endpoints + geo-replication (DR)
  admin_enabled       = false     # no admin user; AKS pulls via managed identity
  tags                = var.tags
}

# ── AKS cluster ───────────────────────────────────────────────────
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.prefix}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${var.prefix}-aks"
  kubernetes_version  = var.kubernetes_version
  tags                = var.tags

  # System (CPU) pool: gateway, LightGBM serving, Airflow, Prometheus.
  default_node_pool {
    name                         = "system"
    vm_size                      = var.cpu_node_size
    vnet_subnet_id               = azurerm_subnet.aks.id
    enable_auto_scaling          = true
    min_count                    = var.cpu_node_min
    max_count                    = var.cpu_node_max
    orchestrator_version         = var.kubernetes_version
    only_critical_addons_enabled = false
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "calico" # default-deny between tenant namespaces (SECURITY.md)
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  # Key Vault CSI driver → mount secrets as files, no secrets in env/images.
  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }
}

# ── GPU node pool (the accelerator tier) ──────────────────────────
# Tainted + labeled exactly as kubernetes/serving/ expects, and scale-to-zero
# capable so idle A100s cost nothing. This is GPU_DESIGN.md as IaC.
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.gpu_node_size
  vnet_subnet_id        = azurerm_subnet.aks.id

  enable_auto_scaling = true
  min_count           = var.gpu_node_min # 0 → scale-to-zero
  max_count           = var.gpu_node_max

  # Repel non-GPU pods; only workloads that tolerate this land here.
  node_taints = ["nvidia.com/gpu=present:NoSchedule"]
  # Matches nodeSelector: { accelerator: nvidia-a100 } in the KServe manifests.
  node_labels = {
    accelerator = "nvidia-a100"
    pool        = "gpu"
  }
  tags = var.tags
}

# ── AKS → ACR pull permission (no passwords) ──────────────────────
resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                            = azurerm_container_registry.main.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  skip_service_principal_aad_check = true
}

# ── Artifact storage (Blob) — models, DVC remote, MLflow artifacts ─
resource "azurerm_storage_account" "artifacts" {
  name                            = replace("${var.prefix}art", "-", "")
  resource_group_name             = azurerm_resource_group.main.name
  location                        = azurerm_resource_group.main.location
  account_tier                    = "Standard"
  account_replication_type        = "GRS" # geo-redundant → regional durability (DR)
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  tags                            = var.tags
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.artifacts.name
  container_access_type = "private"
}

# ── Key Vault (secrets: API keys, HF token, storage creds) ────────
resource "azurerm_key_vault" "main" {
  name                       = replace("${var.prefix}-kv", "_", "-")
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  purge_protection_enabled   = true
  soft_delete_retention_days = 30
  enable_rbac_authorization  = true # RBAC, not access policies
  tags                       = var.tags
}

# Let the AKS Key Vault CSI identity read secrets.
resource "azurerm_role_assignment" "aks_kv_reader" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_kubernetes_cluster.main.key_vault_secrets_provider[0].secret_identity[0].object_id
}
