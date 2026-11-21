variable "prefix" {
  description = "Name prefix for all resources."
  type        = string
  default     = "mlops-aiplat"
}

variable "location" {
  description = "Azure region. Must have A100 (NC-A100-v4) quota for the GPU pool."
  type        = string
  default     = "eastus"
}

variable "kubernetes_version" {
  type    = string
  default = "1.29"
}

# ── CPU (system) node pool ────────────────────────────────────────
variable "cpu_node_size" {
  type    = string
  default = "Standard_D4s_v5"
}

variable "cpu_node_min" {
  type    = number
  default = 2
}

variable "cpu_node_max" {
  type    = number
  default = 6
}

# ── GPU node pool ─────────────────────────────────────────────────
variable "gpu_node_size" {
  description = "A100 VM. Verify regional quota before apply."
  type        = string
  default     = "Standard_NC24ads_A100_v4"
}

variable "gpu_node_min" {
  description = "0 enables scale-to-zero — no idle A100 cost when there is no GPU work."
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  type    = number
  default = 3
}

variable "tags" {
  type = map(string)
  default = {
    project = "nyc-taxi-ai-platform"
    owner   = "platform-eng"
    managed = "terraform"
  }
}
