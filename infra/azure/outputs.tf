output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "acr_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "artifacts_storage_account" {
  value = azurerm_storage_account.artifacts.name
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}

output "get_credentials_command" {
  description = "Run this to point kubectl at the new cluster."
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.main.name} --name ${azurerm_kubernetes_cluster.main.name}"
}
