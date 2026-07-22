terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.116"
    }
  }

  # Remote state belongs in a storage account, not local disk — uncomment and set
  # for a real deployment so state is shared, locked, and recoverable (DR).
  # backend "azurerm" {
  #   resource_group_name  = "tfstate-rg"
  #   storage_account_name = "mlopstfstate"
  #   container_name       = "tfstate"
  #   key                  = "ai-platform.tfstate"
  # }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false # never auto-purge secrets on destroy
    }
  }
}
