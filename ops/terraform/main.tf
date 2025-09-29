# Main Terraform configuration
# This file serves as the entry point and documents the module structure

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# All other resources are defined in separate files:
# - provider.tf: Provider configuration
# - variables.tf: Input variables
# - buckets.tf: GCS buckets
# - artifact_registry.tf: Docker repository
# - iam.tf: Service accounts and IAM bindings
# - secret_manager.tf: Secrets
# - cloud_run.tf: Cloud Run service
# - monitoring.tf: Alerts and metrics
# - outputs.tf: Output values

