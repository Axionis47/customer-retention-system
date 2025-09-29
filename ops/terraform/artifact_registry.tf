# Artifact Registry for Docker images

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.ar_repo_name
  description   = "Docker repository for churn-saver images"
  format        = "DOCKER"

  labels = {
    environment = var.environment
  }
}

