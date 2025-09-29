# Cloud Run service

resource "google_cloud_run_service" "app" {
  name     = "${var.service_name}-${var.environment}"
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.app_runtime.email

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ar_repo_name}/app:latest"

        resources {
          limits = {
            cpu    = "1"
            memory = "2Gi"
          }
        }

        env {
          name  = "GCS_MODEL_BUCKET"
          value = google_storage_bucket.model_bucket.name
        }

        env {
          name  = "GCS_DATA_BUCKET"
          value = google_storage_bucket.data_bucket.name
        }

        env {
          name  = "GCS_LOG_BUCKET"
          value = google_storage_bucket.log_bucket.name
        }

        env {
          name  = "FORCE_BASELINE"
          value = "false"
        }

        env {
          name  = "SECRET_RLHF_TOKEN"
          value = "projects/${var.project_id}/secrets/${google_secret_manager_secret.rlhf_token.secret_id}/versions/latest"
        }

        ports {
          container_port = 8080
        }

        # Liveness probe
        liveness_probe {
          http_get {
            path = "/healthz"
            port = 8080
          }
          initial_delay_seconds = 10
          period_seconds        = 30
          timeout_seconds       = 5
          failure_threshold     = 3
        }

        # Startup probe
        startup_probe {
          http_get {
            path = "/healthz"
            port = 8080
          }
          initial_delay_seconds = 0
          period_seconds        = 10
          timeout_seconds       = 5
          failure_threshold     = 10
        }
      }

      container_concurrency = 80
      timeout_seconds       = 60
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"
        "autoscaling.knative.dev/maxScale" = "10"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image,
      template[0].metadata[0].annotations["client.knative.dev/user-image"],
      template[0].metadata[0].annotations["run.googleapis.com/client-name"],
      template[0].metadata[0].annotations["run.googleapis.com/client-version"],
    ]
  }
}

# Allow unauthenticated access (adjust for production)
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.app.name
  location = google_cloud_run_service.app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

