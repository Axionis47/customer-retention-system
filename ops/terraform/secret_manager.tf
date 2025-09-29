# Secret Manager for sensitive configuration

resource "google_secret_manager_secret" "rlhf_token" {
  secret_id = "rlhf-token"

  replication {
    automatic = true
  }

  labels = {
    environment = var.environment
  }
}

# Grant app runtime access to secrets
resource "google_secret_manager_secret_iam_member" "app_secret_accessor" {
  secret_id = google_secret_manager_secret.rlhf_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app_runtime.email}"
}

# Note: Secret versions must be created manually or via separate process
# Example: echo -n "your-token" | gcloud secrets versions add rlhf-token --data-file=-

