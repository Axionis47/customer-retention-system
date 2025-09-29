# Terraform outputs

output "service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_service.app.status[0].url
}

output "data_bucket_name" {
  description = "GCS data bucket name"
  value       = google_storage_bucket.data_bucket.name
}

output "model_bucket_name" {
  description = "GCS model bucket name"
  value       = google_storage_bucket.model_bucket.name
}

output "log_bucket_name" {
  description = "GCS log bucket name"
  value       = google_storage_bucket.log_bucket.name
}

output "artifact_registry_repo" {
  description = "Artifact Registry repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ar_repo_name}"
}

output "app_runtime_sa_email" {
  description = "App runtime service account email"
  value       = google_service_account.app_runtime.email
}

output "ci_builder_sa_email" {
  description = "CI builder service account email"
  value       = google_service_account.ci_builder.email
}

output "secret_rlhf_token_id" {
  description = "RLHF token secret ID"
  value       = google_secret_manager_secret.rlhf_token.secret_id
}

