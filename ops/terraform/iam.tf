# Service accounts and IAM bindings

# Service account for Cloud Run app
resource "google_service_account" "app_runtime" {
  account_id   = "app-runtime"
  display_name = "Churn Saver App Runtime"
  description  = "Service account for Cloud Run application"
}

# Service account for Cloud Build
resource "google_service_account" "ci_builder" {
  account_id   = "ci-builder"
  display_name = "Churn Saver CI Builder"
  description  = "Service account for Cloud Build CI/CD"
}

# Grant app runtime access to model bucket (read)
resource "google_storage_bucket_iam_member" "app_model_reader" {
  bucket = google_storage_bucket.model_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.app_runtime.email}"
}

# Grant app runtime access to data bucket (read)
resource "google_storage_bucket_iam_member" "app_data_reader" {
  bucket = google_storage_bucket.data_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.app_runtime.email}"
}

# Grant app runtime access to log bucket (write)
resource "google_storage_bucket_iam_member" "app_log_writer" {
  bucket = google_storage_bucket.log_bucket.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.app_runtime.email}"
}

# Grant CI builder access to Artifact Registry (push)
resource "google_artifact_registry_repository_iam_member" "ci_ar_writer" {
  location   = google_artifact_registry_repository.docker_repo.location
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.ci_builder.email}"
}

# Grant CI builder Cloud Run admin (for deployment)
resource "google_project_iam_member" "ci_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.ci_builder.email}"
}

# Grant CI builder service account user (to act as app runtime)
resource "google_service_account_iam_member" "ci_sa_user" {
  service_account_id = google_service_account.app_runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.ci_builder.email}"
}

