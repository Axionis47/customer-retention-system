variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "data_bucket" {
  description = "GCS bucket for training data"
  type        = string
}

variable "model_bucket" {
  description = "GCS bucket for model artifacts"
  type        = string
}

variable "log_bucket" {
  description = "GCS bucket for logs"
  type        = string
}

variable "ar_repo_name" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "churn-saver-repo"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "churn-retain-api"
}

variable "environment" {
  description = "Environment (dev, test, prod)"
  type        = string
  default     = "dev"
}

