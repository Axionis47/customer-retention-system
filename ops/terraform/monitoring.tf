# Cloud Monitoring alerts and dashboards

# Log-based metric for 5xx errors
resource "google_logging_metric" "error_5xx" {
  name   = "churn_saver_5xx_errors"
  filter = <<-EOT
    resource.type="cloud_run_revision"
    resource.labels.service_name="${var.service_name}-${var.environment}"
    httpRequest.status>=500
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

# Log-based metric for safety violations
resource "google_logging_metric" "safety_violations" {
  name   = "churn_saver_safety_violations"
  filter = <<-EOT
    resource.type="cloud_run_revision"
    resource.labels.service_name="${var.service_name}-${var.environment}"
    jsonPayload.message=~"Safety violations"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

# Alert policy for high error rate
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "Churn Saver - High 5xx Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "5xx error rate > 5%"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.error_5xx.name}\" resource.type=\"cloud_run_revision\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = []  # Add notification channels here

  alert_strategy {
    auto_close = "1800s"
  }
}

# Alert policy for safety violations
resource "google_monitoring_alert_policy" "safety_violations" {
  display_name = "Churn Saver - Safety Violations Detected"
  combiner     = "OR"

  conditions {
    display_name = "Safety violations > 10 per minute"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.safety_violations.name}\" resource.type=\"cloud_run_revision\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = []  # Add notification channels here

  alert_strategy {
    auto_close = "1800s"
  }
}

