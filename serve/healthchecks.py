"""Health check utilities for serving."""
import os
from typing import Dict, Tuple


def check_models_loaded(policy_loader) -> Tuple[bool, str]:
    """
    Check if models are loaded successfully.

    Args:
        policy_loader: PolicyLoader instance

    Returns:
        (is_healthy, message)
    """
    if policy_loader.ppo_policy is None:
        return False, "PPO policy not loaded"

    # RLHF model is optional
    # if policy_loader.rlhf_model is None:
    #     return False, "RLHF model not loaded"

    return True, "Models loaded successfully"


def check_secrets() -> Tuple[bool, str]:
    """
    Check if required secrets are accessible.

    Returns:
        (is_healthy, message)
    """
    # Check for required env vars
    required_vars = ["GCS_MODEL_BUCKET"]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"

    return True, "Secrets accessible"


def check_gcs_access() -> Tuple[bool, str]:
    """
    Check if GCS is accessible.

    Returns:
        (is_healthy, message)
    """
    try:
        from google.cloud import storage

        bucket_name = os.getenv("GCS_MODEL_BUCKET")
        if not bucket_name:
            return True, "GCS not configured (using local models)"

        # Try to access bucket (lightweight check)
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Check if bucket exists (doesn't fetch all objects)
        exists = bucket.exists()

        if not exists:
            return False, f"GCS bucket {bucket_name} not accessible"

        return True, "GCS accessible"

    except Exception as e:
        return False, f"GCS check failed: {str(e)}"


def run_all_health_checks(policy_loader) -> Dict[str, Dict]:
    """
    Run all health checks.

    Args:
        policy_loader: PolicyLoader instance

    Returns:
        Dict with check results
    """
    checks = {
        "models": check_models_loaded(policy_loader),
        "secrets": check_secrets(),
        "gcs": check_gcs_access(),
    }

    results = {}
    overall_healthy = True

    for check_name, (is_healthy, message) in checks.items():
        results[check_name] = {
            "healthy": is_healthy,
            "message": message,
        }
        if not is_healthy:
            overall_healthy = False

    results["overall"] = {
        "healthy": overall_healthy,
        "message": "All checks passed" if overall_healthy else "Some checks failed",
    }

    return results

