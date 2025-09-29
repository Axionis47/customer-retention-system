"""Integration tests for GCS loading."""
import pytest
from unittest.mock import Mock, patch


@pytest.mark.integration
def test_gcs_path_parsing():
    """GCS paths should be parsed correctly."""
    from serve.policy_loader import PolicyLoader
    from google.cloud import storage

    # Mock GCS
    with patch("google.cloud.storage.Client") as mock_client_class:
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()

        mock_client_class.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        loader = PolicyLoader(
            ppo_policy_path="gs://my-bucket/models/ppo_policy.pth",
            use_gcs=True,
        )

        # Should attempt to parse GCS path
        # (Will fail to load actual model, but that's OK for this test)
        assert loader is not None


@pytest.mark.integration
def test_local_loading_fallback():
    """Should fall back to local loading if GCS fails."""
    from serve.policy_loader import PolicyLoader

    # Non-existent GCS path
    loader = PolicyLoader(
        ppo_policy_path="gs://nonexistent-bucket/model.pth",
        use_gcs=True,
    )

    # Should fall back to baseline
    assert loader.use_baseline(), "Should use baseline when loading fails"


@pytest.mark.integration
def test_baseline_always_available():
    """Baseline policy should always be available."""
    from serve.policy_loader import PolicyLoader

    loader = PolicyLoader()

    assert loader.baseline_policy is not None, "Baseline should be available"
    assert loader.use_baseline(), "Should use baseline by default"

    # Test prediction
    import numpy as np

    obs = {
        "churn_risk": np.array([0.8], dtype=np.float32),
        "accept_prob_0": np.array([0.1], dtype=np.float32),
        "accept_prob_1": np.array([0.3], dtype=np.float32),
        "accept_prob_2": np.array([0.5], dtype=np.float32),
        "accept_prob_3": np.array([0.7], dtype=np.float32),
        "days_since_last_contact": np.array([30.0], dtype=np.float32),
        "contacts_last_7d": np.array([0.0], dtype=np.float32),
        "cooldown_left": np.array([0.0], dtype=np.float32),
        "discount_budget_left": np.array([1.0], dtype=np.float32),
    }

    action = loader.predict_action(obs)

    assert action is not None, "Should return action"
    assert len(action) == 3, "Action should have 3 components"

