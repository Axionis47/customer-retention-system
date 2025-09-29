"""Policy loader for serving."""
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from agents.baselines.propensity_threshold import PropensityThresholdPolicy


class PolicyLoader:
    """
    Load and manage policies for serving.

    Supports:
        - Local file loading
        - GCS loading
        - Quantized inference (placeholder)
        - Fallback to baseline
    """

    def __init__(
        self,
        ppo_policy_path: Optional[str] = None,
        rlhf_model_path: Optional[str] = None,
        use_gcs: bool = False,
        quantize: bool = False,
    ):
        """
        Initialize policy loader.

        Args:
            ppo_policy_path: Path to PPO policy checkpoint
            rlhf_model_path: Path to RLHF model checkpoint
            use_gcs: Whether to load from GCS
            quantize: Whether to use quantized inference
        """
        self.ppo_policy_path = ppo_policy_path
        self.rlhf_model_path = rlhf_model_path
        self.use_gcs = use_gcs
        self.quantize = quantize

        self.ppo_policy = None
        self.rlhf_model = None
        self.baseline_policy = PropensityThresholdPolicy()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        self._load_models()

    def _load_from_gcs(self, gcs_path: str, local_path: str):
        """Download file from GCS to local path."""
        try:
            from google.cloud import storage

            # Parse gs://bucket/path
            parts = gcs_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ""

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Download
            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))

            print(f"Downloaded {gcs_path} to {local_path}")
            return str(local_file)

        except Exception as e:
            print(f"Failed to load from GCS: {e}")
            return None

    def _load_models(self):
        """Load PPO and RLHF models."""
        # Load PPO policy
        if self.ppo_policy_path:
            try:
                if self.use_gcs and self.ppo_policy_path.startswith("gs://"):
                    local_path = "/tmp/ppo_policy.pth"
                    self.ppo_policy_path = self._load_from_gcs(self.ppo_policy_path, local_path)

                if self.ppo_policy_path and Path(self.ppo_policy_path).exists():
                    # Load PPO policy (simplified - would need full PolicyNetwork class)
                    # For now, use baseline
                    print(f"PPO policy path exists: {self.ppo_policy_path}")
                    # self.ppo_policy = torch.load(self.ppo_policy_path, map_location=self.device)
                    self.ppo_policy = self.baseline_policy  # Fallback for now
                else:
                    print("PPO policy not found, using baseline")
                    self.ppo_policy = self.baseline_policy

            except Exception as e:
                print(f"Failed to load PPO policy: {e}")
                self.ppo_policy = self.baseline_policy

        else:
            print("No PPO policy path provided, using baseline")
            self.ppo_policy = self.baseline_policy

        # Load RLHF model (optional)
        if self.rlhf_model_path:
            try:
                if self.use_gcs and self.rlhf_model_path.startswith("gs://"):
                    local_path = "/tmp/rlhf_model"
                    self._load_from_gcs(self.rlhf_model_path, local_path)

                # Load RLHF model (placeholder)
                print(f"RLHF model path: {self.rlhf_model_path}")
                # self.rlhf_model = load_rlhf_model(self.rlhf_model_path)

            except Exception as e:
                print(f"Failed to load RLHF model: {e}")

    def predict_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict action using loaded policy.

        Args:
            obs: Observation dict

        Returns:
            action: [contact, offer_idx, delay_idx]
        """
        try:
            if self.ppo_policy:
                return self.ppo_policy(obs)
            else:
                return self.baseline_policy(obs)

        except Exception as e:
            print(f"Policy prediction failed: {e}, using baseline")
            return self.baseline_policy(obs)

    def generate_message(
        self,
        customer_context: Dict,
        offer_pct: float,
    ) -> str:
        """
        Generate retention message using RLHF model.

        Args:
            customer_context: Customer context dict
            offer_pct: Discount percentage (0-1)

        Returns:
            Generated message
        """
        try:
            if self.rlhf_model:
                # Use RLHF model (placeholder)
                # message = self.rlhf_model.generate(customer_context, offer_pct)
                pass

            # Fallback: template-based message
            message = self._template_message(customer_context, offer_pct)
            return message

        except Exception as e:
            print(f"Message generation failed: {e}, using template")
            return self._template_message(customer_context, offer_pct)

    def _template_message(self, customer_context: Dict, offer_pct: float) -> str:
        """Generate template-based message."""
        tenure = customer_context.get("tenure_months", 0)

        if tenure > 24:
            greeting = "As a valued long-term customer"
        elif tenure > 12:
            greeting = "We appreciate your loyalty"
        else:
            greeting = "Thank you for being with us"

        message = (
            f"{greeting}, we'd like to offer you {offer_pct * 100:.0f}% off "
            f"for the next 3 months. Let us know if you're interested!"
        )

        return message

    def use_baseline(self) -> bool:
        """Check if using baseline policy."""
        return isinstance(self.ppo_policy, PropensityThresholdPolicy)

