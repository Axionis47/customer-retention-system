"""FastAPI serving application for churn retention."""
import logging
import os
import time
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from serve.policy_loader import PolicyLoader
from serve.healthchecks import run_all_health_checks
from rlhf.safety.shield import SafetyShield

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Churn Retention API",
    description="PPO + RLHF retention policy serving",
    version="0.1.0",
)

# Global state
policy_loader: Optional[PolicyLoader] = None
safety_shield: Optional[SafetyShield] = None
force_baseline: bool = False


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global policy_loader, safety_shield, force_baseline

    logger.info("Starting up Churn Retention API...")

    # Check kill switch
    force_baseline = os.getenv("FORCE_BASELINE", "false").lower() == "true"
    if force_baseline:
        logger.warning("FORCE_BASELINE is enabled - using baseline policy only")

    # Load models
    ppo_policy_path = os.getenv("PPO_POLICY_PATH", "checkpoints/ppo_policy.pth")
    rlhf_model_path = os.getenv("RLHF_MODEL_PATH", "checkpoints/ppo_text_model")
    use_gcs = os.getenv("GCS_MODEL_BUCKET") is not None

    try:
        policy_loader = PolicyLoader(
            ppo_policy_path=ppo_policy_path if not force_baseline else None,
            rlhf_model_path=rlhf_model_path if not force_baseline else None,
            use_gcs=use_gcs,
            quantize=True,
        )
        logger.info("Policy loader initialized")

    except Exception as e:
        logger.error(f"Failed to initialize policy loader: {e}")
        # Continue with baseline
        policy_loader = PolicyLoader()

    # Initialize safety shield
    try:
        safety_shield = SafetyShield()
        logger.info("Safety shield initialized")
    except Exception as e:
        logger.error(f"Failed to initialize safety shield: {e}")
        safety_shield = SafetyShield()  # Use defaults

    logger.info("Startup complete")


class CustomerRequest(BaseModel):
    """Request schema for retention endpoint."""

    customer_id: str = Field(..., description="Unique customer identifier")
    churn_risk: float = Field(..., ge=0, le=1, description="Churn risk probability")
    tenure_months: int = Field(..., ge=0, description="Months as customer")
    monthly_spend: float = Field(..., ge=0, description="Average monthly spend")
    contacts_last_7d: int = Field(default=0, ge=0, description="Contacts in last 7 days")
    days_since_last_contact: int = Field(default=30, ge=0, description="Days since last contact")
    cooldown_left: int = Field(default=0, ge=0, description="Cooldown days remaining")
    discount_budget_left: float = Field(default=1.0, ge=0, le=1, description="Budget remaining (normalized)")


class RetentionResponse(BaseModel):
    """Response schema for retention endpoint."""

    customer_id: str
    decision: str  # "contact" or "no_contact"
    offer_pct: float
    delay_days: int
    message: Optional[str] = None
    safety_flags: list = []
    using_baseline: bool = False
    latency_ms: float


@app.get("/healthz")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": "churn-retention-api"}


@app.get("/readyz")
async def readiness_check():
    """Readiness check with model validation."""
    if policy_loader is None:
        raise HTTPException(status_code=503, detail="Policy loader not initialized")

    # Run health checks
    health_results = run_all_health_checks(policy_loader)

    if not health_results["overall"]["healthy"]:
        raise HTTPException(status_code=503, detail=health_results)

    return {
        "status": "ready",
        "checks": health_results,
        "using_baseline": policy_loader.use_baseline(),
        "force_baseline": force_baseline,
    }


@app.post("/retain", response_model=RetentionResponse)
async def retain_customer(request: CustomerRequest, http_request: Request):
    """
    Main retention endpoint.

    Args:
        request: Customer context

    Returns:
        Retention decision and message
    """
    start_time = time.time()

    if policy_loader is None:
        raise HTTPException(status_code=503, detail="Policy loader not initialized")

    try:
        # Build observation
        obs = {
            "churn_risk": np.array([request.churn_risk], dtype=np.float32),
            "accept_prob_0": np.array([0.1], dtype=np.float32),  # Placeholder
            "accept_prob_1": np.array([0.3], dtype=np.float32),
            "accept_prob_2": np.array([0.5], dtype=np.float32),
            "accept_prob_3": np.array([0.7], dtype=np.float32),
            "days_since_last_contact": np.array([request.days_since_last_contact], dtype=np.float32),
            "contacts_last_7d": np.array([request.contacts_last_7d], dtype=np.float32),
            "cooldown_left": np.array([request.cooldown_left], dtype=np.float32),
            "discount_budget_left": np.array([request.discount_budget_left], dtype=np.float32),
        }

        # Predict action
        action = policy_loader.predict_action(obs)
        contact, offer_idx, delay_idx = action

        # Map to values
        offers = [0.0, 0.05, 0.10, 0.20]
        delays = [0, 1, 3]

        offer_pct = offers[offer_idx]
        delay_days = delays[delay_idx]

        # Generate message if contacting
        message = None
        safety_flags = []

        if contact == 1:
            customer_context = {
                "tenure_months": request.tenure_months,
                "monthly_spend": request.monthly_spend,
                "churn_risk": request.churn_risk,
            }

            message = policy_loader.generate_message(customer_context, offer_pct)

            # Safety check
            if safety_shield:
                is_safe, violations, penalty = safety_shield.check_message(message)
                if not is_safe:
                    safety_flags = violations
                    logger.warning(
                        f"Safety violations for customer {request.customer_id}: {violations}"
                    )

        # Log decision
        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Customer {request.customer_id}: "
            f"decision={'contact' if contact == 1 else 'no_contact'}, "
            f"offer={offer_pct * 100:.0f}%, "
            f"latency={latency_ms:.1f}ms"
        )

        return RetentionResponse(
            customer_id=request.customer_id,
            decision="contact" if contact == 1 else "no_contact",
            offer_pct=offer_pct,
            delay_days=delay_days,
            message=message,
            safety_flags=safety_flags,
            using_baseline=policy_loader.use_baseline() or force_baseline,
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Error processing request for {request.customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Churn Retention API",
        "version": "0.1.0",
        "endpoints": ["/healthz", "/readyz", "/retain"],
    }

