"""FastAPI serving application for churn retention."""
import logging
import os
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
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


class PolicyOverrides(BaseModel):
    """Policy override options."""
    force_baseline: bool = False


class RetainRequest(BaseModel):
    """Request schema for /retain endpoint."""
    customer_facts: Dict = Field(..., description="Arbitrary customer key/value pairs")
    policy_overrides: Optional[PolicyOverrides] = None
    debug: bool = False


class DecisionOutput(BaseModel):
    """Decision output schema."""
    contact: bool
    offer_level: int = Field(..., ge=0, le=3)
    followup_days: int = Field(..., ge=0)


class ScoresOutput(BaseModel):
    """Model scores output schema."""
    p_churn: float = Field(..., ge=0, le=1)
    p_accept: list[float] = Field(..., description="Acceptance probabilities for offer levels 0-3")


class SafetyOutput(BaseModel):
    """Safety check output schema."""
    violations: int = 0
    applied_disclaimers: list[str] = []


class RetainResponse(BaseModel):
    """Response schema for /retain endpoint."""
    decision: DecisionOutput
    scores: ScoresOutput
    message: str
    safety: SafetyOutput


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


@app.post("/retain", response_model=RetainResponse)
async def retain_customer(request: RetainRequest):
    """
    Main retention endpoint.

    Args:
        request: Customer facts and policy overrides

    Returns:
        Retention decision, scores, message, and safety info
    """
    if policy_loader is None:
        raise HTTPException(status_code=503, detail="Policy loader not initialized")

    try:
        facts = request.customer_facts
        overrides = request.policy_overrides or PolicyOverrides()
        use_baseline = overrides.force_baseline or force_baseline

        # Extract or compute scores
        # In production, these would come from risk/acceptance models
        p_churn = facts.get("churn_risk", 0.5)
        if not (0 <= p_churn <= 1):
            p_churn = 0.5

        # Compute acceptance probabilities for each offer level
        # Stub: higher offers = higher acceptance
        base_accept = 0.1
        p_accept = [
            base_accept,
            base_accept + 0.1,
            base_accept + 0.2,
            base_accept + 0.3,
        ]

        # Decision policy
        if use_baseline:
            # Baseline: simple threshold
            contact = p_churn > 0.4
            offer_level = 2 if p_churn > 0.6 else 1
            followup_days = 7
        else:
            # Use learned policy (stub for now)
            obs = {
                "churn_risk": np.array([p_churn], dtype=np.float32),
                "accept_prob_0": np.array([p_accept[0]], dtype=np.float32),
                "accept_prob_1": np.array([p_accept[1]], dtype=np.float32),
                "accept_prob_2": np.array([p_accept[2]], dtype=np.float32),
                "accept_prob_3": np.array([p_accept[3]], dtype=np.float32),
                "days_since_last_contact": np.array([facts.get("days_since_last_contact", 30)], dtype=np.float32),
                "contacts_last_7d": np.array([facts.get("contacts_last_7d", 0)], dtype=np.float32),
                "cooldown_left": np.array([facts.get("cooldown_left", 0)], dtype=np.float32),
                "discount_budget_left": np.array([facts.get("discount_budget_left", 1.0)], dtype=np.float32),
            }

            action = policy_loader.predict_action(obs)
            contact_int, offer_idx, delay_idx = action

            contact = contact_int == 1
            offer_level = offer_idx
            followup_days = [3, 7, 14][delay_idx] if delay_idx < 3 else 7

        # Generate message
        if contact:
            customer_context = {
                "tenure_months": facts.get("tenure", facts.get("tenure_months", 12)),
                "monthly_spend": facts.get("spend", facts.get("monthly_spend", 50.0)),
                "churn_risk": p_churn,
                "name": facts.get("name", "valued customer"),
            }

            offer_pct = [0.0, 0.05, 0.10, 0.20][offer_level]
            message = policy_loader.generate_message(customer_context, offer_pct)

            # Safety check
            violations_list = []
            disclaimers = []

            if safety_shield:
                is_safe, violations, _ = safety_shield.check_message(message)
                if not is_safe:
                    violations_list = violations
                    logger.warning(f"Safety violations: {violations}")

                # Add disclaimers
                disclaimers.append("Offer valid until end of month")

        else:
            message = ""
            violations_list = []
            disclaimers = []

        # Build response
        return RetainResponse(
            decision=DecisionOutput(
                contact=contact,
                offer_level=offer_level,
                followup_days=followup_days,
            ),
            scores=ScoresOutput(
                p_churn=p_churn,
                p_accept=p_accept,
            ),
            message=message,
            safety=SafetyOutput(
                violations=len(violations_list),
                applied_disclaimers=disclaimers,
            ),
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Churn Retention API",
        "version": "0.1.0",
        "endpoints": ["/healthz", "/readyz", "/retain"],
    }

