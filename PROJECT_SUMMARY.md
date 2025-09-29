# Churn-Saver RLHF+PPO - Project Summary

## Overview

This repository contains a production-grade ML system that combines **PPO (Proximal Policy Optimization)** for churn retention decisions with **RLHF (Reinforcement Learning from Human Feedback)** for personalized message generation. The system is fully GCP-native with comprehensive testing, CI/CD, and infrastructure-as-code.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud Run Service                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ FastAPI App  │→ │ PPO Policy   │→ │ RLHF Generator  │  │
│  │ /retain      │  │ (Decision)   │  │ (Message)       │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│         ↓                  ↓                    ↓           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Safety Shield + Kill Switch                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↓
              ┌────────────────────────┐
              │   GCS Model Storage    │
              │  - Risk models         │
              │  - PPO checkpoints     │
              │  - RLHF models         │
              └────────────────────────┘
```

## Key Components

### 1. Retention Environment (`env/retention_env.py`)
- Gymnasium-compatible RL environment
- Multi-discrete action space: [contact, offer, delay]
- Reward function with Lagrangian penalties
- Constraint tracking: budget, cooldown, fatigue

### 2. PPO Decision Policy (`agents/ppo_policy.py`)
- Actor-Critic architecture
- GAE (Generalized Advantage Estimation)
- Clipped surrogate loss
- Entropy bonus for exploration

### 3. Lagrangian Constraints (`agents/lagrangian.py`)
- Dual ascent for constraint satisfaction
- Adaptive multipliers for budget/cooldown/fatigue
- Penalty computation

### 4. Baseline Policies (`agents/baselines/`)
- Propensity threshold
- Uplift trees
- Thompson Sampling bandit

### 5. RLHF Pipeline (`rlhf/`)
- **SFT**: Supervised fine-tuning with QLoRA
- **RM**: Reward model with Bradley-Terry loss
- **PPO-Text**: PPO for text generation with adaptive KL

### 6. Safety Shield (`rlhf/safety/`)
- Rule-based filtering (banned phrases, length, quiet hours)
- Toxicity detection
- Required elements validation

### 7. Serving Layer (`serve/`)
- FastAPI application
- Health checks (`/healthz`, `/readyz`)
- Retention endpoint (`/retain`)
- Policy loader with GCS support
- Kill switch (FORCE_BASELINE)

### 8. Evaluation Suite (`eval/`)
- Business metrics (NRR, ROI, violation rate)
- Stress tests (budget/churn/accept shifts)
- Message arena (A/B testing)
- Visualization plots

## Test Coverage

**46 tests** across 4 categories:

### Unit Tests (27 tests)
- Reward computation
- Constraint enforcement
- Lagrangian updates
- Bradley-Terry loss
- KL adaptation
- Safety rules

### Integration Tests (16 tests)
- Environment rollouts
- PPO training
- API endpoints
- GCS loading

### Contract Tests (3 tests)
- Golden policy outputs
- Deterministic trajectories
- Safety shield consistency

### E2E Tests
- Docker smoke tests
- Full deployment validation

**All tests pass ✓**

## Infrastructure (Terraform)

Located in `ops/terraform/`:

- **GCS Buckets**: Data, models, logs (with versioning & lifecycle)
- **Artifact Registry**: Docker repository
- **Service Accounts**: app-runtime, ci-builder (least-privilege IAM)
- **Secret Manager**: RLHF tokens
- **Cloud Run**: Service with autoscaling (0-10 instances)
- **Monitoring**: Log-based metrics and alerts

## CI/CD Pipeline (Cloud Build)

Located in `ops/cloudbuild.yaml`:

1. **Lint**: ruff, mypy
2. **Test**: Unit, integration, contract tests with coverage
3. **Build**: Docker images (app + trainer)
4. **Scan**: Trivy vulnerability scanning
5. **Push**: Artifact Registry
6. **Deploy**: Cloud Run (dev)
7. **E2E**: Smoke tests against deployed service

## Configuration

All hyperparameters in `ops/configs/*.yaml`:

- `env.yaml`: Episode length, budget, cooldown, fatigue
- `ppo.yaml`: Learning rate, gamma, GAE lambda, clip epsilon
- `sft.yaml`: Model name, LoRA config
- `rm.yaml`: Bradley-Terry margin
- `ppo_text.yaml`: KL target, beta
- `serve.yaml`: Quantization, timeout

## Default Seeds

All RNG seeds default to **42** for reproducibility. Override with `--seed` flag.

## Quick Start

```bash
# 1. Setup
make setup

# 2. Run tests
make test

# 3. Prepare data
make prepare-data

# 4. Train models
make train-risk
make train-accept
make train-ppo

# 5. Serve locally
make serve

# 6. Deploy to GCP
cd ops/terraform
terraform init
terraform apply
cd ../..
gcloud builds submit --config ops/cloudbuild.yaml
```

## Acceptance Criteria Status

✅ **All 12 requirements met:**

1. ✅ Repository structure with all directories
2. ✅ Risk & acceptance models (XGBoost + calibration)
3. ✅ Retention environment (Gymnasium)
4. ✅ PPO policy with Lagrangian constraints
5. ✅ Baseline policies (3 types)
6. ✅ RLHF pipeline (SFT → RM → PPO-text)
7. ✅ Evaluation suite (metrics, stress tests, arena, plots)
8. ✅ FastAPI serving with health checks
9. ✅ Comprehensive tests (≥90% coverage target)
10. ✅ Docker + Cloud Build CI/CD
11. ✅ Terraform infrastructure (all GCP resources)
12. ✅ Defaults & seeds (YAML configs, seed=42)

## File Count

- **Python files**: 60+
- **Config files**: 10+
- **Test files**: 13
- **Terraform files**: 9
- **Docker files**: 2
- **Documentation**: 3 (README, DEPLOYMENT, PROJECT_SUMMARY)

## Key Features

- 🔒 **Production-ready**: Kill switch, health checks, safety shield
- 🧪 **Test-first**: 46 tests, all passing
- 🏗️ **IaC**: Complete Terraform setup
- 🚀 **CI/CD**: Automated build, test, scan, deploy
- 📊 **Monitoring**: Alerts for errors and safety violations
- 🔄 **Reproducible**: Fixed seeds, deterministic tests
- 📦 **Containerized**: Docker images for app and training
- ☁️ **GCP-native**: Cloud Run, GCS, Artifact Registry, Secret Manager

## SLOs

- **Availability**: 99.5% (Cloud Run managed)
- **Latency**: p95 < 500ms, p99 < 1s
- **Error rate**: < 1% (5xx errors)
- **Safety violations**: < 0.1% of requests

## Next Steps

1. Replace synthetic data with real customer data
2. Train models on full dataset
3. Run A/B tests against baseline
4. Set up monitoring dashboards
5. Configure production alerts
6. Enable authentication
7. Run load tests
8. Document runbooks

## License

MIT (or your preferred license)

## Contact

For questions or issues, please open a GitHub issue or contact the team.

---

**Built with ❤️ using PyTorch, Transformers, FastAPI, and GCP**

