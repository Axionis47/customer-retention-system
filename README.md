# Churn-Saver RLHF+PPO

Production-grade churn retention system combining:
- **PPO decision policy** for optimal contact timing, offer selection, and budget management
- **RLHF message generation** (SFT → Reward Model → PPO) for personalized, policy-compliant retention messages
- **Lagrangian constraints** for budget, cooldown, and fatigue management
- **GCP-native deployment** with Cloud Run, Artifact Registry, Secret Manager, and Terraform IaC
- **Comprehensive testing** with 90%+ coverage, CI/CD via Cloud Build

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Customer Event                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Risk/Accept Models                            │
│  (XGBoost churn risk, offer acceptance probability)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PPO Decision Policy                             │
│  Action: {contact?, offer_idx, delay} with Lagrangian            │
│  constraints (budget, cooldown, fatigue)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │   Contact?      │
                    └────────┬────────┘
                             │ Yes
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RLHF Message Policy                             │
│  SFT → Reward Model → PPO-text with adaptive KL                 │
│  + Safety Shield (banned phrases, length, toxicity)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Retention Message                             │
│  {decision, offer, message, safety_flags}                        │
└─────────────────────────────────────────────────────────────────┘
```

## Quickstart (Local)

### Prerequisites
- Python 3.11
- Docker (for E2E tests and deployment)
- GCP account with billing enabled (for deployment)

### Setup
```bash
make setup
```

### Run Tests
```bash
make test          # All tests with coverage
make test-unit     # Unit tests only
make lint          # Ruff + mypy
```

### Serve Locally
```bash
make serve
# API available at http://localhost:8080
# Endpoints: /healthz, /readyz, /retain
```

### Test API
```bash
curl http://localhost:8080/healthz

curl -X POST http://localhost:8080/retain \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C12345",
    "churn_risk": 0.75,
    "tenure_months": 24,
    "monthly_spend": 89.99,
    "contacts_last_7d": 0,
    "days_since_last_contact": 30
  }'
```

## Data Preparation

### Local Development
```bash
# Generate tiny demo datasets
python ops/scripts/prepare_data_local.py

# Upload to GCS (requires GCP_PROJECT_ID, GCS_DATA_BUCKET env vars)
python ops/scripts/upload_to_gcs.py
```

### Expected CSV Schema

**Churn training data** (`data/churn_train.csv`):
- `customer_id`, `tenure_months`, `monthly_spend`, `support_tickets`, `contract_type`, `churned` (0/1)

**Acceptance training data** (`data/accept_train.csv`):
- `customer_id`, `offer_pct`, `churn_risk`, `tenure_months`, `accepted` (0/1)

**RLHF pairs** (`data/rlhf_pairs.jsonl`):
- `{"prompt": "...", "chosen": "...", "rejected": "..."}`

## Training Pipeline

### 1. Risk & Acceptance Models
```bash
make train-risk
# Outputs: models/risk_accept/artifacts/{churn_model.pkl, accept_model.pkl, calibrator.pkl}
```

### 2. PPO Decision Policy
```bash
make train-ppo
# Trains PPO agent in retention environment with Lagrangian constraints
# Outputs: checkpoints/ppo_policy_*.pth
```

### 3. RLHF Message Pipeline
```bash
make train-sft       # Supervised fine-tuning
make train-rm        # Reward model (Bradley-Terry)
make train-ppo-text  # PPO with adaptive KL
# Outputs: checkpoints/sft_model/, checkpoints/rm_model/, checkpoints/ppo_text_model/
```

## GCP Deployment

### Environment Variables

Create `.env` file (or set in shell):
```bash
# Required
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
export GCS_DATA_BUCKET="your-project-churn-data"
export GCS_MODEL_BUCKET="your-project-churn-models"
export GCS_LOG_BUCKET="your-project-churn-logs"
export AR_REPO="churn-saver-repo"
export SERVICE_NAME="churn-retain-api"

# Optional
export SECRET_RLHF_TOKEN="projects/${GCP_PROJECT_ID}/secrets/rlhf-token/versions/latest"
export FORCE_BASELINE="false"  # Kill switch: true = use propensity baseline only
```

### Terraform Infrastructure

```bash
cd ops/terraform

# Initialize
terraform init

# Plan (review changes)
terraform plan \
  -var="project_id=${GCP_PROJECT_ID}" \
  -var="region=${GCP_REGION}" \
  -var="data_bucket=${GCS_DATA_BUCKET}" \
  -var="model_bucket=${GCS_MODEL_BUCKET}" \
  -var="log_bucket=${GCS_LOG_BUCKET}"

# Apply
terraform apply \
  -var="project_id=${GCP_PROJECT_ID}" \
  -var="region=${GCP_REGION}" \
  -var="data_bucket=${GCS_DATA_BUCKET}" \
  -var="model_bucket=${GCS_MODEL_BUCKET}" \
  -var="log_bucket=${GCS_LOG_BUCKET}"

# Outputs: service_url, bucket_names, service_account_emails
```

### Create Secrets
```bash
# Example: store any API keys
echo -n "your-secret-token" | gcloud secrets create rlhf-token \
  --data-file=- \
  --replication-policy="automatic" \
  --project=${GCP_PROJECT_ID}
```

### Deploy via Cloud Build
```bash
make deploy
# Triggers Cloud Build: lint → test → build → scan → deploy to Cloud Run (dev)
```

### Manual Cloud Run Deploy (alternative)
```bash
gcloud run deploy ${SERVICE_NAME} \
  --image=${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/app:latest \
  --region=${GCP_REGION} \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="GCS_MODEL_BUCKET=${GCS_MODEL_BUCKET},FORCE_BASELINE=false"
```

## Configuration

All hyperparameters in `ops/configs/*.yaml`:
- `env.yaml`: Retention environment (episode length, constraints, reward weights)
- `ppo.yaml`: PPO decision policy (learning rate, GAE lambda, clip epsilon)
- `sft.yaml`: Supervised fine-tuning (LoRA rank, learning rate, epochs)
- `rm.yaml`: Reward model (Bradley-Terry margin, batch size)
- `ppo_text.yaml`: PPO text generation (KL target, adaptive beta, rollout length)
- `serve.yaml`: API serving (timeout, quantization, fallback thresholds)

## Testing

### Coverage Target: ≥90%
```bash
make test
# Runs unit + integration + contract tests with coverage report
```

### E2E Docker Smoke Test
```bash
make docker-build
make e2e
# Builds images, starts container, hits /healthz and /retain
```

### Test Categories
- **Unit**: Reward math, constraints, Lagrangian, BT loss, KL adaptation, safety rules
- **Integration**: Environment rollouts, PPO training, API endpoints, GCS loading
- **Contract**: Golden policy outputs (deterministic seed)
- **E2E**: Docker smoke tests

## Security & Compliance

- **Secrets**: All sensitive data in Secret Manager (never plaintext)
- **IAM**: Least-privilege service accounts (CI builder, app runtime)
- **Image Scanning**: Trivy vulnerability scan in CI/CD
- **Kill Switch**: `FORCE_BASELINE=true` disables ML models, falls back to simple propensity threshold
- **Safety Shield**: Blocks banned phrases, excessive length, quiet-hour violations

## Monitoring & SLOs

Terraform provisions:
- **Log-based metrics**: 5xx rate, p95 latency, safety violations
- **Alerts**: Email/Slack on SLO breach
- **Dashboards**: Cloud Monitoring for request volume, error rate, model scores

**Target SLOs**:
- Availability: 99.5%
- p95 latency: <500ms
- Safety violation rate: <0.1%

## Cost Optimization

- **Cloud Run**: Min instances=0 (scale to zero), max=10
- **Model quantization**: 8-bit inference (bitsandbytes)
- **GCS lifecycle**: Archive logs >90 days, delete >365 days
- **Spot instances**: Use for training jobs (not implemented in v0.1)

### Estimated Monthly Cost (dev)
- Cloud Run: ~$5-20 (low traffic)
- GCS: ~$1-5 (small datasets)
- Artifact Registry: ~$0.10/GB
- Secret Manager: ~$0.06/secret/month
- **Total**: ~$10-30/month for dev environment

## Teardown

```bash
cd ops/terraform
terraform destroy \
  -var="project_id=${GCP_PROJECT_ID}" \
  -var="region=${GCP_REGION}"

# Manually delete:
# - Cloud Build history
# - Container images in Artifact Registry (if desired)
```

## Development Workflow

1. **Feature branch**: `git checkout -b feature/my-feature`
2. **Make changes**: Edit code, add tests
3. **Local validation**: `make lint && make test`
4. **Commit**: Pre-commit hooks run automatically
5. **Push**: `git push origin feature/my-feature`
6. **PR**: Cloud Build runs full CI pipeline
7. **Merge**: Auto-deploy to dev environment

## Troubleshooting

### Tests fail with "No module named 'env'"
```bash
make setup  # Reinstall in editable mode
```

### Docker build fails
```bash
# Check .dockerignore excludes .venv
# Ensure ops/docker/Dockerfile.app uses correct base image
```

### Cloud Run deploy fails
```bash
# Check service account has storage.objectViewer on model bucket
gcloud projects get-iam-policy ${GCP_PROJECT_ID}
```

### API returns 503
```bash
# Check logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}" --limit 50

# Verify models loaded
curl https://your-service-url/readyz
```

## License

MIT

## Contact

For questions or issues, open a GitHub issue or contact team@example.com.

