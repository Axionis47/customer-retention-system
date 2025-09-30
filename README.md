# Churn-Saver RLHF+PPO

A churn retention system that uses machine learning to decide when to contact customers and what offers to give them.

What it does:
- **PPO decision policy** - Decides when to contact, which offer to give, and manages budget
- **RLHF message generation** - Creates personalized messages using SFT → Reward Model → PPO
- **Lagrangian constraints** - Keeps budget, cooldown, and contact frequency under control
- **GCP deployment** - Runs on Cloud Run with Terraform for infrastructure
- **Testing** - 90%+ code coverage with automated CI/CD

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

## Quick Start (Local)

### What you need
- Python 3.11
- Docker (for tests and deployment)
- GCP account with billing (for cloud deployment)

### Setup
```bash
make setup
```

### Run Tests
```bash
make test          # All tests
make test-unit     # Only unit tests
make lint          # Check code quality
```

### Run Locally
```bash
make serve
# API runs at http://localhost:8080
# Available endpoints: /healthz, /readyz, /retain
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

## Preparing Data

### For Local Testing
```bash
# Create small demo datasets
python ops/scripts/prepare_data_local.py

# Upload to GCS (needs GCP_PROJECT_ID, GCS_DATA_BUCKET set)
python ops/scripts/upload_to_gcs.py
```

### Data Format

**Churn training data** (`data/churn_train.csv`):
- `customer_id`, `tenure_months`, `monthly_spend`, `support_tickets`, `contract_type`, `churned` (0/1)

**Acceptance training data** (`data/accept_train.csv`):
- `customer_id`, `offer_pct`, `churn_risk`, `tenure_months`, `accepted` (0/1)

**RLHF pairs** (`data/rlhf_pairs.jsonl`):
- `{"prompt": "...", "chosen": "...", "rejected": "..."}`

## Training Models

### 1. Risk & Acceptance Models
```bash
make train-risk
# Creates: models/risk_accept/artifacts/{churn_model.pkl, accept_model.pkl, calibrator.pkl}
```

### 2. PPO Decision Policy
```bash
make train-ppo
# Trains PPO agent with budget and contact constraints
# Creates: checkpoints/ppo_policy_*.pth
```

### 3. RLHF Message Pipeline
```bash
make train-sft       # Fine-tune base model
make train-rm        # Train reward model
make train-ppo-text  # Train PPO for text
# Creates: checkpoints/sft_model/, checkpoints/rm_model/, checkpoints/ppo_text_model/
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

All settings are in `ops/configs/*.yaml`:
- `env.yaml`: Environment settings (episode length, budget limits, reward weights)
- `ppo.yaml`: PPO settings (learning rate, GAE lambda, clip epsilon)
- `sft.yaml`: Fine-tuning settings (LoRA rank, learning rate, epochs)
- `rm.yaml`: Reward model settings (Bradley-Terry margin, batch size)
- `ppo_text.yaml`: Text generation settings (KL target, beta, rollout length)
- `serve.yaml`: API settings (timeout, quantization, fallback thresholds)

## Testing

### Target: 90% code coverage
```bash
make test
# Runs all tests and shows coverage
```

### Docker Tests
```bash
make docker-build
make e2e
# Builds images, starts container, tests /healthz and /retain
```

### Test Types
- **Unit**: Reward calculations, constraints, Lagrangian, BT loss, KL adaptation, safety rules
- **Integration**: Environment runs, PPO training, API endpoints, GCS loading
- **Contract**: Fixed outputs with same seed
- **E2E**: Docker smoke tests

## Security

- **Secrets**: All passwords and tokens stored in Secret Manager (not in code)
- **IAM**: Service accounts with minimum required permissions
- **Image Scanning**: Trivy checks for vulnerabilities in CI/CD
- **Kill Switch**: Set `FORCE_BASELINE=true` to disable ML models and use simple rules
- **Safety Shield**: Blocks bad phrases, too long messages, and messages during quiet hours

## Monitoring

Terraform sets up:
- **Metrics**: 5xx errors, latency, safety violations
- **Alerts**: Email/Slack when things go wrong
- **Dashboards**: Cloud Monitoring for requests, errors, model performance

**Targets**:
- Availability: 99.5%
- p95 latency: under 500ms
- Safety violations: under 0.1%

## Cost

- **Cloud Run**: Scales to zero when not used, max 10 instances
- **Model quantization**: Uses 8-bit to save memory
- **GCS lifecycle**: Archives old logs after 90 days, deletes after 365 days
- **Spot instances**: Can use for training (not added yet)

### Monthly Cost (dev environment)
- Cloud Run: $5-20 (low traffic)
- GCS: $1-5 (small datasets)
- Artifact Registry: $0.10/GB
- Secret Manager: $0.06/secret/month
- **Total**: Around $10-30/month for dev

## Cleanup

```bash
cd ops/terraform
terraform destroy \
  -var="project_id=${GCP_PROJECT_ID}" \
  -var="region=${GCP_REGION}"

# Delete manually:
# - Cloud Build history
# - Container images in Artifact Registry (if you want)
```

## Development Process

1. **Create branch**: `git checkout -b feature/my-feature`
2. **Make changes**: Edit code, add tests
3. **Check locally**: `make lint && make test`
4. **Commit**: Pre-commit hooks run automatically
5. **Push**: `git push origin feature/my-feature`
6. **PR**: Cloud Build runs all checks
7. **Merge**: Deploys to dev automatically

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

