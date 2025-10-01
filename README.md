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

## Data & Serving Quickstart

### What you need
- Python 3.11
- Kaggle account and API credentials
- Docker (optional, for tests and deployment)
- GCP account with billing (optional, for cloud deployment)

### 1. Setup Kaggle Credentials

```bash
# Get your Kaggle API token
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/kaggle.json

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Install Dependencies

```bash
make setup
```

### 3. Download and Process Data

```bash
# Download and process all datasets
make data.telco    # IBM Telco Customer Churn from Kaggle
make data.bank     # UCI Bank Marketing
make data.sft      # OASST1 for SFT training
make data.prefs    # SHP-2 + HH-RLHF for reward model

# Or run all at once
make data.all

# View catalog
make data.catalog
```

This creates:
- `data/processed/telco/telco.parquet` (+ train/valid/test splits)
- `data/processed/bank_marketing/bank.parquet` (+ splits)
- `data/processed/oasst1/sft_train.jsonl` and `sft_valid.jsonl`
- `data/processed/preferences/pairs.jsonl` and `pairs_valid.jsonl`
- `data/catalog.yaml` with checksums and metadata

### 4. Run Tests

```bash
make test          # All tests
make test-unit     # Only unit tests
make lint          # Check code quality
```

### 5. Start the API

```bash
make serve
# API runs at http://localhost:8080
```

### 6. Test the /retain Endpoint

```bash
# Health check
curl http://localhost:8080/healthz

# Readiness check
curl http://localhost:8080/readyz

# Get retention decision
curl -X POST http://localhost:8080/retain \
  -H "Content-Type: application/json" \
  -d '{
    "customer_facts": {
      "tenure": 17,
      "plan": "Pro",
      "churn_risk": 0.65,
      "name": "Sam"
    },
    "policy_overrides": {
      "force_baseline": false
    },
    "debug": false
  }'
```

Response:
```json
{
  "decision": {
    "contact": true,
    "offer_level": 2,
    "followup_days": 7
  },
  "scores": {
    "p_churn": 0.65,
    "p_accept": [0.1, 0.2, 0.3, 0.4]
  },
  "message": "Hi Sam — thanks for being with us...",
  "safety": {
    "violations": 0,
    "applied_disclaimers": ["Offer valid until end of month"]
  }
}
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

## Datasets

The system uses 4 real-world datasets:

### 1. IBM Telco Customer Churn (Kaggle)
- **Purpose**: Train churn risk model `p_churn(x)`
- **Source**: Kaggle dataset `blastchar/telco-customer-churn`
- **Size**: ~7,000 customers
- **Features**: tenure, monthly charges, contract type, payment method
- **Target**: Churn (Yes/No)
- **Output**: `data/processed/telco/telco.parquet` with 80/10/10 splits

### 2. UCI Bank Marketing
- **Purpose**: Train offer acceptance model `p_accept(x, offer_level)`
- **Source**: UCI ML Repository
- **Size**: ~41,000 contacts
- **Features**: age, job, education, campaign history
- **Target**: Accepted offer (yes/no)
- **Proxy**: `offer_level` (0-3) computed from campaign intensity
- **Output**: `data/processed/bank_marketing/bank.parquet` with splits

### 3. OpenAssistant (OASST1)
- **Purpose**: Supervised fine-tuning (SFT) for message generation
- **Source**: HuggingFace `OpenAssistant/oasst1`
- **Size**: Capped at 60k train + 2k valid pairs
- **Format**: `{prompt, response}` JSONL
- **Output**: `data/processed/oasst1/sft_train.jsonl`

### 4. SHP-2 + HH-RLHF
- **Purpose**: Preference pairs for reward model training
- **Sources**:
  - HuggingFace `stanfordnlp/SHP-2` (~60k pairs)
  - HuggingFace `Anthropic/hh-rlhf` (~40k pairs)
- **Size**: Capped at 100k total pairs
- **Format**: `{prompt, chosen, rejected, source}` JSONL
- **Output**: `data/processed/preferences/pairs.jsonl`

All datasets are automatically downloaded, processed, and validated by the data pipeline.

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

