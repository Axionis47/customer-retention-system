# Deployment Guide

This guide walks through deploying the Churn-Saver RLHF+PPO system to Google Cloud Platform.

## Prerequisites

1. **GCP Project**: Active GCP project with billing enabled
2. **Tools Installed**:
   - `gcloud` CLI (authenticated)
   - `terraform` >= 1.5
   - `docker`
   - `make`
3. **APIs Enabled**: The Terraform will enable required APIs, but you can pre-enable:
   ```bash
   gcloud services enable \
     run.googleapis.com \
     artifactregistry.googleapis.com \
     cloudbuild.googleapis.com \
     secretmanager.googleapis.com \
     storage.googleapis.com
   ```

## Step 1: Infrastructure Setup

### 1.1 Configure Terraform Variables

```bash
cd ops/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your values:
```hcl
project_id   = "my-gcp-project"
region       = "us-central1"
environment  = "dev"

data_bucket  = "my-project-churn-data"
model_bucket = "my-project-churn-models"
log_bucket   = "my-project-churn-logs"
```

### 1.2 Initialize and Apply Terraform

```bash
terraform init
terraform plan
terraform apply
```

This creates:
- 3 GCS buckets (data, models, logs)
- Artifact Registry repository
- Service accounts (app-runtime, ci-builder)
- IAM bindings
- Secret Manager secrets
- Cloud Run service (initial deployment)
- Monitoring alerts

Save the outputs:
```bash
terraform output > ../../terraform-outputs.txt
```

## Step 2: Prepare Training Data

### 2.1 Generate Synthetic Data (for testing)

```bash
make prepare-data
```

This creates:
- `data/churn_train.csv`
- `data/accept_train.csv`
- `data/rlhf_pairs.jsonl`

### 2.2 Upload to GCS

```bash
export GCS_DATA_BUCKET=$(terraform output -raw data_bucket_name)
make upload-data
```

Or manually:
```bash
gsutil -m cp -r data/* gs://${GCS_DATA_BUCKET}/
```

## Step 3: Train Models

### 3.1 Train Risk & Acceptance Models

```bash
make train-risk
make train-accept
```

Or using Docker (recommended for consistency):
```bash
docker build -f ops/docker/Dockerfile.trainer -t churn-trainer .

docker run -v $(pwd):/workspace churn-trainer \
  python models/risk_accept/train_churn.py \
  --data-path data/churn_train.csv \
  --output-path models/risk_accept/artifacts/churn_model.pkl

docker run -v $(pwd):/workspace churn-trainer \
  python models/risk_accept/train_accept.py \
  --data-path data/accept_train.csv \
  --output-path models/risk_accept/artifacts/accept_model.pkl
```

### 3.2 Train PPO Decision Policy

```bash
make train-ppo
```

Or:
```bash
python agents/ppo_policy.py \
  --config ops/configs/ppo.yaml \
  --output checkpoints/ppo_policy.pth
```

### 3.3 Train RLHF Pipeline

```bash
# Supervised Fine-Tuning
make train-sft

# Reward Model
make train-rm

# PPO for Text
make train-ppo-text
```

### 3.4 Upload Models to GCS

```bash
export GCS_MODEL_BUCKET=$(terraform output -raw model_bucket_name)

gsutil -m cp -r models/risk_accept/artifacts/* gs://${GCS_MODEL_BUCKET}/risk_accept/
gsutil -m cp -r checkpoints/* gs://${GCS_MODEL_BUCKET}/checkpoints/
```

## Step 4: Set Secrets

```bash
# Example: HuggingFace token for RLHF models
echo -n "your-hf-token" | gcloud secrets versions add rlhf-token --data-file=-
```

## Step 5: Build and Deploy

### 5.1 Local Testing

```bash
# Build app image
make docker-build

# Run locally
docker run -p 8080:8080 \
  -e FORCE_BASELINE=true \
  churn-saver-app:latest

# Test
curl http://localhost:8080/healthz
curl -X POST http://localhost:8080/retain \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"C123","churn_risk":0.75,"tenure_months":24,"monthly_spend":89.99}'
```

### 5.2 Deploy via Cloud Build

```bash
# Trigger Cloud Build
gcloud builds submit \
  --config ops/cloudbuild.yaml \
  --substitutions=_ENV=dev,_REGION=us-central1
```

This will:
1. Run linters (ruff, mypy)
2. Run all tests with coverage
3. Build Docker images
4. Scan for vulnerabilities
5. Push to Artifact Registry
6. Deploy to Cloud Run
7. Run E2E smoke tests

### 5.3 Manual Deployment

```bash
# Build and push
export REGION=us-central1
export PROJECT_ID=$(gcloud config get-value project)
export AR_REPO=$(terraform output -raw artifact_registry_repo)

docker build -f ops/docker/Dockerfile.app -t ${AR_REPO}/app:latest .
docker push ${AR_REPO}/app:latest

# Deploy to Cloud Run
gcloud run deploy churn-retain-api-dev \
  --image=${AR_REPO}/app:latest \
  --region=${REGION} \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=10 \
  --set-env-vars="GCS_MODEL_BUCKET=gs://${GCS_MODEL_BUCKET},FORCE_BASELINE=false" \
  --service-account=$(terraform output -raw app_runtime_sa_email)
```

## Step 6: Verify Deployment

```bash
export SERVICE_URL=$(gcloud run services describe churn-retain-api-dev \
  --region=${REGION} --format='value(status.url)')

# Health check
curl ${SERVICE_URL}/healthz

# Test retention endpoint
curl -X POST ${SERVICE_URL}/retain \
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

## Step 7: Monitoring

### View Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=churn-retain-api-dev" \
  --limit 50 --format json
```

### View Metrics
```bash
# In GCP Console:
# Cloud Run > churn-retain-api-dev > Metrics
# - Request count
# - Request latency
# - Container CPU/Memory utilization
```

### Alerts
Configured alerts:
- High 5xx error rate (>5%)
- Safety violations (>10/min)

## Kill Switch

To immediately fall back to baseline policy:

```bash
gcloud run services update churn-retain-api-dev \
  --region=${REGION} \
  --set-env-vars="FORCE_BASELINE=true"
```

## Rollback

```bash
# List revisions
gcloud run revisions list --service=churn-retain-api-dev --region=${REGION}

# Rollback to previous revision
gcloud run services update-traffic churn-retain-api-dev \
  --region=${REGION} \
  --to-revisions=REVISION_NAME=100
```

## Cleanup

```bash
cd ops/terraform
terraform destroy
```

## Troubleshooting

### Service won't start
- Check logs: `gcloud logging read ...`
- Verify service account has GCS access
- Check that models exist in GCS bucket
- Try with `FORCE_BASELINE=true`

### High latency
- Increase CPU/memory in Cloud Run
- Enable request caching
- Increase min instances to avoid cold starts

### Model loading fails
- Verify GCS paths are correct
- Check service account IAM permissions
- Ensure models are in correct format

### Tests fail in CI
- Check that all dependencies are in pyproject.toml
- Verify test fixtures exist
- Check for environment-specific issues

## Production Checklist

Before going to production:

- [ ] Replace synthetic data with real data
- [ ] Train models on full dataset
- [ ] Run full evaluation suite
- [ ] Set up proper monitoring dashboards
- [ ] Configure alerting channels (email, Slack, PagerDuty)
- [ ] Enable authentication on Cloud Run
- [ ] Set up proper IAM roles (remove allUsers invoker)
- [ ] Configure VPC connector if needed
- [ ] Set up CI/CD triggers on git push
- [ ] Document runbooks for common issues
- [ ] Set up backup/restore procedures
- [ ] Configure budget alerts
- [ ] Run load tests
- [ ] Set up A/B testing framework
- [ ] Configure proper logging retention
- [ ] Set up Terraform remote state in GCS

