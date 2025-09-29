# Quick Start Guide

Get the Churn-Saver RLHF+PPO system running in 5 minutes.

## Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- GCP account (optional, for cloud deployment)

## Local Development

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd churn-saver-rlhf-ppo

# Create virtual environment and install dependencies
make setup

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run Tests

```bash
# Run all tests
make test

# Or run specific test suites
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/contract/ -v
```

Expected output: **46 tests passed** ✓

### 3. Generate Synthetic Data

```bash
# Generate test data
make prepare-data

# This creates:
# - data/churn_train.csv
# - data/accept_train.csv
# - data/rlhf_pairs.jsonl
```

### 4. Train Models (Optional)

```bash
# Train risk models
make train-risk
make train-accept

# Train PPO policy
make train-ppo

# Train RLHF models (requires GPU for best performance)
make train-sft
make train-rm
make train-ppo-text
```

**Note**: Training is optional. The service will use baseline policies if models aren't available.

### 5. Run the Service Locally

```bash
# Start the FastAPI server
make serve

# Or manually:
uvicorn serve.app:app --reload --port 8080
```

The service will be available at `http://localhost:8080`

### 6. Test the API

```bash
# Health check
curl http://localhost:8080/healthz

# Get retention decision
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

Expected response:
```json
{
  "customer_id": "C12345",
  "decision": {
    "contact": true,
    "offer_idx": 2,
    "delay_days": 0,
    "message": "Thank you for being with us! We'd like to offer you 10% off..."
  },
  "metadata": {
    "policy_type": "baseline",
    "timestamp": "2025-09-29T12:00:00Z"
  }
}
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build the app image
docker build -f ops/docker/Dockerfile.app -t churn-saver-app .

# Run the container
docker run -p 8080:8080 \
  -e FORCE_BASELINE=true \
  churn-saver-app

# Test
curl http://localhost:8080/healthz
```

### Run E2E Tests

```bash
# Run Docker smoke tests
pytest tests/e2e/ -v
```

## GCP Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete GCP deployment instructions.

### Quick GCP Deploy

```bash
# 1. Set up infrastructure
cd ops/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your GCP project details
terraform init
terraform apply

# 2. Deploy via Cloud Build
cd ../..
gcloud builds submit --config ops/cloudbuild.yaml
```

## Environment Variables

Key environment variables for configuration:

```bash
# Force baseline policy (kill switch)
FORCE_BASELINE=false

# Model paths (local or GCS)
PPO_POLICY_PATH=checkpoints/ppo_policy.pth
RLHF_MODEL_PATH=checkpoints/ppo_text_model

# GCS integration
GCS_MODEL_BUCKET=gs://your-project-churn-models
GCS_DATA_BUCKET=gs://your-project-churn-data

# Serving config
QUANTIZE=true
TIMEOUT_SECONDS=5
```

## Common Commands

```bash
# Development
make setup          # Install dependencies
make test           # Run all tests
make lint           # Run linters (ruff, mypy)
make format         # Format code (black, ruff)

# Data & Training
make prepare-data   # Generate synthetic data
make upload-data    # Upload to GCS
make train-risk     # Train churn risk model
make train-accept   # Train acceptance model
make train-ppo      # Train PPO policy
make train-sft      # Train SFT model
make train-rm       # Train reward model
make train-ppo-text # Train PPO text model

# Serving
make serve          # Run FastAPI locally
make docker-build   # Build Docker images
make e2e            # Run E2E tests

# Deployment
make deploy         # Deploy to GCP (requires terraform)
```

## Project Structure

```
churn-saver-rlhf-ppo/
├── env/              # Retention environment
├── agents/           # PPO policy + baselines
├── rlhf/             # RLHF pipeline
├── serve/            # FastAPI app
├── eval/             # Evaluation suite
├── models/           # Risk models
├── ops/              # Configs, Docker, Terraform
├── tests/            # Test suite
└── data/             # Training data
```

## Troubleshooting

### Tests fail with import errors
```bash
# Reinstall in editable mode
pip install -e .
```

### Service won't start
```bash
# Check logs
tail -f logs/app.log

# Use baseline mode
export FORCE_BASELINE=true
uvicorn serve.app:app --reload
```

### Models not loading
```bash
# Verify model files exist
ls -la checkpoints/
ls -la models/risk_accept/artifacts/

# Use baseline policy
export FORCE_BASELINE=true
```

### Docker build fails
```bash
# Check Docker daemon is running
docker ps

# Build with verbose output
docker build -f ops/docker/Dockerfile.app -t churn-saver-app . --progress=plain
```

## Next Steps

1. **Read the docs**: See [README.md](README.md) for architecture details
2. **Deploy to GCP**: Follow [DEPLOYMENT.md](DEPLOYMENT.md)
3. **Review the code**: Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
4. **Customize**: Modify configs in `ops/configs/`
5. **Train on real data**: Replace synthetic data with production data

## Support

For issues or questions:
- Check the documentation in this repository
- Review test files for usage examples
- Open a GitHub issue

## License

MIT (or your preferred license)

