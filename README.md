# Customer Retention System

This system helps reduce customer churn by deciding when to contact customers and what offers to give them. It uses machine learning to make smart decisions.

What it does:
- Predicts which customers might leave
- Decides when to contact them and what discount to offer
- Writes personalized messages for each customer
- Keeps track of budget and doesn't spam customers
- Runs on Google Cloud

## How it works

1. Customer data comes in
2. ML models predict if customer will leave
3. System decides: contact now or wait?
4. If contacting, picks best discount offer
5. Writes a personalized message
6. Checks message is safe and appropriate
7. Sends everything back

## What you need

- Python 3.11
- Kaggle account (for downloading datasets)
- Docker (if you want to test locally)
- Google Cloud account (if you want to deploy)

## Quick Start

### Step 1: Get Kaggle API key

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Save the file to `~/.kaggle/kaggle.json`

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Install everything

```bash
make setup
```

### Step 3: Download data

```bash
make data.all
```

This downloads 4 datasets:
- IBM Telco Customer Churn (7,000 customers)
- UCI Bank Marketing (41,000 customers)
- OASST1 conversations (16,000 pairs)
- Human preference data (100,000 pairs)

### Step 4: Run tests

```bash
make test
```

### Step 5: Start the API

```bash
make serve
```

API will run at http://localhost:8080

### Step 6: Test it

```bash
curl http://localhost:8080/healthz

curl -X POST http://localhost:8080/retain \
  -H "Content-Type: application/json" \
  -d '{
    "customer_facts": {
      "tenure": 17,
      "plan": "Pro",
      "churn_risk": 0.65,
      "name": "Sam"
    }
  }'
```

You'll get back:
```json
{
  "decision": {
    "contact": true,
    "offer_level": 2,
    "followup_days": 7
  },
  "message": "Hi Sam, thanks for being with us..."
}
```

## Training Models

If you want to train models yourself:

```bash
# Train churn prediction model
make train-risk

# Train offer acceptance model
make train-accept

# Train message generation models
make train-sft
make train-rm
make train-ppo-text

# Train decision policy
make train-ppo
```

## Deploying to Google Cloud

Set these environment variables:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
```

Then run:

```bash
cd ops/terraform
terraform init
terraform apply
```

This sets up:
- Storage buckets
- Docker registry
- Service accounts
- Cloud Run service

Deploy the API:

```bash
make deploy
```

## Project Structure

```
├── agents/          # PPO policy and baselines
├── data/            # Data processors and catalog
├── env/             # Retention environment
├── eval/            # Evaluation metrics and tests
├── models/          # Risk and acceptance models
├── rlhf/            # SFT, RM, PPO-text training
├── serve/           # FastAPI application
├── ops/             # Deployment configs and scripts
└── tests/           # All tests
```

## What's Inside

**Models trained:**
- Churn risk predictor (XGBoost)
- Offer acceptance predictor (XGBoost)
- Message generator (OPT-350m fine-tuned)
- Reward model (for rating messages)
- Decision policy (PPO with constraints)

**Datasets used:**
- IBM Telco (7,000 customers)
- UCI Bank Marketing (41,000 customers)
- OASST1 conversations (16,000 pairs)
- Human preferences (100,000 pairs)

**Tests:**
- 46 tests total
- 90%+ code coverage
- Unit, integration, contract, and e2e tests
## Common Issues

**Tests fail:**
```bash
make setup
```

**Docker build fails:**
Check that `.dockerignore` excludes `.venv` folder

**API returns errors:**
```bash
# Check logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50

# Check if models loaded
curl https://your-service-url/readyz
```

## Cost

Training all models: ~$90 (one-time)
Running API: ~$10-30/month (low traffic)

## Notes

- All random seeds set to 42 for reproducibility
- Models use 8-bit quantization to save memory
- API scales to zero when not used
- Safety checks run on all generated messages

