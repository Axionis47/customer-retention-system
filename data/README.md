# Data Directory

Place your training data CSVs here before uploading to GCS.

## Expected Files

### Churn Training Data
**File**: `churn_train.csv`

**Schema**:
- `customer_id` (str): Unique customer identifier
- `tenure_months` (int): Months as customer
- `monthly_spend` (float): Average monthly spend
- `support_tickets` (int): Number of support tickets in last 90 days
- `contract_type` (str): "month-to-month", "one-year", "two-year"
- `churned` (int): 0 or 1

### Acceptance Training Data
**File**: `accept_train.csv`

**Schema**:
- `customer_id` (str): Unique customer identifier
- `offer_pct` (float): Discount percentage offered (0-100)
- `churn_risk` (float): Predicted churn probability (0-1)
- `tenure_months` (int): Months as customer
- `monthly_spend` (float): Average monthly spend
- `accepted` (int): 0 or 1

### RLHF Preference Pairs
**File**: `rlhf_pairs.jsonl`

**Schema** (one JSON object per line):
```json
{
  "prompt": "Customer context and offer details",
  "chosen": "High-quality retention message",
  "rejected": "Low-quality retention message"
}
```

## Data Preparation

### Option 1: Use Demo Data Generator
```bash
python ops/scripts/prepare_data_local.py
```

### Option 2: Provide Your Own
Place your CSVs in this directory matching the schemas above.

## Upload to GCS
```bash
# Set environment variables first
export GCP_PROJECT_ID="your-project"
export GCS_DATA_BUCKET="your-bucket-name"

# Upload
python ops/scripts/upload_to_gcs.py
```

## Notes
- CSV files in this directory are gitignored (except test fixtures)
- Production training should read directly from GCS
- Ensure data is properly anonymized/compliant before upload

