#!/bin/bash
# Upload processed data and configs to GCS
# Usage: ./ops/scripts/upload_data_to_gcs.sh [PROJECT_ID]

set -e

PROJECT_ID=${1:-plotpointe}
DATA_BUCKET="${PROJECT_ID}-churn-data"
MODEL_BUCKET="${PROJECT_ID}-churn-models"

echo "=================================================="
echo "Uploading Data to GCS"
echo "=================================================="
echo "Project: $PROJECT_ID"
echo "Data Bucket: gs://$DATA_BUCKET"
echo "Model Bucket: gs://$MODEL_BUCKET"
echo "=================================================="

# Upload processed data
echo ""
echo "Step 1: Uploading processed datasets..."
gsutil -m rsync -r data/processed/ gs://${DATA_BUCKET}/processed/
echo "✓ Processed data uploaded to gs://${DATA_BUCKET}/processed/"

# Upload experiment config
echo ""
echo "Step 2: Uploading experiment config..."
gsutil cp ops/configs/experiment_exp_001_mvp.yaml gs://${MODEL_BUCKET}/configs/
echo "✓ Config uploaded to gs://${MODEL_BUCKET}/configs/"

# Upload locally trained models (if any)
echo ""
echo "Step 3: Uploading locally trained models (if any)..."
if [ -d "models/risk_accept/artifacts" ]; then
    gsutil -m rsync -r models/risk_accept/artifacts/ gs://${MODEL_BUCKET}/artifacts/
    echo "✓ Local models uploaded to gs://${MODEL_BUCKET}/artifacts/"
else
    echo "No local models found, skipping..."
fi

# List uploaded files
echo ""
echo "=================================================="
echo "Upload Complete!"
echo "=================================================="
echo ""
echo "Data files:"
gsutil ls -r gs://${DATA_BUCKET}/processed/ | head -20
echo ""
echo "Config files:"
gsutil ls gs://${MODEL_BUCKET}/configs/
echo ""
echo "=================================================="

