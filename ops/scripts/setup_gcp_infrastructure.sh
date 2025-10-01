#!/bin/bash
# Setup GCP infrastructure for training
# Usage: ./ops/scripts/setup_gcp_infrastructure.sh [PROJECT_ID] [REGION]

set -e

PROJECT_ID=${1:-plotpointe}
REGION=${2:-us-central1}

echo "=================================================="
echo "Setting up GCP Infrastructure"
echo "=================================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "=================================================="

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo ""
echo "Step 1: Enabling required APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    compute.googleapis.com \
    --project=$PROJECT_ID

echo "✓ APIs enabled"

# Create GCS buckets
echo ""
echo "Step 2: Creating GCS buckets..."

# Models bucket
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://${PROJECT_ID}-churn-models/ 2>/dev/null || echo "Models bucket already exists"
gsutil versioning set on gs://${PROJECT_ID}-churn-models/
echo "✓ Models bucket: gs://${PROJECT_ID}-churn-models/"

# Data bucket
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://${PROJECT_ID}-churn-data/ 2>/dev/null || echo "Data bucket already exists"
echo "✓ Data bucket: gs://${PROJECT_ID}-churn-data/"

# Create Artifact Registry repository
echo ""
echo "Step 3: Creating Artifact Registry repository..."
gcloud artifacts repositories create churn-saver-repo \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker images for Churn-Saver training and serving" \
    --project=$PROJECT_ID 2>/dev/null || echo "Repository already exists"

echo "✓ Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/churn-saver-repo"

# Configure Docker authentication
echo ""
echo "Step 4: Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo "✓ Docker authentication configured"

# Create service account for training jobs
echo ""
echo "Step 5: Creating service account for training..."
gcloud iam service-accounts create churn-trainer \
    --display-name="Churn Saver Training Service Account" \
    --project=$PROJECT_ID 2>/dev/null || echo "Service account already exists"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:churn-trainer@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin" \
    --condition=None

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:churn-trainer@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user" \
    --condition=None

echo "✓ Service account: churn-trainer@${PROJECT_ID}.iam.gserviceaccount.com"

echo ""
echo "=================================================="
echo "GCP Infrastructure Setup Complete!"
echo "=================================================="
echo ""
echo "Resources created:"
echo "  - GCS Bucket (models): gs://${PROJECT_ID}-churn-models/"
echo "  - GCS Bucket (data): gs://${PROJECT_ID}-churn-data/"
echo "  - Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/churn-saver-repo"
echo "  - Service Account: churn-trainer@${PROJECT_ID}.iam.gserviceaccount.com"
echo ""
echo "Next steps:"
echo "  1. Upload data: ./ops/scripts/upload_data_to_gcs.sh"
echo "  2. Submit training jobs: ./ops/scripts/submit_training_jobs.sh"
echo "=================================================="

