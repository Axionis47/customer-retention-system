#!/bin/bash
# Complete training and evaluation pipeline
# This script runs everything from data prep to evaluation

set -e

PROJECT_ID=${1:-plotpointe}
REGION=${2:-us-central1}
EXPERIMENT=exp_001_mvp

echo "=================================================="
echo "COMPLETE TRAINING & EVALUATION PIPELINE"
echo "=================================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Experiment: $EXPERIMENT"
echo "=================================================="

# Step 1: Cancel any pending jobs from old code
echo ""
echo "Step 1: Cleaning up old jobs..."
echo "=================================================="
OLD_JOBS=$(gcloud ai custom-jobs list \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --filter="state:JOB_STATE_PENDING AND displayName:exp_001_mvp-20251001-22" \
    --format="value(name)" 2>/dev/null || true)

if [ -n "$OLD_JOBS" ]; then
    echo "Found pending jobs from old code, cancelling..."
    echo "$OLD_JOBS" | while read job; do
        gcloud ai custom-jobs cancel $job --region=${REGION} 2>/dev/null || true
        echo "  Cancelled: $job"
    done
    echo "✓ Old jobs cancelled"
else
    echo "✓ No old jobs to cancel"
fi

# Step 2: Upload updated config
echo ""
echo "Step 2: Uploading updated experiment config..."
echo "=================================================="
gsutil cp ops/configs/experiment_${EXPERIMENT}.yaml gs://${PROJECT_ID}-churn-models/configs/
echo "✓ Config uploaded"

# Step 3: Rebuild Docker image with fixes
echo ""
echo "Step 3: Rebuilding Docker trainer image with fixes..."
echo "=================================================="
echo "This will take ~15 minutes..."

gcloud builds submit \
    --config=ops/cloudbuild_trainer.yaml \
    --project=${PROJECT_ID} \
    --timeout=20m \
    .

echo "✓ Docker image rebuilt with all fixes"

# Step 4: Submit training jobs with dependencies
echo ""
echo "Step 4: Submitting training jobs with proper dependencies..."
echo "=================================================="
./ops/scripts/submit_training_jobs.sh ${PROJECT_ID} ${REGION}

echo ""
echo "=================================================="
echo "✓ ALL JOBS SUBMITTED SUCCESSFULLY"
echo "=================================================="
echo ""
echo "Jobs are now running with:"
echo "  ✅ Fixed PPO text (CLI args + RM loading)"
echo "  ✅ Fixed PPO decision (models passed to env)"
echo "  ✅ Fixed RetentionEnv (uses real models)"
echo "  ✅ Fixed job dependencies (proper wait logic)"
echo "  ✅ Model validation (catches errors early)"
echo ""
echo "Monitor progress at:"
echo "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "Expected completion: ~8 hours"
echo "Expected cost: ~\$90"
echo ""
echo "=================================================="
echo "Next: Wait for training to complete, then run:"
echo "  ./ops/scripts/evaluate_results.sh ${PROJECT_ID} ${REGION}"
echo "=================================================="

