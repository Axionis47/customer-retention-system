#!/bin/bash
# Submit all training jobs to GCP Vertex AI
# Usage: ./ops/scripts/submit_training_jobs.sh [PROJECT_ID] [REGION]

set -e

PROJECT_ID=${1:-$GCP_PROJECT_ID}
REGION=${2:-us-central1}
EXPERIMENT=exp_001_mvp
BUCKET_NAME="${PROJECT_ID}-churn-models"
DATA_BUCKET="${PROJECT_ID}-churn-data"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID not set. Usage: $0 PROJECT_ID [REGION]"
    exit 1
fi

echo "=================================================="
echo "Submitting Training Jobs to GCP Vertex AI"
echo "=================================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Experiment: $EXPERIMENT"
echo "Model Bucket: gs://$BUCKET_NAME"
echo "Data Bucket: gs://$DATA_BUCKET"
echo "=================================================="

# Build and push trainer image
echo ""
echo "Step 1: Building and pushing trainer image..."
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/churn-saver-repo/trainer:${EXPERIMENT}"

gcloud builds submit \
    --config=ops/cloudbuild_trainer.yaml \
    --substitutions=_IMAGE_URI=${IMAGE_URI} \
    --project=${PROJECT_ID}

echo "✓ Trainer image built and pushed: ${IMAGE_URI}"

# Upload data to GCS if not already there
echo ""
echo "Step 2: Uploading processed data to GCS..."
gsutil -m rsync -r data/processed/ gs://${DATA_BUCKET}/processed/ || echo "Data already uploaded"
echo "✓ Data uploaded to gs://${DATA_BUCKET}/processed/"

# Upload experiment config
echo ""
echo "Step 3: Uploading experiment config..."
gsutil cp ops/configs/experiment_${EXPERIMENT}.yaml gs://${BUCKET_NAME}/configs/
echo "✓ Config uploaded"

# Job 1: Train Risk Model (CPU)
echo ""
echo "=================================================="
echo "Job 1: Training Risk Model (Churn Prediction)"
echo "=================================================="
JOB_NAME="risk-model-${EXPERIMENT}-$(date +%Y%m%d-%H%M%S)"

gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=${IMAGE_URI} \
    --args="python,models/risk_accept/train_churn.py,--config,/gcs/${BUCKET_NAME}/configs/experiment_${EXPERIMENT}.yaml,--train-data,/gcs/${DATA_BUCKET}/processed/telco/telco_train.parquet,--valid-data,/gcs/${DATA_BUCKET}/processed/telco/telco_valid.parquet,--test-data,/gcs/${DATA_BUCKET}/processed/telco/telco_test.parquet,--output,/gcs/${BUCKET_NAME}/artifacts/${EXPERIMENT}_risk_model.pkl" \
    --project=${PROJECT_ID}

echo "✓ Risk model training job submitted: ${JOB_NAME}"

# Job 2: Train Acceptance Model (CPU)
echo ""
echo "=================================================="
echo "Job 2: Training Acceptance Model"
echo "=================================================="
JOB_NAME="accept-model-${EXPERIMENT}-$(date +%Y%m%d-%H%M%S)"

gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=${IMAGE_URI} \
    --args="python,models/risk_accept/train_accept.py,--config,/gcs/${BUCKET_NAME}/configs/experiment_${EXPERIMENT}.yaml,--train-data,/gcs/${DATA_BUCKET}/processed/bank_marketing/bank_train.parquet,--valid-data,/gcs/${DATA_BUCKET}/processed/bank_marketing/bank_valid.parquet,--test-data,/gcs/${DATA_BUCKET}/processed/bank_marketing/bank_test.parquet,--output,/gcs/${BUCKET_NAME}/artifacts/${EXPERIMENT}_accept_model.pkl" \
    --project=${PROJECT_ID}

echo "✓ Acceptance model training job submitted: ${JOB_NAME}"

# Job 3: Train SFT Model (GPU - L4)
echo ""
echo "=================================================="
echo "Job 3: Training SFT Model (GPU)"
echo "=================================================="
JOB_NAME="sft-model-${EXPERIMENT}-$(date +%Y%m%d-%H%M%S)"

gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec=machine-type=g2-standard-4,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=${IMAGE_URI} \
    --args="python,rlhf/sft_train.py,--config,/gcs/${BUCKET_NAME}/configs/experiment_${EXPERIMENT}.yaml,--train-data,/gcs/${DATA_BUCKET}/processed/oasst1/sft_train.jsonl,--valid-data,/gcs/${DATA_BUCKET}/processed/oasst1/sft_valid.jsonl,--output,/gcs/${BUCKET_NAME}/checkpoints/${EXPERIMENT}_sft" \
    --project=${PROJECT_ID}

echo "✓ SFT training job submitted: ${JOB_NAME}"

# Job 4: Train Reward Model (GPU - L4)
echo ""
echo "=================================================="
echo "Job 4: Training Reward Model (GPU)"
echo "=================================================="
JOB_NAME="rm-model-${EXPERIMENT}-$(date +%Y%m%d-%H%M%S)"

gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec=machine-type=g2-standard-4,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=${IMAGE_URI} \
    --args="python,rlhf/rm_train.py,--config,/gcs/${BUCKET_NAME}/configs/experiment_${EXPERIMENT}.yaml,--train-data,/gcs/${DATA_BUCKET}/processed/preferences/pairs.jsonl,--valid-data,/gcs/${DATA_BUCKET}/processed/preferences/pairs_valid.jsonl,--output,/gcs/${BUCKET_NAME}/checkpoints/${EXPERIMENT}_rm" \
    --project=${PROJECT_ID}

echo "✓ Reward model training job submitted: ${JOB_NAME}"

# Wait for SFT and RM to complete before starting PPO Text
echo ""
echo "=================================================="
echo "Waiting for SFT and RM models to complete..."
echo "=================================================="
echo "This ensures PPO Text has the required dependencies"

# Function to wait for a job to complete
wait_for_job() {
    local job_pattern=$1
    local max_wait_seconds=14400  # 4 hours
    local elapsed=0
    local check_interval=60  # Check every minute

    echo "Waiting for job matching pattern: ${job_pattern}"

    while [ $elapsed -lt $max_wait_seconds ]; do
        # Get job state
        state=$(gcloud ai custom-jobs list \
            --region=${REGION} \
            --project=${PROJECT_ID} \
            --filter="displayName:${job_pattern}" \
            --format="value(state)" \
            --sort-by=~createTime \
            --limit=1)

        if [ "$state" = "JOB_STATE_SUCCEEDED" ]; then
            echo "✓ Job ${job_pattern} completed successfully"
            return 0
        elif [ "$state" = "JOB_STATE_FAILED" ]; then
            echo "✗ Job ${job_pattern} failed!"
            return 1
        elif [ "$state" = "JOB_STATE_CANCELLED" ]; then
            echo "✗ Job ${job_pattern} was cancelled"
            return 1
        fi

        echo "  Job state: ${state}, waiting... (${elapsed}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    echo "✗ Timeout waiting for job ${job_pattern}"
    return 1
}

# Wait for SFT model
wait_for_job "sft-model-${EXPERIMENT}" || {
    echo "✗ SFT training failed, skipping PPO Text"
    exit 1
}

# Wait for RM model
wait_for_job "rm-model-${EXPERIMENT}" || {
    echo "✗ RM training failed, skipping PPO Text"
    exit 1
}

# Job 5: Train PPO Text (GPU - L4) - Depends on SFT and RM
echo ""
echo "=================================================="
echo "Job 5: Training PPO Text Model (GPU)"
echo "=================================================="
echo "✓ Dependencies met: SFT and RM models are ready"
JOB_NAME="ppo-text-${EXPERIMENT}-$(date +%Y%m%d-%H%M%S)"

gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec=machine-type=g2-standard-4,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=${IMAGE_URI} \
    --args="python,rlhf/ppo_text.py,--config,/gcs/${BUCKET_NAME}/configs/experiment_${EXPERIMENT}.yaml,--sft-path,/gcs/${BUCKET_NAME}/checkpoints/${EXPERIMENT}_sft,--rm-path,/gcs/${BUCKET_NAME}/checkpoints/${EXPERIMENT}_rm,--output,/gcs/${BUCKET_NAME}/checkpoints/${EXPERIMENT}_ppo_text" \
    --project=${PROJECT_ID}

echo "✓ PPO text training job submitted: ${JOB_NAME}"

# Wait for Risk and Accept models before starting PPO Decision
echo ""
echo "=================================================="
echo "Waiting for Risk and Accept models to complete..."
echo "=================================================="

wait_for_job "risk-model-${EXPERIMENT}" || {
    echo "✗ Risk model training failed, skipping PPO Decision"
    exit 1
}

wait_for_job "accept-model-${EXPERIMENT}" || {
    echo "✗ Accept model training failed, skipping PPO Decision"
    exit 1
}

# Job 6: Train PPO Decision (CPU) - Depends on Risk and Accept models
echo ""
echo "=================================================="
echo "Job 6: Training PPO Decision Policy (CPU)"
echo "=================================================="
echo "✓ Dependencies met: Risk and Accept models are ready"
JOB_NAME="ppo-decision-${EXPERIMENT}-$(date +%Y%m%d-%H%M%S)"

gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=${IMAGE_URI} \
    --args="python,agents/ppo_policy.py,--config,/gcs/${BUCKET_NAME}/configs/experiment_${EXPERIMENT}.yaml,--risk-model,/gcs/${BUCKET_NAME}/artifacts/${EXPERIMENT}_risk_model.pkl,--accept-model,/gcs/${BUCKET_NAME}/artifacts/${EXPERIMENT}_accept_model.pkl,--output,/gcs/${BUCKET_NAME}/checkpoints/${EXPERIMENT}_ppo_decision" \
    --project=${PROJECT_ID}

echo "✓ PPO decision training job submitted: ${JOB_NAME}"

echo ""
echo "=================================================="
echo "All Training Jobs Submitted!"
echo "=================================================="
echo ""
echo "Monitor jobs at:"
echo "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "Check logs with:"
echo "gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID}"
echo ""
echo "Estimated completion time: 8-10 hours"
echo "Estimated cost: ~\$90"
echo "=================================================="

