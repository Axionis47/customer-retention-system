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

## Technical Architecture

This project combines 3 different ML systems working together:

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Customer Input                               │
│  {tenure, plan, monthly_charges, churn_risk, name, ...}         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              SYSTEM 1: Tabular Predictors                        │
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Risk Model      │         │  Accept Model    │             │
│  │  (XGBoost)       │         │  (XGBoost)       │             │
│  │                  │         │                  │             │
│  │  Input: Customer │         │  Input: Customer │             │
│  │  features        │         │  + Offer level   │             │
│  │                  │         │                  │             │
│  │  Output:         │         │  Output:         │             │
│  │  P(churn)        │         │  P(accept|offer) │             │
│  └──────────────────┘         └──────────────────┘             │
│         │                              │                        │
└─────────┼──────────────────────────────┼────────────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           SYSTEM 2: PPO Decision Policy                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Retention Environment (Gymnasium)                      │    │
│  │                                                          │    │
│  │  State: [churn_risk, accept_probs, budget, cooldown,   │    │
│  │          fatigue, days_since_contact]                   │    │
│  │                                                          │    │
│  │  Actions: {contact: yes/no, offer: 0-3, delay: 0-7}    │    │
│  │                                                          │    │
│  │  Reward: revenue_retained - offer_cost - penalties     │    │
│  └────────────────────────────────────────────────────────┘    │
│                         │                                        │
│                         ▼                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  PPO Policy Network (Actor-Critic)                      │    │
│  │                                                          │    │
│  │  Actor: π(a|s) → action probabilities                   │    │
│  │  Critic: V(s) → state value                             │    │
│  │                                                          │    │
│  │  Training: Clipped surrogate loss + GAE                 │    │
│  └────────────────────────────────────────────────────────┘    │
│                         │                                        │
│                         ▼                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Lagrangian Constraints                                 │    │
│  │                                                          │    │
│  │  Budget: Σ offer_cost ≤ budget_limit                    │    │
│  │  Fatigue: contacts_per_customer ≤ fatigue_cap          │    │
│  │  Cooldown: min_days_between_contacts ≥ cooldown        │    │
│  │                                                          │    │
│  │  Penalty: λ₁·budget_violation + λ₂·fatigue_violation   │    │
│  └────────────────────────────────────────────────────────┘    │
│                         │                                        │
└─────────────────────────┼────────────────────────────────────────┘
                          │
                          ▼
                   Decision: Contact?
                          │
                    ┌─────┴─────┐
                    │           │
                   No          Yes
                    │           │
                    ▼           ▼
                  Wait    ┌─────────────────────────────────────┐
                          │  SYSTEM 3: RLHF Message Generator   │
                          │                                     │
                          │  ┌────────────────────────────┐    │
                          │  │  SFT Model (OPT-350m)      │    │
                          │  │  + LoRA (r=16, α=32)       │    │
                          │  │  + 8-bit quantization      │    │
                          │  │                             │    │
                          │  │  Trained on: OASST1        │    │
                          │  │  (16k conversation pairs)  │    │
                          │  └────────────┬───────────────┘    │
                          │               │                     │
                          │               ▼                     │
                          │  ┌────────────────────────────┐    │
                          │  │  Reward Model              │    │
                          │  │  (OPT-350m + LoRA)         │    │
                          │  │                             │    │
                          │  │  Trained on: SHP-2 + HH    │    │
                          │  │  (100k preference pairs)   │    │
                          │  │                             │    │
                          │  │  Loss: Bradley-Terry       │    │
                          │  └────────────┬───────────────┘    │
                          │               │                     │
                          │               ▼                     │
                          │  ┌────────────────────────────┐    │
                          │  │  PPO-Text                  │    │
                          │  │                             │    │
                          │  │  Policy: Generate message  │    │
                          │  │  Reward: RM score          │    │
                          │  │  Constraint: KL(π||π_SFT)  │    │
                          │  │              ≤ target_KL   │    │
                          │  │                             │    │
                          │  │  Adaptive β control        │    │
                          │  └────────────┬───────────────┘    │
                          │               │                     │
                          │               ▼                     │
                          │  ┌────────────────────────────┐    │
                          │  │  Safety Shield             │    │
                          │  │                             │    │
                          │  │  ✓ No banned phrases       │    │
                          │  │  ✓ Length check            │    │
                          │  │  ✓ Quiet hours check       │    │
                          │  │  ✓ Required elements       │    │
                          │  └────────────┬───────────────┘    │
                          │               │                     │
                          └───────────────┼─────────────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │  Final Output                 │
                          │                               │
                          │  {                            │
                          │    decision: {contact, offer} │
                          │    message: "Hi Sam, ..."     │
                          │    safety: {violations: 0}    │
                          │  }                            │
                          └───────────────────────────────┘
```

### System 1: Tabular Predictors (XGBoost)

**Purpose**: Predict customer behavior

**Models**:
1. **Risk Model** - Predicts probability customer will churn
   - Algorithm: XGBoost classifier
   - Features: tenure, monthly_charges, contract_type, payment_method, etc.
   - Training data: IBM Telco (7,032 customers)
   - Output: P(churn) ∈ [0, 1]
   - Calibration: Isotonic regression (ensures probabilities are accurate)
   - Exit criteria: AUC ≥ 0.78, ECE ≤ 0.05

2. **Accept Model** - Predicts probability customer accepts offer
   - Algorithm: XGBoost classifier
   - Features: customer features + offer_level (0-3)
   - Training data: UCI Bank Marketing (41,188 contacts)
   - Output: P(accept | offer_level) ∈ [0, 1]
   - Calibration: Isotonic regression
   - Exit criteria: AUC ≥ 0.70, ECE ≤ 0.05

**Why XGBoost?**
- Handles tabular data very well
- Fast training and inference
- Built-in feature importance
- Works with missing values

**Calibration**:
```
Raw XGBoost → Isotonic Regression → Calibrated Probabilities
```
This ensures when model says "70% chance", it's actually 70% in real data.

### System 2: PPO Decision Policy (Reinforcement Learning)

**Purpose**: Decide when to contact customers and what offer to give

**Environment** (`RetentionEnv`):
- **State space** (9 dimensions):
  ```
  [churn_risk, accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3,
   budget_remaining, days_since_contact, cooldown_active, fatigue_count]
  ```

- **Action space** (3 dimensions):
  ```
  contact ∈ {0, 1}           # Don't contact or contact
  offer_idx ∈ {0, 1, 2, 3}   # Which discount: 0%, 5%, 10%, 15%
  delay ∈ {0, 1, ..., 7}     # Days to wait before next decision
  ```

- **Reward function**:
  ```
  reward = revenue_retained - offer_cost - λ₁·budget_violation - λ₂·fatigue_violation

  where:
    revenue_retained = customer_value × (1 if retained else 0)
    offer_cost = customer_value × offer_percentage
    budget_violation = max(0, total_spent - budget_limit)
    fatigue_violation = max(0, contacts - fatigue_cap)
  ```

**PPO Algorithm**:
```
1. Collect rollouts using current policy π_θ
2. Compute advantages using GAE (λ=0.95):
   A_t = Σ (γλ)^k δ_{t+k}
   where δ_t = r_t + γV(s_{t+1}) - V(s_t)

3. Update policy with clipped objective:
   L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
   where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

4. Update value function:
   L^VF(θ) = E[(V_θ(s_t) - V_target)²]

5. Add entropy bonus for exploration:
   L(θ) = L^CLIP(θ) - c₁·L^VF(θ) + c₂·H(π_θ)
```

**Network Architecture**:
```
Input (state) → Shared layers (128 → 128) → Split into 4 heads:
                                              ├─ Contact head (2 actions)
                                              ├─ Offer head (4 actions)
                                              ├─ Delay head (8 actions)
                                              └─ Value head (1 value)
```

**Lagrangian Constraints**:
Instead of hard constraints, we use adaptive penalties:
```
λ_budget(t+1) = λ_budget(t) + α · (budget_used - budget_limit)
λ_fatigue(t+1) = λ_fatigue(t) + α · (contacts - fatigue_cap)
```
This allows the policy to learn to respect constraints automatically.

### System 3: RLHF Message Generator

**Purpose**: Write personalized retention messages

**Three-stage training pipeline**:

#### Stage 1: Supervised Fine-Tuning (SFT)
```
Base Model: facebook/opt-350m (350M parameters)
           ↓
    Add LoRA adapters (r=16, α=32)
           ↓
    Train on OASST1 conversations
           ↓
    SFT Model (can generate coherent messages)
```

**LoRA** (Low-Rank Adaptation):
- Instead of fine-tuning all 350M parameters, we add small adapter matrices
- Only train ~0.5M parameters (99.8% reduction!)
- Formula: `W' = W + BA` where B is r×d and A is d×r
- Saves memory and training time

**8-bit Quantization**:
- Store weights in 8-bit instead of 32-bit
- 4x memory reduction
- Minimal accuracy loss

#### Stage 2: Reward Model (RM)
```
Base Model: facebook/opt-350m
           ↓
    Add LoRA adapters
           ↓
    Replace head with reward head (outputs scalar)
           ↓
    Train on preference pairs (chosen vs rejected)
           ↓
    Reward Model (scores message quality)
```

**Training objective** (Bradley-Terry):
```
L = -E[log σ(r(x, y_chosen) - r(x, y_rejected))]

where:
  r(x, y) = reward model score for prompt x and response y
  σ = sigmoid function
```

This teaches the model: "chosen message is better than rejected message"

**Training data**:
- SHP-2: Reddit posts with upvotes (60k pairs)
- HH-RLHF: Human preferences from Anthropic (40k pairs)

#### Stage 3: PPO-Text
```
SFT Model → Generate message → Reward Model → Score
     ↑                                           │
     └───────────── Update policy ───────────────┘
```

**Objective**:
```
maximize: E[r(x, y)] - β·KL(π_θ || π_SFT)

where:
  r(x, y) = reward model score
  KL(π_θ || π_SFT) = KL divergence from SFT model
  β = adaptive coefficient
```

**Why KL constraint?**
Without it, the model might generate high-reward but nonsensical text.
KL keeps it close to the original SFT model.

**Adaptive β control**:
```
if KL > target_KL:
    β = β × 1.5  # Increase penalty
elif KL < target_KL / 2:
    β = β / 1.5  # Decrease penalty
```

**Safety Shield**:
After generation, check:
- ✓ No banned phrases (profanity, false promises)
- ✓ Length between 50-200 characters
- ✓ Not during quiet hours (10pm-8am)
- ✓ Contains required elements (customer name, offer details)

### Data Pipeline

**4 real datasets, 166K+ data points**:

1. **IBM Telco Customer Churn** (Kaggle)
   - 7,032 customers
   - Features: tenure, charges, contract, services
   - Target: Churn (Yes/No)
   - Used for: Risk model training

2. **UCI Bank Marketing**
   - 41,188 contacts
   - Features: age, job, education, campaign history
   - Target: Accepted offer (yes/no)
   - Proxy: offer_level computed from campaign intensity
   - Used for: Accept model training

3. **OASST1** (HuggingFace)
   - 16,440 conversation pairs
   - Format: {prompt, response}
   - Quality: Human-rated conversations
   - Used for: SFT training

4. **SHP-2 + HH-RLHF** (HuggingFace)
   - 100,000 preference pairs
   - Format: {prompt, chosen, rejected}
   - Sources: Reddit upvotes + human feedback
   - Used for: Reward model training

**Processing**:
```
Raw data → Download → Clean → Feature engineering → Train/valid/test split (80/10/10) → Save as Parquet/JSONL
```

All splits use fixed seed (42) for reproducibility.

### Training Infrastructure (GCP)

**Vertex AI Custom Jobs**:
```
Job 1: Risk Model      → n1-standard-4 (CPU)  → 30 min → $2
Job 2: Accept Model    → n1-standard-4 (CPU)  → 30 min → $3
Job 3: SFT Model       → g2-standard-4 (L4)   → 2 hrs  → $30
Job 4: RM Model        → g2-standard-4 (L4)   → 1 hr   → $15
Job 5: PPO-Text        → g2-standard-4 (L4)   → 2 hrs  → $30
Job 6: PPO-Decision    → n1-standard-4 (CPU)  → 1 hr   → $10
                                                Total: ~$90
```

**Docker Image**:
- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.2.0 with CUDA 12.1
- All dependencies from `pyproject.toml`

**Storage**:
- `gs://plotpointe-churn-data/` - Processed datasets
- `gs://plotpointe-churn-models/` - Trained models and configs

**Job Dependencies**:
```
Risk Model ─┐
            ├─→ PPO Decision
Accept Model┘

SFT Model ─┐
           ├─→ PPO Text
RM Model ──┘
```

Jobs wait for dependencies before starting.

### Evaluation Metrics

**Tabular Models**:
- AUC-ROC: Area under ROC curve (discrimination)
- ECE: Expected Calibration Error (calibration quality)
- Precision/Recall at different thresholds

**PPO Decision**:
- NRR: Net Revenue Retention
- ROI: Return on Investment
- Constraint violations: Budget and fatigue
- Contact rate: % of customers contacted

**RLHF**:
- Win rate: % of times PPO-text beats baseline
- KL divergence: Distance from SFT model
- Safety violations: % of unsafe messages
- Human evaluation: Quality ratings

### What Makes This Hard?

1. **Multi-objective optimization**: Maximize retention, minimize cost, respect constraints
2. **Sequential decisions**: When to contact affects future opportunities
3. **Exploration vs exploitation**: Try new strategies vs use known good ones
4. **Constraint satisfaction**: Hard limits on budget and contact frequency
5. **Text generation quality**: Must be helpful, safe, and personalized
6. **Calibration**: Probabilities must be accurate for good decisions
7. **Production deployment**: Must handle real traffic, scale, and fail gracefully

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

