# System Architecture

This document explains the complete architecture of the Customer Retention System with visual diagrams.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [System 1: Tabular Predictors](#system-1-tabular-predictors)
3. [System 2: PPO Decision Policy](#system-2-ppo-decision-policy)
4. [System 3: RLHF Message Generator](#system-3-rlhf-message-generator)
5. [Data Pipeline](#data-pipeline)
6. [Training Infrastructure](#training-infrastructure)
7. [Deployment Architecture](#deployment-architecture)

---

## High-Level Overview

The system combines 3 ML systems that work together to make retention decisions and generate personalized messages.

```mermaid
graph TB
    Input[Customer Data<br/>tenure, charges, contract, etc.]
    
    subgraph System1[System 1: Tabular Predictors]
        Risk[Risk Model<br/>XGBoost]
        Accept[Accept Model<br/>XGBoost]
    end
    
    subgraph System2[System 2: PPO Decision Policy]
        Env[Retention Environment<br/>Gymnasium]
        PPO[PPO Policy Network<br/>Actor-Critic]
        Constraints[Lagrangian Constraints<br/>Budget + Fatigue]
    end
    
    subgraph System3[System 3: RLHF Message Generator]
        SFT[SFT Model<br/>OPT-350m + LoRA]
        RM[Reward Model<br/>OPT-350m + LoRA]
        PPOText[PPO-Text<br/>RLHF]
        Safety[Safety Shield<br/>Validation]
    end
    
    Decision{Contact?}
    Output[Final Output<br/>decision + message]
    
    Input --> Risk
    Input --> Accept
    Risk --> Env
    Accept --> Env
    Env --> PPO
    PPO --> Constraints
    Constraints --> Decision
    
    Decision -->|No| Output
    Decision -->|Yes| SFT
    SFT --> RM
    RM --> PPOText
    PPOText --> Safety
    Safety --> Output
    
    style System1 fill:#e1f5ff
    style System2 fill:#fff4e1
    style System3 fill:#f0e1ff
```

**Flow:**
1. Customer data enters the system
2. Risk and Accept models predict probabilities
3. PPO policy decides whether to contact and what offer to give
4. If contacting, RLHF generates a personalized message
5. Safety shield validates the message
6. Final decision and message returned

---

## System 1: Tabular Predictors

**Purpose**: Predict customer behavior using structured data

```mermaid
graph LR
    subgraph Input
        Features[Customer Features<br/>tenure: 12<br/>charges: 89.99<br/>contract: month-to-month<br/>payment: electronic]
    end
    
    subgraph RiskModel[Risk Model]
        XGB1[XGBoost Classifier<br/>100 trees, depth=6]
        Calib1[Isotonic Calibration]
        XGB1 --> Calib1
    end
    
    subgraph AcceptModel[Accept Model]
        XGB2[XGBoost Classifier<br/>100 trees, depth=6]
        Calib2[Isotonic Calibration]
        XGB2 --> Calib2
    end
    
    subgraph Output
        Risk[P churn = 0.68]
        Accept0[P accept|0% = 0.12]
        Accept1[P accept|5% = 0.45]
        Accept2[P accept|10% = 0.73]
        Accept3[P accept|15% = 0.89]
    end
    
    Features --> XGB1
    Features --> XGB2
    Calib1 --> Risk
    Calib2 --> Accept0
    Calib2 --> Accept1
    Calib2 --> Accept2
    Calib2 --> Accept3
    
    style RiskModel fill:#ffcccc
    style AcceptModel fill:#ccffcc
```

### Risk Model

**Algorithm**: XGBoost with isotonic calibration

**Training Data**: IBM Telco (7,032 customers)

**Features**:
- `tenure`: Months with company
- `monthly_charges`: Monthly bill amount
- `contract`: Month-to-month, One year, Two year
- `payment_method`: Electronic check, Mailed check, Bank transfer, Credit card
- `services`: Internet, Phone, Streaming, etc.

**Output**: Calibrated probability of churn P(churn) ∈ [0, 1]

**Calibration**:
```
Raw XGBoost scores → Isotonic Regression → Calibrated probabilities
```

This ensures when model says "70% chance of churn", it's actually 70% in real data.

**Exit Criteria**:
- AUC-ROC ≥ 0.78
- Expected Calibration Error (ECE) ≤ 0.05

### Accept Model

**Algorithm**: XGBoost with isotonic calibration

**Training Data**: UCI Bank Marketing (41,188 contacts)

**Features**:
- Customer features (same as risk model)
- `offer_level`: 0, 1, 2, 3 (representing 0%, 5%, 10%, 15% discount)

**Output**: Calibrated probability of accepting offer P(accept | offer_level) ∈ [0, 1]

**Exit Criteria**:
- AUC-ROC ≥ 0.70
- Expected Calibration Error (ECE) ≤ 0.05

---

## System 2: PPO Decision Policy

**Purpose**: Decide when to contact customers and what offer to give

```mermaid
graph TB
    subgraph Environment[Retention Environment]
        State[State Space 9D<br/>churn_risk, accept_probs 0-3,<br/>budget, days_since_contact,<br/>cooldown, fatigue]
        Action[Action Space 3D<br/>contact: 0/1<br/>offer: 0-3<br/>delay: 0-7]
        Reward[Reward Function<br/>revenue - cost - penalties]
    end
    
    subgraph Policy[PPO Policy Network]
        Shared[Shared Layers<br/>128 → 128 ReLU]
        Contact[Contact Head<br/>2 actions]
        Offer[Offer Head<br/>4 actions]
        Delay[Delay Head<br/>8 actions]
        Value[Value Head<br/>1 value]
        
        Shared --> Contact
        Shared --> Offer
        Shared --> Delay
        Shared --> Value
    end
    
    subgraph Training[PPO Training]
        Rollout[Collect Rollouts<br/>2048 steps]
        GAE[Compute Advantages<br/>GAE λ=0.95]
        Update[Update Policy<br/>Clipped Loss]
        
        Rollout --> GAE
        GAE --> Update
    end
    
    subgraph Constraints[Lagrangian Constraints]
        Budget[Budget Constraint<br/>Σ cost ≤ limit]
        Fatigue[Fatigue Constraint<br/>contacts ≤ cap]
        Penalty[Adaptive Penalties<br/>λ₁, λ₂]
        
        Budget --> Penalty
        Fatigue --> Penalty
    end
    
    State --> Shared
    Contact --> Action
    Offer --> Action
    Delay --> Action
    Action --> Reward
    Reward --> Rollout
    Penalty --> Reward
    
    style Environment fill:#e1f5ff
    style Policy fill:#fff4e1
    style Training fill:#f0e1ff
    style Constraints fill:#ffe1e1
```

### State Space (9 dimensions)

```python
state = [
    churn_risk,          # P(churn) from risk model
    accept_prob_0,       # P(accept | 0% offer)
    accept_prob_1,       # P(accept | 5% offer)
    accept_prob_2,       # P(accept | 10% offer)
    accept_prob_3,       # P(accept | 15% offer)
    budget_remaining,    # Normalized budget left
    days_since_contact,  # Days since last contact
    cooldown_active,     # 1 if in cooldown, 0 otherwise
    fatigue_count        # Number of contacts this period
]
```

### Action Space (3 dimensions)

```python
action = {
    'contact': 0 or 1,           # Don't contact or contact
    'offer_idx': 0, 1, 2, or 3,  # Which discount: 0%, 5%, 10%, 15%
    'delay': 0 to 7              # Days to wait before next decision
}
```

### Reward Function

```python
reward = revenue_retained - offer_cost - λ₁·budget_violation - λ₂·fatigue_violation

where:
    revenue_retained = customer_value × (1 if retained else 0)
    offer_cost = customer_value × offer_percentage
    budget_violation = max(0, total_spent - budget_limit)
    fatigue_violation = max(0, contacts - fatigue_cap)
```

### PPO Algorithm

```mermaid
graph LR
    subgraph Step1[1. Collect Rollouts]
        Roll[Use current policy π_θ<br/>to collect 2048 steps]
    end
    
    subgraph Step2[2. Compute Advantages]
        GAE[Generalized Advantage<br/>Estimation λ=0.95<br/>A_t = Σ γλ^k δ_t+k]
    end
    
    subgraph Step3[3. Update Policy]
        Clip[Clipped Surrogate Loss<br/>min r_t A_t, clip r_t A_t]
    end
    
    subgraph Step4[4. Update Value]
        VF[Value Function Loss<br/>V_θ - V_target ²]
    end
    
    subgraph Step5[5. Add Entropy]
        Ent[Entropy Bonus<br/>for exploration]
    end
    
    Roll --> GAE
    GAE --> Clip
    Clip --> VF
    VF --> Ent
    Ent --> Roll
    
    style Step1 fill:#e1f5ff
    style Step2 fill:#fff4e1
    style Step3 fill:#f0e1ff
    style Step4 fill:#ffe1e1
    style Step5 fill:#e1ffe1
```

**Mathematical Details**:

1. **Advantage Estimation** (GAE):
   ```
   A_t = Σ_{k=0}^∞ (γλ)^k δ_{t+k}
   where δ_t = r_t + γV(s_{t+1}) - V(s_t)
   ```

2. **Clipped Surrogate Loss**:
   ```
   L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
   where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
   ```

3. **Value Function Loss**:
   ```
   L^VF(θ) = E[(V_θ(s_t) - V_target)²]
   ```

4. **Total Loss**:
   ```
   L(θ) = L^CLIP(θ) - c₁·L^VF(θ) + c₂·H(π_θ)
   where H(π_θ) is entropy bonus
   ```

### Lagrangian Constraints

Instead of hard constraints, we use adaptive penalties:

```python
# Update Lagrangian multipliers
λ_budget(t+1) = λ_budget(t) + α · (budget_used - budget_limit)
λ_fatigue(t+1) = λ_fatigue(t) + α · (contacts - fatigue_cap)

# Add to reward
reward = base_reward - λ_budget · budget_violation - λ_fatigue · fatigue_violation
```

This allows the policy to learn to respect constraints automatically.

---

## System 3: RLHF Message Generator

**Purpose**: Generate personalized retention messages using Reinforcement Learning from Human Feedback

```mermaid
graph TB
    subgraph Stage1[Stage 1: Supervised Fine-Tuning]
        Base1[Base Model<br/>facebook/opt-350m<br/>350M parameters]
        LoRA1[Add LoRA Adapters<br/>r=16, α=32<br/>~0.5M trainable params]
        Train1[Train on OASST1<br/>16,440 conversations]
        SFT[SFT Model<br/>Can generate coherent text]

        Base1 --> LoRA1
        LoRA1 --> Train1
        Train1 --> SFT
    end

    subgraph Stage2[Stage 2: Reward Model]
        Base2[Base Model<br/>facebook/opt-350m]
        LoRA2[Add LoRA Adapters<br/>r=16, α=32]
        Head[Replace LM head<br/>with reward head]
        Train2[Train on Preferences<br/>100k chosen/rejected pairs]
        RM[Reward Model<br/>Scores message quality]

        Base2 --> LoRA2
        LoRA2 --> Head
        Head --> Train2
        Train2 --> RM
    end

    subgraph Stage3[Stage 3: PPO-Text]
        Generate[Generate Message<br/>using SFT model]
        Score[Score with RM]
        Compute[Compute Reward<br/>+ KL penalty]
        Update[Update Policy<br/>PPO algorithm]

        Generate --> Score
        Score --> Compute
        Compute --> Update
        Update --> Generate
    end

    subgraph Safety[Safety Shield]
        Check1[No banned phrases]
        Check2[Length 50-200 chars]
        Check3[Not quiet hours]
        Check4[Has required elements]
        Pass[Pass/Fail]

        Check1 --> Pass
        Check2 --> Pass
        Check3 --> Pass
        Check4 --> Pass
    end

    SFT --> Generate
    RM --> Score
    Update --> Safety

    style Stage1 fill:#e1f5ff
    style Stage2 fill:#fff4e1
    style Stage3 fill:#f0e1ff
    style Safety fill:#ffe1e1
```

### Stage 1: Supervised Fine-Tuning (SFT)

**Base Model**: `facebook/opt-350m` (350 million parameters)

**LoRA (Low-Rank Adaptation)**:
```
Instead of fine-tuning all 350M parameters:
W' = W + BA

where:
  W = frozen pretrained weights (350M params)
  B = trainable matrix (r × d)
  A = trainable matrix (d × r)
  r = rank = 16 (much smaller than d)

Total trainable: ~0.5M parameters (99.8% reduction!)
```

**8-bit Quantization**:
- Store weights in 8-bit instead of 32-bit
- 4x memory reduction
- Minimal accuracy loss
- Enables training on single GPU

**Training Data**: OASST1 (16,440 conversation pairs)
```json
{
  "prompt": "Write a retention message for a customer at risk of churning",
  "response": "Hi Sam, we value your 12 months with us! Here's a special 10% discount..."
}
```

**Training Objective**: Standard language modeling loss
```
L_SFT = -Σ log P(y_t | y_{<t}, x)
```

**Output**: SFT model that can generate coherent retention messages

### Stage 2: Reward Model (RM)

**Architecture**: Same as SFT but with reward head instead of language modeling head

```mermaid
graph LR
    Input[Prompt + Response]
    Transformer[OPT-350m<br/>Transformer Layers<br/>frozen]
    LoRA[LoRA Adapters<br/>trainable]
    Pool[Mean Pooling<br/>last layer]
    Head[Reward Head<br/>Linear → scalar]
    Score[Reward Score<br/>r ∈ ℝ]

    Input --> Transformer
    Transformer --> LoRA
    LoRA --> Pool
    Pool --> Head
    Head --> Score

    style Transformer fill:#e1f5ff
    style LoRA fill:#fff4e1
    style Head fill:#f0e1ff
```

**Training Data**: SHP-2 + HH-RLHF (100,000 preference pairs)
```json
{
  "prompt": "Write a retention message...",
  "chosen": "Hi Sam, we value your loyalty! Here's 10% off...",
  "rejected": "URGENT!!! DON'T LEAVE!!! 50% OFF NOW!!!"
}
```

**Training Objective**: Bradley-Terry loss
```
L_RM = -E[log σ(r(x, y_chosen) - r(x, y_rejected))]

where:
  r(x, y) = reward model score for prompt x and response y
  σ = sigmoid function
```

This teaches: "chosen message is better than rejected message"

**Output**: Reward model that scores message quality

### Stage 3: PPO-Text

**Objective**: Fine-tune SFT model to maximize reward while staying close to original

```
maximize: E[r(x, y)] - β·KL(π_θ || π_SFT)

where:
  r(x, y) = reward model score
  KL(π_θ || π_SFT) = KL divergence from SFT model
  β = adaptive coefficient
```

**Why KL constraint?**
Without it, model might generate high-reward but nonsensical text like:
```
"AMAZING CUSTOMER BEST EVER PERFECT WONDERFUL EXCELLENT FANTASTIC"
```

KL keeps it close to the original SFT model.

**Adaptive β control**:
```python
if KL > target_KL:
    β = β × 1.5  # Increase penalty (stay closer to SFT)
elif KL < target_KL / 2:
    β = β / 1.5  # Decrease penalty (explore more)
```

**Training Loop**:
```mermaid
graph LR
    Sample[Sample Prompt<br/>from dataset]
    Generate[Generate Response<br/>using current policy]
    Score[Score with RM<br/>r x,y]
    KL[Compute KL<br/>from SFT]
    Reward[Total Reward<br/>r - β·KL]
    Update[Update Policy<br/>PPO algorithm]

    Sample --> Generate
    Generate --> Score
    Score --> KL
    KL --> Reward
    Reward --> Update
    Update --> Sample

    style Sample fill:#e1f5ff
    style Generate fill:#fff4e1
    style Score fill:#f0e1ff
    style Update fill:#ffe1e1
```

### Safety Shield

After generation, validate the message:

```python
def validate_message(message: str) -> dict:
    violations = []

    # Check 1: No banned phrases
    banned = ["urgent", "act now", "limited time", "guaranteed"]
    if any(phrase in message.lower() for phrase in banned):
        violations.append("banned_phrase")

    # Check 2: Length check
    if not (50 <= len(message) <= 200):
        violations.append("invalid_length")

    # Check 3: Quiet hours (10pm - 8am)
    current_hour = datetime.now().hour
    if current_hour >= 22 or current_hour < 8:
        violations.append("quiet_hours")

    # Check 4: Required elements
    if not has_customer_name(message):
        violations.append("missing_name")
    if not has_offer_details(message):
        violations.append("missing_offer")

    return {
        "passed": len(violations) == 0,
        "violations": violations
    }
```

---

## Data Pipeline

```mermaid
graph TB
    subgraph Sources[Data Sources]
        Kaggle[Kaggle<br/>IBM Telco<br/>7,032 customers]
        UCI[UCI Repository<br/>Bank Marketing<br/>41,188 contacts]
        HF1[HuggingFace<br/>OASST1<br/>16,440 conversations]
        HF2[HuggingFace<br/>SHP-2 + HH-RLHF<br/>100,000 preferences]
    end

    subgraph Download[Download Stage]
        API1[Kaggle API]
        API2[HTTP Download]
        API3[HF Datasets API]

        Kaggle --> API1
        UCI --> API2
        HF1 --> API3
        HF2 --> API3
    end

    subgraph Process[Processing Stage]
        Clean[Clean Data<br/>handle missing values<br/>remove duplicates]
        Engineer[Feature Engineering<br/>create derived features<br/>encode categoricals]
        Split[Train/Valid/Test Split<br/>80/10/10<br/>seed=42]
        Format[Format Conversion<br/>Parquet for tabular<br/>JSONL for text]

        Clean --> Engineer
        Engineer --> Split
        Split --> Format
    end

    subgraph Validate[Validation Stage]
        Check1[Schema Validation]
        Check2[Data Quality Checks]
        Check3[Checksum Verification]

        Check1 --> Check2
        Check2 --> Check3
    end

    subgraph Upload[Upload Stage]
        GCS[Google Cloud Storage<br/>gs://plotpointe-churn-data/]
    end

    API1 --> Clean
    API2 --> Clean
    API3 --> Clean
    Format --> Check1
    Check3 --> GCS

    style Sources fill:#e1f5ff
    style Download fill:#fff4e1
    style Process fill:#f0e1ff
    style Validate fill:#ffe1e1
    style Upload fill:#e1ffe1
```

### Dataset Details

| Dataset | Size | Format | Purpose | Location |
|---------|------|--------|---------|----------|
| IBM Telco | 7,032 rows | Parquet | Risk model | `gs://.../processed/ibm_telco/` |
| UCI Bank | 41,188 rows | Parquet | Accept model | `gs://.../processed/uci_bank/` |
| OASST1 | 16,440 pairs | JSONL | SFT training | `gs://.../processed/oasst1/` |
| Preferences | 100,000 pairs | JSONL | RM training | `gs://.../processed/preferences/` |

### Data Processing Commands

```bash
# Download all datasets
make data.all

# Process individual datasets
make data.ibm_telco
make data.uci_bank
make data.oasst1
make data.preferences

# Upload to GCS
make data.upload
```

---

## Training Infrastructure

```mermaid
graph TB
    subgraph Local[Local Development]
        Code[Source Code<br/>Python, configs]
        Docker[Dockerfile<br/>CUDA 12.1, PyTorch 2.2]
    end

    subgraph CloudBuild[Cloud Build]
        Build[Build Docker Image<br/>~15 minutes]
        Push[Push to Artifact Registry<br/>us-central1-docker.pkg.dev]
    end

    subgraph VertexAI[Vertex AI Custom Jobs]
        Job1[Risk Model<br/>n1-standard-4 CPU<br/>30 min, $2]
        Job2[Accept Model<br/>n1-standard-4 CPU<br/>30 min, $3]
        Job3[SFT Model<br/>g2-standard-4 L4 GPU<br/>2 hrs, $30]
        Job4[RM Model<br/>g2-standard-4 L4 GPU<br/>1 hr, $15]
        Job5[PPO-Text<br/>g2-standard-4 L4 GPU<br/>2 hrs, $30]
        Job6[PPO-Decision<br/>n1-standard-4 CPU<br/>1 hr, $10]
    end

    subgraph Storage[Cloud Storage]
        Data[Data Bucket<br/>gs://plotpointe-churn-data/]
        Models[Model Bucket<br/>gs://plotpointe-churn-models/]
    end

    Code --> Build
    Docker --> Build
    Build --> Push
    Push --> Job1
    Push --> Job2
    Push --> Job3
    Push --> Job4
    Push --> Job5
    Push --> Job6

    Data --> Job1
    Data --> Job2
    Data --> Job3
    Data --> Job4
    Data --> Job5
    Data --> Job6

    Job1 --> Models
    Job2 --> Models
    Job3 --> Models
    Job4 --> Models
    Job5 --> Models
    Job6 --> Models

    style Local fill:#e1f5ff
    style CloudBuild fill:#fff4e1
    style VertexAI fill:#f0e1ff
    style Storage fill:#ffe1e1
```

### Job Dependencies

```mermaid
graph LR
    Risk[Risk Model<br/>30 min]
    Accept[Accept Model<br/>30 min]
    SFT[SFT Model<br/>2 hrs]
    RM[RM Model<br/>1 hr]
    PPODec[PPO Decision<br/>1 hr]
    PPOText[PPO Text<br/>2 hrs]

    Risk --> PPODec
    Accept --> PPODec
    SFT --> PPOText
    RM --> PPOText

    style Risk fill:#ccffcc
    style Accept fill:#ccffcc
    style SFT fill:#ffcccc
    style RM fill:#ffcccc
    style PPODec fill:#ccccff
    style PPOText fill:#ccccff
```

**Total Time**: ~8 hours (with parallelization)
**Total Cost**: ~$90

### Training Commands

```bash
# Submit all jobs with dependencies
./ops/scripts/submit_training_jobs.sh plotpointe us-central1

# Monitor progress
gcloud ai custom-jobs list --region=us-central1 --project=plotpointe

# Check logs
gcloud logging read "resource.type=ml_job" --limit=100
```

---

## Deployment Architecture

```mermaid
graph TB
    subgraph Client[Client]
        User[User Application]
        Request[HTTP POST /predict]
    end

    subgraph CloudRun[Cloud Run Service]
        LB[Load Balancer<br/>Auto-scaling]
        API[FastAPI Application<br/>Python 3.11]
        Health[Health Checks<br/>/healthz, /readyz]
    end

    subgraph Models[Model Loading]
        Risk[Risk Model<br/>XGBoost]
        Accept[Accept Model<br/>XGBoost]
        PPODec[PPO Decision<br/>PyTorch]
        PPOText[PPO Text<br/>Transformers]
    end

    subgraph Storage[Cloud Storage]
        GCS[Model Artifacts<br/>gs://plotpointe-churn-models/]
    end

    subgraph Monitoring[Monitoring]
        Logs[Cloud Logging]
        Metrics[Cloud Monitoring]
        Alerts[Alerting]
    end

    User --> Request
    Request --> LB
    LB --> API
    API --> Health

    GCS --> Risk
    GCS --> Accept
    GCS --> PPODec
    GCS --> PPOText

    Risk --> API
    Accept --> API
    PPODec --> API
    PPOText --> API

    API --> Logs
    API --> Metrics
    Metrics --> Alerts

    style Client fill:#e1f5ff
    style CloudRun fill:#fff4e1
    style Models fill:#f0e1ff
    style Storage fill:#ffe1e1
    style Monitoring fill:#e1ffe1
```

### API Endpoints

**POST /predict**
```json
{
  "customer_id": "C12345",
  "name": "Sam",
  "tenure": 12,
  "monthly_charges": 89.99,
  "contract": "Month-to-month",
  "payment_method": "Electronic check"
}
```

**Response**:
```json
{
  "decision": {
    "contact": true,
    "offer_level": 2,
    "offer_percentage": 10,
    "delay_days": 0,
    "estimated_retention_prob": 0.73
  },
  "message": "Hi Sam, we value your 12 months with us!...",
  "safety": {
    "passed": true,
    "violations": []
  },
  "metadata": {
    "churn_risk": 0.68,
    "accept_prob": 0.73,
    "expected_value": 45.23
  }
}
```

### Deployment Commands

```bash
# Deploy to Cloud Run
make deploy

# Check service status
gcloud run services describe churn-saver --region=us-central1

# View logs
gcloud logging read "resource.type=cloud_run_revision" --limit=50

# Test endpoint
curl -X POST https://your-service-url/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "C12345", ...}'
```

---

## Summary

This architecture combines:
- **XGBoost** for fast, accurate tabular predictions
- **PPO** for sequential decision-making with constraints
- **RLHF** for high-quality text generation
- **GCP** for scalable, cost-effective deployment

**Key Features**:
- ✅ Production-ready (90%+ test coverage)
- ✅ Scalable (auto-scaling, handles millions of customers)
- ✅ Cost-optimized (~$90 training, ~$10-30/month inference)
- ✅ Safe (safety shield, monitoring, alerts)
- ✅ Maintainable (clean code, comprehensive docs)


