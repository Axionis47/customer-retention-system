# Training Playbook: exp_001_mvp

## Status: IN PROGRESS

This document tracks the implementation of the complete training pipeline for the MVP experiment under $100 budget.

## ✅ Completed

### 1. Lock Configs for Reproducibility ✓
- **File**: `ops/configs/experiment_exp_001_mvp.yaml`
- **Status**: COMPLETE
- **Details**:
  - Experiment tag: `exp_001_mvp`
  - Global seed: 42
  - All hyperparameters frozen
  - Data caps configured (Telco: 7k, Bank: 41k, OASST1: 60k, Prefs: 100k)
  - Exit criteria defined for all models
  - Cost guardrails: $100 budget, L4 preemptible GPU
  - Monitoring and canary config included

### 2. Data Pipeline ✓
- **Status**: COMPLETE & TESTED
- **Datasets**:
  - Telco: 7,032 rows (churn risk)
  - Bank: 41,188 rows (offer acceptance)
  - OASST1: 18,440 pairs (SFT)
  - Preferences: 100,000 pairs (RM)
- **Catalog**: All artifacts tracked with checksums
- **API**: `/retain` endpoint tested and working

### 3. Training Infrastructure ✓
- **Updated Files**:
  - `models/risk_accept/train_churn.py` - Updated for Telco data with calibration
  - `Makefile` - Added `train.risk`, `train.accept`, `train.sft`, `train.rm`, `train.ppo.*` targets
- **Features**:
  - Loads from processed parquet files
  - Calibration with isotonic regression
  - ECE (Expected Calibration Error) computation
  - Exit criteria validation
  - Metrics saved as JSON

## 🚧 In Progress

### Task 2: Train and Calibrate Tabular Models
**Status**: BLOCKED - XGBoost dependency issue

**Issue**: XGBoost requires `libomp` (OpenMP runtime) on macOS
```
Error: Library not loaded: @rpath/libomp.dylib
```

**Solutions**:
1. **Install libomp** (Recommended):
   ```bash
   brew install libomp
   ```

2. **Use scikit-learn models instead** (Alternative):
   - Replace XGBoost with RandomForest or GradientBoosting
   - Slightly lower performance but no dependency issues

**Next Steps**:
```bash
# Option 1: Install libomp and run training
brew install libomp
make train.risk
make train.accept

# Option 2: Switch to sklearn models
# (requires updating train_churn.py and train_accept.py)
```

**Exit Criteria**:
- Risk model AUC ≥ 0.78
- Acceptance model AUC ≥ 0.70
- Both models ECE < 0.05

## 📋 Remaining Tasks

### Task 3: Train PPO Decision Policy
**Status**: NOT STARTED
**Dependencies**: Task 2 (tabular models)

**Steps**:
1. Update `agents/ppo_policy.py` to load from experiment config
2. Integrate calibrated risk and acceptance models
3. Train Lagrangian-PPO with budget/contact constraints
4. Compare to 3 baselines:
   - Propensity threshold + fixed 10% offer
   - Thompson Sampling over {0%, 5%, 10%, 20%}
   - Uplift tree (two-model approach)
5. Generate comparison plots and tables

**Exit Criteria**:
- NRR improvement ≥ +3% vs best baseline
- Offer cost reduction ≥ -5% vs best baseline
- Constraint violations ≈ 0%
- Fatigue over-cap < 1%

**Command**:
```bash
make train.ppo.decision
```

### Task 4: Train RLHF Pipeline (SFT → RM → PPO)
**Status**: NOT STARTED
**Dependencies**: None (can run in parallel with Task 3)

**Steps**:

#### 4a. SFT (Supervised Fine-Tuning)
- Base model: `facebook/opt-350m` (small for budget)
- Train on OASST1 (18,440 pairs)
- LoRA adapters (r=16, alpha=32)
- 8-bit quantization
- Max 2,000 steps

**Command**:
```bash
make train.sft
```

**Exit Criteria**:
- Training loss < 2.5
- Eval perplexity reasonable

#### 4b. Reward Model
- Base model: `facebook/opt-350m`
- Train on SHP-2 + HH-RLHF (100k pairs)
- LoRA adapters
- 8-bit quantization
- Max 1,000 steps

**Command**:
```bash
make train.rm
```

**Exit Criteria**:
- Validation AUC ≥ 0.70
- Accuracy ≥ 0.65

#### 4c. PPO Text Generation
- Load SFT model + Reward model
- Adaptive KL control (target=0.15)
- Reward shaping: safety (2.0x) + brevity (0.5x) + RM (1.0x)
- Max 180 tokens per message
- 200 iterations

**Command**:
```bash
make train.ppo.text
```

**Exit Criteria**:
- Win-rate vs SFT ≥ +8pp (95% CI > 0)
- Safety violations < 1%
- Avg tokens ≤ 162 (-10% from 180)

### Task 5: Create Offline Evaluation Bundle
**Status**: NOT STARTED
**Dependencies**: Tasks 3 & 4

**Deliverables**:
1. **Plots**:
   - PPO decision learning curves (NRR, cost, violations)
   - Constraint violations → 0 over time
   - Action entropy histogram
   - NRR vs baselines (bar chart with CI)
   - Cost vs baselines (bar chart with CI)
   - RM AUC ROC curve
   - RLHF win-rate bar chart
   - KL divergence trace
   - Safety probe pass-rate

2. **Tables**:
   - Decision policy metrics (mean ± 95% CI)
   - Baseline comparison
   - RLHF metrics
   - Safety statistics

**Location**: `eval/results/exp_001_mvp/`

**Command**:
```bash
python eval/generate_report.py --experiment exp_001_mvp
```

### Task 6: Integrate Models into API (Shadow Mode)
**Status**: NOT STARTED
**Dependencies**: Tasks 3 & 4

**Steps**:
1. Update `serve/app.py` to load models from experiment artifacts
2. Add environment variables:
   ```bash
   export RISK_MODEL_PATH=models/risk_accept/artifacts/exp_001_mvp_risk_model.pkl
   export ACCEPT_MODEL_PATH=models/risk_accept/artifacts/exp_001_mvp_accept_model.pkl
   export SFT_PATH=checkpoints/exp_001_mvp_sft
   export RM_PATH=checkpoints/exp_001_mvp_rm
   export RLHF_PATH=checkpoints/exp_001_mvp_ppo_text
   export PPO_POLICY_PATH=checkpoints/exp_001_mvp_ppo_decision/policy.pt
   export FORCE_BASELINE=true  # Shadow mode
   ```
3. Add shadow logging:
   - Log both baseline and learned decisions
   - Log both SFT and RLHF messages
   - Track latency, safety, scores
4. Monitor shadow KPIs:
   - Latency p95 < 500ms
   - Safety violations < 0.5%
   - Message length distribution
   - RM score distribution

**Command**:
```bash
make serve
```

### Task 7: Deploy Canary with Monitoring
**Status**: NOT STARTED
**Dependencies**: Task 6 (shadow mode validated)

**Steps**:
1. Set up monitoring dashboards
2. Configure alerts:
   - 5xx rate > 1%
   - p95 latency > 500ms
   - Safety violations > 0.5%
   - NRR proxy drop > 3pp vs baseline
3. Enable canary:
   ```bash
   export FORCE_BASELINE=false
   export CANARY_PERCENT=5
   ```
4. Monitor for 24 hours
5. Implement auto-rollback:
   - If any alert persists > 10 min
   - Automatically set `FORCE_BASELINE=true`
   - Page on-call

**Rollback Command**:
```bash
export FORCE_BASELINE=true
# Restart service
```

### Task 8: Create Operator Runbook
**Status**: NOT STARTED

**Sections**:
1. **Model Version Updates**
   - How to update model paths
   - How to validate new models
   - Rollback procedure

2. **Canary Management**
   - How to enable/disable canary
   - How to adjust traffic percentage
   - How to monitor canary health

3. **Dashboard Reading**
   - Key metrics to watch
   - Normal vs abnormal patterns
   - Alert interpretation

4. **Common Remediations**
   - High latency → increase resources
   - High 5xx → rollback models
   - Safety violations → tighten rules
   - NRR drop → revert to baseline

**Location**: `docs/OPERATOR_RUNBOOK.md`

## Quick Commands

```bash
# Complete training pipeline
make train.all

# Individual steps
make train.risk          # Train churn risk model
make train.accept        # Train offer acceptance model
make train.sft           # Train SFT model
make train.rm            # Train reward model
make train.ppo.text      # Train PPO text generation
make train.ppo.decision  # Train PPO decision policy

# Evaluation
python eval/generate_report.py --experiment exp_001_mvp

# Serve with trained models
export RISK_MODEL_PATH=models/risk_accept/artifacts/exp_001_mvp_risk_model.pkl
export ACCEPT_MODEL_PATH=models/risk_accept/artifacts/exp_001_mvp_accept_model.pkl
make serve
```

## Cost Tracking

| Component | Estimated | Actual | Status |
|-----------|-----------|--------|--------|
| Risk + Accept Training | $5 | - | Pending |
| SFT Training | $30 | - | Pending |
| RM Training | $15 | - | Pending |
| PPO Text Training | $30 | - | Pending |
| PPO Decision Training | $10 | - | Pending |
| **Total** | **$90** | **$0** | **Under Budget** |

## Timeline

- **Day 1**: Train tabular models (Tasks 2)
- **Day 2-3**: Train RLHF pipeline (Task 4)
- **Day 3-4**: Train PPO decision (Task 3)
- **Day 4**: Generate evaluation bundle (Task 5)
- **Day 5**: Shadow mode testing (Task 6)
- **Day 6-7**: Canary deployment (Task 7)
- **Day 7**: Create runbook (Task 8)

**Total**: ~7 days to production canary

## Next Immediate Action

**Install libomp and train tabular models:**
```bash
brew install libomp
make train.risk
make train.accept
```

Once tabular models are trained, proceed with PPO decision training and RLHF pipeline in parallel.

