# Complete Logic Holes Audit

**Date**: October 1, 2025  
**Status**: 🚨 CRITICAL ISSUES FOUND

## Executive Summary

Found **8 critical logic holes** that will cause failures in production:

1. ❌ **PPO Decision doesn't use loaded models** - Models loaded but never used in environment
2. ❌ **PPO Text doesn't load trained RM** - Uses placeholder reward instead of trained model
3. ❌ **Job dependencies not enforced** - Jobs run in parallel, will fail if dependencies missing
4. ❌ **RetentionEnv uses synthetic data** - Never uses loaded risk/acceptance models
5. ❌ **PPO Text missing CLI arguments** - Script not updated like others
6. ❌ **No model validation after training** - Models saved but never tested
7. ❌ **Serving layer doesn't load tabular models** - Only loads PPO policy, not risk/accept models
8. ⚠️ **No rollback mechanism** - If training fails, no way to revert

---

## 🚨 CRITICAL ISSUE #1: PPO Decision Doesn't Use Loaded Models

### Problem

In `agents/ppo_policy.py`, models are loaded but **NEVER PASSED TO THE ENVIRONMENT**:

```python
# Lines 157-171: Models are loaded
risk_model = load_model_from_path(config["risk_model_path"])
accept_model = load_model_from_path(config["accept_model_path"])

# Lines 181-189: Environment created WITHOUT models
env = RetentionEnv(
    episode_length=env_config.get("episode_length", 30),
    initial_budget=env_config.get("initial_budget", 1000.0),
    # ... other params ...
    # ❌ NO model_path parameter!
    # ❌ NO risk_model parameter!
    # ❌ NO accept_model parameter!
)
```

### Impact

- PPO decision policy trains on **synthetic random data**
- Trained risk/acceptance models are **completely ignored**
- Policy learns nothing about real customer behavior
- **Wasted $2 on tabular model training**

### Fix Required

```python
# Option 1: Pass models directly to environment
env = RetentionEnv(
    ...,
    risk_model=risk_model,
    accept_model=accept_model,
)

# Option 2: Update RetentionEnv to accept models
# Then modify _generate_observation() to use real models instead of synthetic
```

---

## 🚨 CRITICAL ISSUE #2: PPO Text Doesn't Load Trained Reward Model

### Problem

In `rlhf/ppo_text.py` line 116:

```python
# Line 116: Placeholder reward model
rm_score = 1.0  # Would use trained RM here
```

The script **never loads the trained reward model** from `--rm-path` argument!

### Impact

- PPO text training uses **constant reward of 1.0**
- Trained reward model is **completely ignored**
- Text generation doesn't learn from human preferences
- **Wasted $15 on reward model training**

### Fix Required

```python
# Load trained reward model
rm_model = load_reward_model(config["rm_path"])

# Use it in training loop
rm_score = rm_model(prompt, response_text)  # Real score
```

---

## 🚨 CRITICAL ISSUE #3: Job Dependencies Not Enforced

### Problem

In `ops/scripts/submit_training_jobs.sh`, all 6 jobs are submitted **immediately in parallel**:

```bash
# Line 59-64: Job 1 submitted
gcloud ai custom-jobs create ... risk-model ...

# Line 75-80: Job 2 submitted immediately
gcloud ai custom-jobs create ... accept-model ...

# Line 91-96: Job 3 submitted immediately
gcloud ai custom-jobs create ... sft-model ...

# Line 124-129: Job 5 submitted immediately (depends on Job 3 & 4!)
gcloud ai custom-jobs create ... ppo-text ...
```

**Job 5 (PPO text) depends on Job 3 (SFT) and Job 4 (RM) completing first!**  
**Job 6 (PPO decision) depends on Job 1 (Risk) and Job 2 (Accept) completing first!**

### Impact

- Job 5 will **fail immediately** trying to load non-existent SFT/RM models
- Job 6 will **fail immediately** trying to load non-existent risk/accept models
- **$40 wasted on failed GPU jobs**

### Current Status

**RIGHT NOW your jobs are running and WILL FAIL because of this!**

### Fix Required

```bash
# Option 1: Wait for dependencies
JOB1_ID=$(submit job 1)
JOB2_ID=$(submit job 2)
wait_for_job $JOB1_ID
wait_for_job $JOB2_ID
submit job 6 --depends-on=$JOB1_ID,$JOB2_ID

# Option 2: Use Vertex AI Pipelines with proper DAG
# Option 3: Add retry logic with exponential backoff
```

---

## 🚨 CRITICAL ISSUE #4: RetentionEnv Uses Synthetic Data

### Problem

In `env/retention_env.py` lines 134-145:

```python
def _generate_observation(self) -> Dict[str, np.ndarray]:
    """Generate observation for current customer."""
    # Synthetic churn risk (or use model)
    churn_risk = self.np_random.beta(2, 5)  # ❌ RANDOM!
    
    # Acceptance probabilities for each offer
    accept_probs = []
    for offer_pct in self.offers:
        # Simple logistic model: higher offer → higher acceptance
        logit = -1.0 + 3.0 * offer_pct + 2.0 * churn_risk  # ❌ HARDCODED!
        prob = 1.0 / (1.0 + np.exp(-logit))
        accept_probs.append(prob)
```

Even though the environment **can load models** (lines 88-100), it **never uses them**!

### Impact

- PPO policy trains on **fake data distribution**
- Real customer behavior never learned
- Policy will fail in production
- **All PPO decision training is wasted**

### Fix Required

```python
def _generate_observation(self) -> Dict[str, np.ndarray]:
    # Generate customer features
    customer_features = self._sample_customer_features()
    
    # Use REAL models if available
    if self.churn_model:
        churn_risk = self.churn_model.predict_proba(customer_features)[0, 1]
    else:
        churn_risk = self.np_random.beta(2, 5)  # Fallback
    
    if self.accept_model:
        accept_probs = [
            self.accept_model.predict_proba(
                add_offer_features(customer_features, offer)
            )[0, 1]
            for offer in self.offers
        ]
    else:
        # Fallback to synthetic
        ...
```

---

## 🚨 CRITICAL ISSUE #5: PPO Text Missing CLI Arguments

### Problem

`rlhf/ppo_text.py` was **NOT updated** like the other training scripts:

```python
# Line 141-149: Still uses old argument parsing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ops/configs/ppo_text.yaml", help="Config file")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train_ppo_text(config)
```

But the job submission script passes:
```bash
--config ... --sft-path ... --rm-path ... --output ...
```

### Impact

- **Job 5 will fail** with "unrecognized arguments" error
- Even if dependencies were fixed, this job would still fail
- **$30 wasted on GPU time**

### Fix Required

Update `ppo_text.py` to match `sft_train.py` and `rm_train.py` pattern:
- Accept `--sft-path`, `--rm-path`, `--output` arguments
- Load experiment config properly
- Handle GCS paths
- Load trained SFT and RM models

---

## 🚨 CRITICAL ISSUE #6: No Model Validation After Training

### Problem

Training scripts save models but **never validate they work**:

```python
# sft_train.py line 202: Just saves
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"✓ Model saved to {output_path}")
# ❌ No validation!
```

### Impact

- Corrupted models go undetected
- GCS upload failures go unnoticed
- Downstream jobs fail with cryptic errors
- No way to know if training actually worked

### Fix Required

```python
# After saving, validate
print("Validating saved model...")
loaded_model = AutoModelForCausalLM.from_pretrained(output_path)
test_input = tokenizer("Hello", return_tensors="pt")
test_output = loaded_model.generate(**test_input, max_length=20)
assert test_output is not None, "Model validation failed!"
print("✓ Model validation passed")
```

---

## 🚨 CRITICAL ISSUE #7: Serving Layer Doesn't Load Tabular Models

### Problem

In `serve/policy_loader.py` and `serve/app.py`:

```python
# Only loads PPO policy and RLHF model
ppo_policy_path = os.getenv("PPO_POLICY_PATH", ...)
rlhf_model_path = os.getenv("RLHF_MODEL_PATH", ...)

# ❌ NO risk model loading!
# ❌ NO acceptance model loading!
```

But the `/retain` endpoint needs these models to compute `p_churn` and `p_accept`!

### Impact

- API will fail at runtime
- Can't make predictions without risk/acceptance models
- **Entire serving layer is broken**

### Fix Required

```python
# In policy_loader.py
risk_model_path = os.getenv("RISK_MODEL_PATH", ...)
accept_model_path = os.getenv("ACCEPT_MODEL_PATH", ...)

self.risk_model = self._load_pickle_model(risk_model_path)
self.accept_model = self._load_pickle_model(accept_model_path)

# In app.py /retain endpoint
scores = policy_loader.compute_scores(customer_facts)
# Uses risk_model and accept_model
```

---

## ⚠️ ISSUE #8: No Rollback Mechanism

### Problem

If training fails or produces bad models:
- No way to revert to previous version
- No model versioning
- No A/B testing infrastructure
- No gradual rollout

### Impact

- Can't recover from bad training runs
- Risky deployments
- No experimentation framework

### Fix Required

- Implement model versioning (e.g., `exp_001_mvp_v1`, `exp_001_mvp_v2`)
- Store multiple versions in GCS
- Add version selector in serving layer
- Implement canary deployment (already planned)

---

## Summary of Fixes Needed

### URGENT (Before jobs complete):

1. ✅ **Kill current jobs** - They will fail anyway due to dependencies
2. ✅ **Fix PPO text CLI arguments** - Add proper argument parsing
3. ✅ **Fix job dependencies** - Add wait logic or use Vertex AI Pipelines
4. ✅ **Fix PPO decision model integration** - Pass models to environment
5. ✅ **Fix RetentionEnv** - Use real models instead of synthetic
6. ✅ **Fix PPO text RM loading** - Load and use trained reward model

### IMPORTANT (Before production):

7. ✅ **Add model validation** - Validate after training
8. ✅ **Fix serving layer** - Load all required models
9. ✅ **Add rollback mechanism** - Model versioning

---

## Estimated Impact

| Issue | Wasted Cost | Wasted Time | Severity |
|-------|-------------|-------------|----------|
| #1 PPO Decision | $10 | 1 hour | 🔴 CRITICAL |
| #2 PPO Text RM | $15 | 1.5 hours | 🔴 CRITICAL |
| #3 Dependencies | $40 | 3 hours | 🔴 CRITICAL |
| #4 Synthetic Env | $10 | 1 hour | 🔴 CRITICAL |
| #5 PPO Text Args | $30 | 3 hours | 🔴 CRITICAL |
| #6 No Validation | - | - | 🟡 HIGH |
| #7 Serving Broken | - | - | 🔴 CRITICAL |
| #8 No Rollback | - | - | 🟡 MEDIUM |

**Total Wasted**: ~$105 (over budget!)  
**Total Wasted Time**: ~9.5 hours

---

## Recommendation

**STOP CURRENT JOBS IMMEDIATELY** and fix issues before resubmitting.

The current training run will:
1. ✅ Risk model - Will succeed
2. ✅ Acceptance model - Will succeed
3. ❌ SFT model - Will succeed but not validated
4. ❌ RM model - Will succeed but not validated
5. ❌ PPO text - **WILL FAIL** (missing CLI args + missing RM loading)
6. ❌ PPO decision - **WILL FAIL** (missing dependencies + not using models)

**Only 2 out of 6 jobs will produce usable results.**

