# Data Pipeline Documentation

## Overview

The churn-saver system uses 4 real-world datasets for training:
1. **IBM Telco** - Churn risk modeling
2. **UCI Bank Marketing** - Offer acceptance modeling
3. **OASST1** - SFT training for message generation
4. **SHP-2 + HH-RLHF** - Preference pairs for reward model

All data is downloaded, processed, validated, and tracked automatically.

## Directory Structure

```
data/
  raw/                          # Downloaded raw data
    telco/
      WA_Fn-UseC_-Telco-Customer-Churn.csv
    bank_marketing/
      bank-additional/
        bank-additional-full.csv
    oasst1/                     # HF cache
    preferences/                # HF cache
  
  processed/                    # Processed, model-ready data
    telco/
      telco.parquet             # Full dataset
      telco_train.parquet       # 80% train
      telco_valid.parquet       # 10% valid
      telco_test.parquet        # 10% test
      stats.json                # Dataset statistics
    
    bank_marketing/
      bank.parquet
      bank_train.parquet
      bank_valid.parquet
      bank_test.parquet
      stats.json
    
    oasst1/
      sft_train.jsonl           # 60k max prompt-response pairs
      sft_valid.jsonl           # 2k validation pairs
    
    preferences/
      pairs.jsonl               # 100k max preference pairs
      pairs_valid.jsonl         # 10k validation pairs
      hh_probe.jsonl            # 1k HH harmlessness probe
  
  catalog.yaml                  # Artifact registry with checksums
```

## Quick Start

```bash
# Setup Kaggle credentials first
mkdir -p ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Process all datasets
make data.all

# Or process individually
make data.telco
make data.bank
make data.sft
make data.prefs

# View catalog
make data.catalog
```

## Dataset Details

### 1. IBM Telco Customer Churn

**Source**: Kaggle `blastchar/telco-customer-churn`

**Processing**:
- Coerce `TotalCharges` to numeric, drop NAs
- Map `Churn` to binary (Yes→1, No→0)
- Keep columns: `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, `Churn`
- Create 80/10/10 stratified splits

**Validation**:
- Min 5,000 rows
- Churn rate between 0.1 and 0.6
- All required columns present

**Output**: Parquet files with deterministic splits (seed=42)

**Stats**: `data/processed/telco/stats.json`
```json
{
  "total_rows": 7043,
  "churn_rate": 0.265,
  "train_rows": 5634,
  "valid_rows": 704,
  "test_rows": 705
}
```

### 2. UCI Bank Marketing

**Source**: UCI ML Repository (bank-additional-full.csv)

**Processing**:
- Load semicolon-separated CSV
- Map `y` to binary (yes→1, no→0)
- Compute `offer_level` (0-3) proxy from campaign features:
  - Uses `campaign`, `pdays`, `previous` to estimate incentive intensity
  - Higher campaign frequency + recent contact = higher offer level
- Create 80/10/10 stratified splits

**Validation**:
- Min 30,000 rows
- Required columns include `y` and `offer_level`

**Output**: Parquet files with splits

**Stats**: `data/processed/bank_marketing/stats.json`
```json
{
  "total_rows": 41188,
  "acceptance_rate": 0.112,
  "offer_level_dist": {"0": 15234, "1": 12456, "2": 8901, "3": 4597}
}
```

### 3. OASST1 (OpenAssistant)

**Source**: HuggingFace `OpenAssistant/oasst1`

**Processing**:
- Extract prompt-response pairs from conversation trees
- Filter by length (min 10 chars) and token count (max 1024 tokens each)
- Cap at 60k train + 2k valid pairs
- Shuffle with seed=42

**Validation**:
- Min 1,000 pairs
- Required fields: `prompt`, `response`
- No empty strings
- Token limits enforced

**Output**: JSONL files
```jsonl
{"prompt": "How do I...", "response": "You can..."}
```

### 4. SHP-2 + HH-RLHF Preferences

**Sources**:
- HuggingFace `stanfordnlp/SHP-2` (capped at 60k)
- HuggingFace `Anthropic/hh-rlhf` (capped at 40k)

**Processing**:
- Extract preference pairs from both datasets
- Normalize to unified format: `{prompt, chosen, rejected, source}`
- Filter by length and token count
- Combine and shuffle
- Cap at 100k total pairs
- Split: 90k train, 10k valid
- Save 1k HH pairs for harmlessness probing

**Validation**:
- Min 10,000 pairs
- Required fields: `prompt`, `chosen`, `rejected`, `source`
- No duplicate lines

**Output**: JSONL files
```jsonl
{"prompt": "...", "chosen": "...", "rejected": "...", "source": "shp2"}
```

## Configuration

All settings in `ops/configs/data_config.yaml`:

```yaml
enable:
  telco: true
  bank: true
  oasst1: true
  shp2: true
  hh: true

caps:
  oasst1_max_rows: 60000
  oasst1_valid_rows: 2000
  prefs_max_pairs: 100000
  prefs_valid_pairs: 10000
  shp2_max: 60000
  hh_max: 40000

text_limits:
  max_prompt_tokens: 1024
  max_response_tokens: 1024
  min_text_length: 10

splits:
  train: 0.8
  valid: 0.1
  test: 0.1
  seed: 42
```

## Data Catalog

The catalog (`data/catalog.yaml`) tracks all processed artifacts:

```yaml
telco:
  path: data/processed/telco/telco.parquet
  rows: 7043
  checksum_sha256: abc123...
  last_updated: 2025-09-30T12:00:00Z
  size_bytes: 524288
  stats:
    churn_rate: 0.265
```

**Features**:
- SHA256 checksums for integrity
- Row counts and file sizes
- Last updated timestamps
- Dataset-specific stats
- Idempotency: skips reprocessing if checksum matches

## Idempotency

The pipeline is idempotent:
- Downloads are skipped if files exist
- Processing is skipped if catalog entry matches current file
- Use `--force` flag to force reprocessing

```bash
# Force reprocess
python data/processors/telco_processor.py --force
```

## Validation

Each dataset has validation checks:

**Telco**:
- Row count ≥ 5,000
- Churn rate between 0.1 and 0.6
- Required columns present

**Bank**:
- Row count ≥ 30,000
- Required columns including `offer_level`

**OASST1**:
- Pair count ≥ 1,000
- Required fields present
- Length and token limits

**Preferences**:
- Pair count ≥ 10,000
- Required fields present
- No duplicates

Validation failures raise exceptions and prevent catalog registration.

## Testing

Data pipeline tests in `tests/data/`:

```bash
# Test data processors
pytest tests/data/test_telco_processor.py
pytest tests/data/test_bank_processor.py
pytest tests/data/test_oasst1_processor.py
pytest tests/data/test_preferences_processor.py

# Test catalog
pytest tests/data/test_catalog.py
```

Tests verify:
- File existence
- Schema correctness
- Row counts > 0
- Split determinism (same seed = same splits)
- Catalog integrity

## Troubleshooting

### Kaggle credentials not found
```
FileNotFoundError: Kaggle credentials not found
```
**Fix**: Download `kaggle.json` from https://www.kaggle.com/settings/account and save to `~/.kaggle/kaggle.json` with chmod 600

### HuggingFace rate limit
```
HTTPError: 429 Too Many Requests
```
**Fix**: Wait a few minutes or set `HF_TOKEN` environment variable with your HF token

### Out of disk space
```
OSError: No space left on device
```
**Fix**: Free up space. Raw data ~2GB, processed data ~1GB

### Checksum mismatch
```
Checksum mismatch for telco.parquet
```
**Fix**: File was modified. Use `--force` to reprocess

## Environment Variables

- `DATA_ROOT`: Base directory for data (default: `data/`)
- `HF_TOKEN`: HuggingFace API token (optional, for private datasets)
- `KAGGLE_USERNAME`: Kaggle username (from kaggle.json)
- `KAGGLE_KEY`: Kaggle API key (from kaggle.json)

## Next Steps

After processing data:

1. **Train risk models**: `make train.risk`
2. **Train acceptance model**: `make train.accept`
3. **Train SFT model**: `make train.sft`
4. **Train reward model**: `make train.rm`
5. **Train PPO policies**: `make train.ppo.decision` and `make train.ppo.text`
6. **Serve API**: `make serve`

