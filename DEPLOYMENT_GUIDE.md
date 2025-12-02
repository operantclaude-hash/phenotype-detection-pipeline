# Deployment Guide - CORRECT NORMALIZATION

**Date:** December 2, 2025
**Status:** Ready for deployment after RESCUE retraining completes

---

## Overview

This guide describes how to deploy all trained models to complete plates using the **EXACT SAME normalization** across all model types.

All models use the **identical 3-stage normalization pipeline**:
1. Percentile clipping [p1, p99]
2. Normalize to [0, 1]
3. ImageNet normalization

---

## Model Types

### 1. RESCUE @ 0ms (3-channel)
- **Purpose:** APP vs WT classification at baseline (T0)
- **Channels:** RFP1, HALO, HALO_MASKED (single timepoint)
- **Models:** `rescue_0ms_replicates_CORRECT_NORM/` (10 replicates)
- **Input:** Single timepoint images at T0

### 2. RESCUE @ 10000ms (3-channel)
- **Purpose:** APP vs WT classification post-stimulation (T15)
- **Channels:** RFP1, HALO, HALO_MASKED (single timepoint)
- **Models:** `rescue_10000ms_replicates_CORRECT_NORM/` (10 replicates)
- **Input:** Single timepoint images at T15

### 3. APP Treatment (6-channel)
- **Purpose:** Temporal change detection in APP neurons (T0 vs T15)
- **Channels:** [RFP1, HALO, HALO_MASKED]_T0 + [RFP1, HALO, HALO_MASKED]_T15
- **Models:** `app_treatment_binary_FIXED_MASK/`
- **Input:** Two timepoint images (T0 + T15) for APP genotype only

### 4. WT Treatment (6-channel)
- **Purpose:** Temporal change detection in WT neurons (T0 vs T15)
- **Channels:** [RFP1, HALO, HALO_MASKED]_T0 + [RFP1, HALO, HALO_MASKED]_T15
- **Models:** `wt_treatment_binary_FIXED_MASK/`
- **Input:** Two timepoint images (T0 + T15) for WT genotype only

---

## Deployment Scripts

### Core Deployment Script
**Location:** `scripts/deploy_all_models_CORRECT_NORM.py`

**Key features:**
- Uses EXACT SAME normalization as training
- Loads percentiles from `intensity_percentiles.json`
- Applies 3-stage normalization pipeline
- Supports both 3-channel and 6-channel models

**Usage:**
```bash
python3 scripts/deploy_all_models_CORRECT_NORM.py \
    --model_type rescue_0ms \
    --checkpoint path/to/checkpoint.ckpt \
    --metadata path/to/metadata.csv \
    --output path/to/predictions.csv \
    --batch_size 64
```

**Model type options:**
- `rescue_0ms`: RESCUE at baseline T0
- `rescue_10000ms`: RESCUE at post-stimulation T15
- `app_treatment`: APP temporal change
- `wt_treatment`: WT temporal change
- `binary`: Generic 6-channel binary classifier

### Master Deployment Script
**Location:** `deploy_complete_plates_all_models.sh`

Orchestrates deployment of all 4 model types to complete plates.

**Status:** Template ready, requires:
1. RESCUE model retraining to complete
2. Complete plate metadata CSVs to be created

---

## Normalization Pipeline (CRITICAL)

**ALL models use this EXACT normalization:**

### Stage 1: Percentile Clipping
```python
# Load percentile bounds
with open('A2screen_combined/data/intensity_percentiles.json') as f:
    percentiles = json.load(f)

# Clip each channel to [p1, p99]
img_clipped = np.clip(img, p1, p99)
```

**Percentile values:**
- RFP1: p1=963, p99=9016
- HALO: p1=1783, p99=11525
- HALO_MASKED: p1=0, p99=238 (uses HALO percentiles in code)

### Stage 2: Normalize to [0, 1]
```python
if p99 > p1:
    img_normalized = (img_clipped - p1) / (p99 - p1)
else:
    img_normalized = img_clipped
```

### Stage 3: ImageNet Normalization
```python
# For 3-channel models (RESCUE)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# For 6-channel models (treatment)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406] * 2,
    std=[0.229, 0.224, 0.225] * 2
)
```

---

## Deployment Workflow

### Step 1: Wait for RESCUE Retraining
**Status:** In progress (background process f9882c)
- Training 20 models (10 @ 0ms + 10 @ 10000ms)
- Estimated completion: 60-80 hours

### Step 2: Verify Model Checkpoints
```bash
# Check RESCUE @ 0ms
ls A2screen_combined/models/rescue_0ms_replicates_CORRECT_NORM/replicate_*/checkpoints/

# Check RESCUE @ 10000ms
ls A2screen_combined/models/rescue_10000ms_replicates_CORRECT_NORM/replicate_*/checkpoints/

# Check APP treatment
ls A2screen_combined/models/app_treatment_binary_FIXED_MASK/checkpoints/

# Check WT treatment
ls A2screen_combined/models/wt_treatment_binary_FIXED_MASK/checkpoints/
```

### Step 3: Create Complete Plate Metadata CSVs
**TODO:** Create metadata files for all wells (columns 1-24) including:

**For RESCUE @ 0ms:**
- File: `complete_plate_0ms_metadata.csv`
- Contents: All wells at T0 (plate1-T0, plate2-T0)
- Columns: neuron_id, plate, well, timepoint, rfp1_path, halo_path, halo_masked_path, genotype, root_dir

**For RESCUE @ 10000ms:**
- File: `complete_plate_10000ms_metadata.csv`
- Contents: All wells at T15 (plate1-T14, plate2-T15)
- Columns: Same as above

**For APP treatment:**
- File: `complete_plate_app_treatment_metadata.csv`
- Contents: APP wells only, both T0 and T15
- Columns: neuron_id, plate, well, rfp1_t0_path, halo_t0_path, halo_masked_t0_path, rfp1_t15_path, halo_t15_path, halo_masked_t15_path, genotype, root_dir

**For WT treatment:**
- File: `complete_plate_wt_treatment_metadata.csv`
- Contents: WT wells only, both T0 and T15
- Columns: Same as APP treatment

### Step 4: Run Deployments
```bash
# Deploy all models
./deploy_complete_plates_all_models.sh
```

Or deploy individually:
```bash
# RESCUE @ 0ms
python3 scripts/deploy_all_models_CORRECT_NORM.py \
    --model_type rescue_0ms \
    --checkpoint A2screen_combined/models/rescue_0ms_replicates_CORRECT_NORM/replicate_01/checkpoints/best*.ckpt \
    --metadata complete_plate_0ms_metadata.csv \
    --output predictions_rescue_0ms_complete.csv

# RESCUE @ 10000ms
python3 scripts/deploy_all_models_CORRECT_NORM.py \
    --model_type rescue_10000ms \
    --checkpoint A2screen_combined/models/rescue_10000ms_replicates_CORRECT_NORM/replicate_01/checkpoints/best*.ckpt \
    --metadata complete_plate_10000ms_metadata.csv \
    --output predictions_rescue_10000ms_complete.csv

# APP Treatment
python3 scripts/deploy_all_models_CORRECT_NORM.py \
    --model_type app_treatment \
    --checkpoint A2screen_combined/models/app_treatment_binary_FIXED_MASK/checkpoints/best*.ckpt \
    --metadata complete_plate_app_treatment_metadata.csv \
    --output predictions_app_treatment_complete.csv

# WT Treatment
python3 scripts/deploy_all_models_CORRECT_NORM.py \
    --model_type wt_treatment \
    --checkpoint A2screen_combined/models/wt_treatment_binary_FIXED_MASK/checkpoints/best*.ckpt \
    --metadata complete_plate_wt_treatment_metadata.csv \
    --output predictions_wt_treatment_complete.csv
```

### Step 5: Aggregate Results
Combine predictions from all models for comprehensive analysis.

---

## Verification Checklist

Before deploying, verify:

- [x] Deployment script uses correct normalization (percentile + ImageNet)
- [x] Percentiles loaded from `intensity_percentiles.json`
- [x] 3-stage pipeline implemented correctly
- [ ] RESCUE model retraining completed
- [ ] All checkpoints exist and are valid
- [ ] Complete plate metadata CSVs created
- [ ] Image paths in metadata are correct
- [ ] All required channels present in data directory

---

## Output Format

Each deployment produces a CSV with columns:
- All original metadata columns
- `predicted_class`: 0 or 1
- `prob_class_0`: Probability of class 0
- `prob_class_1`: Probability of class 1
- `model_type`: Which model was used
- `checkpoint`: Path to checkpoint used

---

## Troubleshooting

### Issue: "Checkpoint not found"
**Solution:** Verify training completed and checkpoint exists:
```bash
find A2screen_combined/models -name "*.ckpt"
```

### Issue: "Metadata file missing columns"
**Solution:** Verify metadata has required columns:
- 3-channel: rfp1_path, halo_path, halo_masked_path
- 6-channel: rfp1_t0_path, halo_t0_path, halo_masked_t0_path, rfp1_t15_path, halo_t15_path, halo_masked_t15_path

### Issue: "Image files not found"
**Solution:** Check root_dir in metadata and verify image paths are relative to that directory

### Issue: "Out of memory during deployment"
**Solution:** Reduce batch size:
```bash
python3 scripts/deploy_all_models_CORRECT_NORM.py ... --batch_size 32
```

---

## References

- **Training Scripts:** `scripts/train_rescue_3channel_CORRECT_NORM.py`, `A2screen_combined/scripts/train_binary_6channel_FIXED_NORM.py`
- **Normalization Documentation:** `NORMALIZATION_DOCUMENTATION.md`
- **Percentile File:** `A2screen_combined/data/intensity_percentiles.json`
