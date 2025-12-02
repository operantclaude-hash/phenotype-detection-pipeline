# Normalization Documentation

## Critical: ALL Models Use SAME Normalization

**Date:** December 2, 2025
**Status:** CORRECTED - All models now use identical normalization

---

## Normalization Pipeline (Applied to ALL Models)

All models (RESCUE 3-channel, APP/WT treatment 6-channel, and binary 6-channel) use the **EXACT SAME** three-stage normalization pipeline:

###  Stage 1: Percentile Clipping
```python
# Clip pixel values to percentile bounds [p1, p99]
img_clipped = np.clip(img, p1, p99)
```

**Percentile values** (computed from training set):
- **RFP1**: p1=963, p99=9016
- **HALO**: p1=1783, p99=11525
- **HALO_MASKED**: p1=0, p99=238

Stored in: `A2screen_combined/data/intensity_percentiles.json`

### Stage 2: Normalize to [0, 1]
```python
# Normalize clipped values to [0, 1] range
if p99 > p1:
    img_normalized = (img_clipped - p1) / (p99 - p1)
else:
    img_normalized = img_clipped
```

### Stage 3: ImageNet Normalization
```python
# Apply ImageNet normalization (after conversion to tensor)
# 3-channel (RESCUE):
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 6-channel (APP/WT/Binary treatment):
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406] * 2,  # Repeat for T0 and T15
    std=[0.229, 0.224, 0.225] * 2
)
```

---

## Model-Specific Details

### 1. RESCUE Models (3-channel: RFP1, HALO, HALO_MASKED)

**Purpose:** APP vs WT classification at single timepoints

**Channels:** 3 (one timepoint)
- Channel 0: RFP1
- Channel 1: HALO
- Channel 2: HALO_MASKED

**Normalization:**
- Percentile clipping → [0,1] normalization → ImageNet (3-channel)

**Training Script:** `scripts/train_rescue_3channel_CORRECT_NORM.py`

**Models:**
- `rescue_0ms_replicates_CORRECT_NORM/` (10 replicates @ T0)
- `rescue_10000ms_replicates_CORRECT_NORM/` (10 replicates @ T15)

---

### 2. APP Treatment Models (6-channel: T0 + T15)

**Purpose:** Detect temporal changes in APP neurons (APP_T0 vs APP_T15)

**Channels:** 6 (two timepoints)
- Channels 0-2: RFP1_T0, HALO_T0, HALO_MASKED_T0
- Channels 3-5: RFP1_T15, HALO_T15, HALO_MASKED_T15

**Normalization:**
- Percentile clipping → [0,1] normalization → ImageNet (6-channel)

**Training Script:** `A2screen_combined/scripts/train_binary_6channel_FIXED_NORM.py`

**Models:**
- `app_treatment_replicates_CORRECT/` (10 replicates)

---

### 3. WT Treatment Models (6-channel: T0 + T15)

**Purpose:** Detect temporal changes in WT neurons (WT_T0 vs WT_T15)

**Channels:** 6 (two timepoints)
- Channels 0-2: RFP1_T0, HALO_T0, HALO_MASKED_T0
- Channels 3-5: RFP1_T15, HALO_T15, HALO_MASKED_T15

**Normalization:**
- Percentile clipping → [0,1] normalization → ImageNet (6-channel)

**Training Script:** `A2screen_combined/scripts/train_binary_6channel_FIXED_NORM.py`

**Models:**
- `wt_treatment_replicates_CORRECT/` (10 replicates)

---

### 4. Binary 6-Channel Models (Generic temporal classifier)

**Purpose:** General temporal change detection (T0 vs T15, any genotype)

**Channels:** 6 (two timepoints)
- Channels 0-2: RFP1_T0, HALO_T0, HALO_MASKED_T0
- Channels 3-5: RFP1_T15, HALO_T15, HALO_MASKED_T15

**Normalization:**
- Percentile clipping → [0,1] normalization → ImageNet (6-channel)

**Training Script:** `A2screen_combined/scripts/train_binary_6channel_FIXED_NORM.py`

**Models:**
- Various binary models in `A2screen_combined/models/binary_6channel*/`

---

## Historical Note: Normalization Bug (FIXED)

### The Problem
Initially, RESCUE models were trained with **fluorescence-specific normalization**:
```python
# OLD INCORRECT normalization (used initially)
mean=[0.2233, 0.2073, 0.0507]
std=[0.1644, 0.1699, 0.1601]
```

This was DIFFERENT from the 6-channel models which used the correct 3-stage pipeline.

### The Fix (December 2, 2025)
ALL RESCUE models were **retrained** with the correct normalization to match 6-channel models:
- Created: `scripts/train_rescue_3channel_CORRECT_NORM.py`
- Retrained: 20 models (10 @ 0ms + 10 @ 10000ms)
- New model directories: `*_CORRECT_NORM/`

### Why This Matters
Using different normalizations across models would make:
1. **Cross-model comparisons invalid** (comparing APP treatment vs RESCUE)
2. **Deployment inconsistent** (each model expecting different inputs)
3. **Results non-reproducible** (unclear which normalization was used)

---

## Deployment Requirements

**CRITICAL:** When deploying models for inference, you MUST use the EXACT SAME normalization pipeline:

```python
# 1. Load percentiles
with open('A2screen_combined/data/intensity_percentiles.json') as f:
    percentiles = json.load(f)

# 2. Apply to each channel
def normalize_channel(img_path, channel_name):
    img = np.array(Image.open(img_path), dtype=np.float32)

    p1 = percentiles[channel_name]['p1']
    p99 = percentiles[channel_name]['p99']

    # Stage 1: Clip
    img_clipped = np.clip(img, p1, p99)

    # Stage 2: Normalize to [0, 1]
    if p99 > p1:
        img_normalized = (img_clipped - p1) / (p99 - p1)
    else:
        img_normalized = img_clipped

    # Convert to tensor
    img_pil = Image.fromarray((img_normalized * 255).astype(np.uint8))
    img_pil = transforms.Resize((224, 224))(img_pil)
    img_tensor = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).unsqueeze(0)

    return img_tensor

# 3. Apply ImageNet normalization (after concatenating channels)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406] * num_timepoints,
    std=[0.229, 0.224, 0.225] * num_timepoints
)
```

---

## Verification Checklist

Before deploying any model, verify:

- [ ] Percentile values loaded from `intensity_percentiles.json`
- [ ] Three-stage pipeline applied: clip → [0,1] → ImageNet
- [ ] ImageNet normalization uses correct number of channels (3 or 6)
- [ ] Image resized to 224×224
- [ ] Data type is float32
- [ ] Values are in expected range after each stage

---

## References

**Training Scripts:**
- RESCUE: `scripts/train_rescue_3channel_CORRECT_NORM.py`
- 6-channel: `A2screen_combined/scripts/train_binary_6channel_FIXED_NORM.py`

**Percentile File:**
- `A2screen_combined/data/intensity_percentiles.json`

**Model Directories:**
- RESCUE (corrected): `A2screen_combined/models/rescue_*_replicates_CORRECT_NORM/`
- APP treatment: `A2screen_combined/models/app_treatment_replicates_CORRECT/`
- WT treatment: `A2screen_combined/models/wt_treatment_replicates_CORRECT/`
