# Production Scripts

This directory contains the 9 core production scripts for the phenotype detection pipeline.

## Overview

The pipeline focuses on detecting protein aggregation phenotypes in neurons using 3-channel imaging (RFP1 + Halo + Halo_masked). These scripts handle the complete workflow from raw HDF5 data to trained models.

---

## Data Preparation Scripts

### 1. `prepare_timecourse.py`
**Purpose:** Extract complete timecourse data (T0-T15) from HDF5 files for DMSO control wells

**Usage:**
```bash
python scripts/prepare_timecourse.py \
  --hdf5_dir /path/to/hdf5/dir \
  --output_dir /path/to/output \
  --wells A1 A2 B1 B2
```

**Output:**
- Images for all 16 timepoints (RFP1 and Halo channels)
- `metadata.csv` with neuron tracking across time

---

### 2. `prepare_drug_data.py`
**Purpose:** Process drug-treated wells, excluding bad wells

**Usage:**
```bash
python scripts/prepare_drug_data.py \
  --hdf5_dir /path/to/hdf5/dir \
  --output_dir /path/to/output \
  --bad_wells C1 C2 D3
```

**Notes:**
- Wrapper around `src/prepare_dataset.py`
- Processing time: ~4 hours
- Automatically excludes specified bad wells

---

### 3. `prepare_masked_from_hdf5.py`
**Purpose:** Generate masked Halo channel images from HDF5 files

**Description:** Creates the third channel (Halo_masked) by applying segmentation masks to Halo channel images. This masked channel helps the model focus on specific cellular regions.

---

### 4. `create_balanced_dataset.py`
**Purpose:** Balance datasets by downsampling majority class

**Usage:**
```bash
python scripts/create_balanced_dataset.py
```

**Method:**
- Identifies minority class size
- Downsamples majority class via stratified neuron sampling
- Preserves temporal information by sampling complete neurons

**Note:** Currently hardcoded paths - modify script for your dataset location

---

### 5. `create_reversion_datasets_v2.py`
**Purpose:** Create three balanced reversion model datasets

**Generates:**
1. **Model 1:** Non-aggregated vs Aggregated (both APP and WT)
2. **Model 2:** APP Non-aggregated vs APP Aggregated (APP-only)
3. **Model 3:** Everything else vs APP_T15_10000ms (most specific)

**Method:** Random downsampling to balance classes

**Note:** Currently hardcoded paths - modify for your dataset

---

### 6. `create_all_reversion_stratified.py`
**Purpose:** Create three stratified reversion datasets with equal subcategory representation

**Generates:** Same 3 models as v2, but uses **stratified sampling** instead of random sampling

**Method:**
- Identifies all subcategories (condition × timepoint × stimulation)
- Samples equal amounts from each subcategory
- Ensures fair representation across all experimental conditions

**Use Case:** Preferred when you want balanced representation across all subcategories, not just balanced class labels

---

## Training Scripts

### 7. `train_threechannel_improved.py`
**Purpose:** Train 3-channel classification models with class weighting

**Features:**
- Class weighting for imbalanced datasets
- Aggressive data augmentation
- Per-class metrics tracking
- Early stopping (patience=15)

**Usage:**
```bash
python scripts/train_threechannel_improved.py \
  --metadata /path/to/metadata.csv \
  --root_dir /path/to/images \
  --output_dir /path/to/output \
  --architecture resnet18 \
  --epochs 100
```

**Architectures:** `resnet18`, `resnet34`, `resnet50`

---

### 8. `train_reversion_3channel.py`
**Purpose:** Train reversion models (T0 vs T15) using 3-channel approach

**Task:** Detect pre-stimulation vs post-stimulation states in the same neurons

**Features:**
- **NO class weighting** (assumes pre-balanced data)
- 3 channels: RFP1 + Halo + Halo_masked
- Early stopping (patience=15)
- Best model selection by validation accuracy

**Usage:**
```bash
python scripts/train_reversion_3channel.py \
  --metadata /path/to/metadata.csv \
  --root_dir /path/to/images \
  --output_dir /path/to/output \
  --architecture resnet18 \
  --epochs 100
```

**Expected Input:** Metadata should contain T0 and T15 timepoints with balanced class distribution

---

## Deployment Scripts

### 9. `deploy_both_reversion_models.py`
**Purpose:** Deploy both reversion models for inference

**Description:** Loads trained reversion model checkpoints and applies them to new data for aggregation detection.

---

## Workflow

### Typical Pipeline Flow:

1. **Extract Data**
   - Use `prepare_timecourse.py` for control wells
   - Use `prepare_drug_data.py` for treatment wells

2. **Generate Masks**
   - Run `prepare_masked_from_hdf5.py` to create third channel

3. **Prepare Datasets**
   - Use `create_balanced_dataset.py` for general balancing
   - Use `create_reversion_datasets_v2.py` or `create_all_reversion_stratified.py` for reversion tasks

4. **Train Models**
   - Use `train_threechannel_improved.py` for general classification
   - Use `train_reversion_3channel.py` for reversion models

5. **Deploy**
   - Use `deploy_both_reversion_models.py` for inference

---

## Architecture Notes

### Three-Channel Approach
All training scripts use a **3-channel imaging approach**:
1. **RFP1:** Red fluorescent protein channel
2. **Halo:** Halo-tag channel (full image)
3. **Halo_masked:** Masked Halo channel (segmented regions)

This multi-channel approach provides richer information for aggregation detection compared to single-channel methods.

### Model Architectures
- Default: `resnet18` (fast, good performance)
- Available: `resnet34`, `resnet50` (more capacity, slower training)

---

## Key Differences: v2 vs Stratified Sampling

**create_reversion_datasets_v2.py:**
- Random sampling from each class
- Balances overall class sizes
- Faster, simpler approach

**create_all_reversion_stratified.py:**
- Equal sampling from each subcategory
- Ensures fair representation across conditions
- More robust when subcategories have different sizes
- **Recommended for production**

---

## Archived Scripts

For experimental and deprecated scripts, see `../archive/README.md`

---

*Last updated: 2025-11-11*
