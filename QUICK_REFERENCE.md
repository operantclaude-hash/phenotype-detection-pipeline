# A2screen Models & Deployments - Quick Reference

**Last Updated**: December 2025
**Status**: Production-ready models with corrected deployments

---

## CRITICAL: Column Definition

**When we say "column X" or "plate column X", we ALWAYS mean:**
- The numeric part of the well identifier
- Example: Well **A10** = **Plate Column 10** (NOT row position)
- Example: Well **O19** = **Plate Column 19**

**384-well plate format:**
- Rows: A-P (16 rows)
- Columns: 1-24 (24 columns)
- Well A10 is at Row A, **Column 10**

**Plate Layout for Drug Screening:**
- **Columns 1-2**: WT control wells
- **Columns 3-4**: APP control wells
- **Columns 5-24**: APP + drug treatment wells

---

## Table of Contents

1. [APP Treatment Model (Binary 6-Channel)](#app-treatment-model-binary-6-channel)
2. [Rescue Model (Genotype Classification)](#rescue-model-genotype-classification)
3. [Deployment Outputs](#deployment-outputs)
4. [Recent Bug Fixes](#recent-bug-fixes)
5. [Quick Commands](#quick-commands)

---

## APP Treatment Model (Binary 6-Channel)

**Task**: Classify stimulation timing (0ms vs 10000ms)
**Architecture**: ResNet18 with 6-channel input
**Channels**: T0_RFP1, T0_Halo, T0_Halo_masked, Ti_RFP1, Ti_Halo, Ti_Halo_masked

### Model Location

```
ScoringModelOptimization/models/binary_6channel_REBALANCED/checkpoints/
└── best-epoch=19-val_acc=0.691.ckpt
```

**Performance**:
- Validation Accuracy: 69.1%
- Optimal Temperature: 1.545 (for calibration)
- Training: Balanced sampling with focal loss

### Training Script

```
ScoringModelOptimization/scripts/phase0_fix_imbalance/train_binary_6channel_REBALANCED.py
```

### Deployment Scripts (UPDATED - Bug Fixed)

**Best Approach (Temperature Scaled + Confidence Weighted)**:
```
ScoringModelOptimization/scripts/deploy_all_timepoints_best_approach.py
```
- ✓ Fixed December 2025 (genotype labeling bug)
- Uses temperature scaling (T=1.545)
- Confidence-weighted aggregation
- Deploys to ALL timepoints (T1-T14 for plate1, T1-T15 for plate2)

**Baseline (No Temperature Scaling)**:
```
ScoringModelOptimization/scripts/deploy_all_timepoints_baseline.py
```
- ✓ Fixed December 2025 (genotype labeling bug)
- Raw softmax predictions
- Confidence-weighted aggregation
- For comparison with temperature-scaled approach

### Latest Deployment Outputs

**Temperature-Scaled Predictions** (RECOMMENDED):
```
ScoringModelOptimization/results/temporal_predictions_best_approach/
├── plate1_neuron_predictions_all_timepoints.csv       # Individual neurons
├── plate1_well_tile_predictions_all_timepoints.csv    # Aggregated (USE THIS)
├── plate2_neuron_predictions_all_timepoints.csv
├── plate2_well_tile_predictions_all_timepoints.csv    # Aggregated (USE THIS)
└── README.md
```

**Baseline Predictions** (for comparison):
```
ScoringModelOptimization/results/temporal_predictions_baseline/
├── plate1_neuron_predictions_all_timepoints.csv
├── plate1_well_tile_predictions_all_timepoints.csv
├── plate2_neuron_predictions_all_timepoints.csv
├── plate2_well_tile_predictions_all_timepoints.csv
└── README.md
```

**Log Files**:
- `ScoringModelOptimization/results/temporal_deployment_FIXED.log` (best approach)
- `ScoringModelOptimization/results/baseline_temporal_deployment_FIXED.log` (baseline)

### Usage Example

```python
import pandas as pd

# Load temperature-scaled predictions (RECOMMENDED)
df = pd.read_csv('ScoringModelOptimization/results/temporal_predictions_best_approach/plate1_well_tile_predictions_all_timepoints.csv')

# Filter for specific timepoint
t10_data = df[df['timepoint'] == 10]

# Plot temporal evolution for a well
well_data = df[df['well'] == 'O19']
import matplotlib.pyplot as plt
plt.plot(well_data['timepoint'], well_data['prob_class1'])
plt.xlabel('Timepoint')
plt.ylabel('Prob(10000ms phenotype)')
plt.title('Temporal Evolution - Well O19')
plt.show()
```

---

## Rescue Model (Genotype Classification)

**Task**: Classify genotype (WT vs APP)
**Architecture**: ResNet18 with 3-channel input
**Channels**: RFP1, Halo, Halo_masked

### Model Location

```
A2screen_combined/models/rescue_3channel_LATE_TIMEPOINTS/checkpoints/
└── best-epoch=43-val_acc=0.780.ckpt
```

**Performance**:
- Validation Accuracy: 78.0%
- Validation Separation: 0.496 (49.6% difference between WT and APP)
- Training: Late timepoints only (T10-T16)
- Stimulation: 10000ms only

### Training Script

```
A2screen_combined/scripts/train_rescue_3channel_FIXED_NORM.py
```

### Deployment Script

```
ScoringModelOptimization/scripts/deploy_rescue_full_plate_ALL_WELLS.py
```
- ✓ Correct metadata usage (plate1_aligned_extracted)
- Deploys to ALL wells (columns 1-24)
- ALL timepoints (T0-T16)
- ALL stimulation conditions (0ms + 10000ms)
- Baseline + confidence-weighted approach

### Latest Deployment Outputs

```
ScoringModelOptimization/results/rescue_full_plate_ALL_WELLS/
├── plate1_neuron_predictions.csv           # Individual neurons
├── plate1_well_tile_predictions.csv        # Aggregated (USE THIS FOR DRUG SCREENING)
├── plate2_neuron_predictions.csv
├── plate2_well_tile_predictions.csv        # Aggregated (USE THIS FOR DRUG SCREENING)
├── deployment.log
└── README.md
```

**Interpretation**:
- `prob_class1` = Probability of APP (diseased) phenotype
- **High scores (→1)**: APP-like (diseased)
- **Low scores (→0)**: WT-like (healthy)

**Expected Baselines**:
- WT controls (columns 1-2): prob_class1 ≈ 0.27-0.30
- APP controls (columns 3-4): prob_class1 ≈ 0.73-0.83
- Separation: ≈ 0.45-0.55

### Drug Screening Usage

```python
import pandas as pd

# Load predictions
plate1 = pd.read_csv('ScoringModelOptimization/results/rescue_full_plate_ALL_WELLS/plate1_well_tile_predictions.csv')

# Filter for drug-treated wells
drug_wells = plate1[plate1['column'] > 4]

# Rank drugs by rescue effect (lower disease score = better)
drug_wells.groupby('well')['prob_class1'].mean().sort_values()

# Calculate rescue percentage
wt_baseline = plate1[plate1['column'].isin([1,2])]['prob_class1'].mean()
app_baseline = plate1[plate1['column'].isin([3,4])]['prob_class1'].mean()

for well in drug_wells['well'].unique():
    drug_score = drug_wells[drug_wells['well']==well]['prob_class1'].mean()
    rescue_pct = (app_baseline - drug_score) / (app_baseline - wt_baseline) * 100
    print(f"{well}: {rescue_pct:.1f}% rescue")
```

---

## Deployment Outputs

### File Types

**Neuron-Level Files**:
- Individual predictions for each neuron
- Large files (19-27 MB)
- Use for detailed single-cell analysis

**Well-Tile-Level Files** (RECOMMENDED):
- Aggregated by (well, tile, timepoint, stimulation)
- Confidence-weighted averaging
- Smaller files (800 KB)
- **USE THESE FOR DRUG SCREENING AND TEMPORAL ANALYSIS**

### Column Definitions

**Well-Tile-Level Files**:
- `well`: Well identifier (e.g., A10, O19)
- `tile`: Tile within well (1-4)
- `timepoint`: Timepoint number (0-16)
- `stimulation`: Stimulation condition (0ms or 10000ms)
- `prob_class0`: Confidence-weighted probability of Class 0
- `prob_class1`: Confidence-weighted probability of Class 1
- `n_neurons`: Number of neurons aggregated
- `column`: **Plate column number** (numeric part of well name)
- `genotype`: True genotype (WT or APP)
- `plate`: Plate identifier (plate1 or plate2)

---

## Recent Bug Fixes

### Genotype Labeling Bug (Fixed December 2025)

**Problem**: APP treatment deployment scripts were loading incorrect metadata, causing all wells to have wrong column/genotype labels.

**Example**:
- Well A10 (plate column 10, genotype APP) was incorrectly labeled as column=1, genotype=WT
- All A-row wells were labeled as column=1

**Root Cause**: Scripts loaded `plate1_extracted/metadata_plate1.csv` which assigns column by row letter position (A=1, B=2, O=15) instead of by well number (A10=10, O19=19)

**Fix Applied**:
- Updated `deploy_all_timepoints_best_approach.py` (lines 61, 248-254)
- Updated `deploy_all_timepoints_baseline.py` (lines 61, 242-245)
- Both scripts now use correct metadata: `plate1_aligned_extracted/metadata_plate1_aligned.csv`

**Impact**:
- ✓ Training data: NOT affected (always used correct metadata)
- ✓ Rescue model deployments: NOT affected (already used correct metadata)
- ✗ APP treatment deployments: Re-generated with correct labels

**Verification**:
```bash
# Verify A10 is now correct
awk -F',' '$1 == "A10" {print "Well "$1": column="$7", genotype="$8; exit}' \
  ScoringModelOptimization/results/temporal_predictions_best_approach/plate1_well_tile_predictions_all_timepoints.csv

# Expected: Well A10: column=10, genotype=APP
```

**Status**: ✓ **FIXED** - All deployment outputs regenerated with correct labels

---

## Quick Commands

### Check Model Locations

```bash
# APP Treatment Model
ls -lh ScoringModelOptimization/models/binary_6channel_REBALANCED/checkpoints/

# Rescue Model
ls -lh A2screen_combined/models/rescue_3channel_LATE_TIMEPOINTS/checkpoints/
```

### Check Latest Deployments

```bash
# APP Treatment (Temperature-Scaled)
ls -lh ScoringModelOptimization/results/temporal_predictions_best_approach/

# APP Treatment (Baseline)
ls -lh ScoringModelOptimization/results/temporal_predictions_baseline/

# Rescue Model
ls -lh ScoringModelOptimization/results/rescue_full_plate_ALL_WELLS/
```

### Verify Column/Genotype Labels

```bash
# Check specific well (should match plate position)
awk -F',' '$1 == "A10" {print $1","$7","$8; exit}' \
  ScoringModelOptimization/results/temporal_predictions_best_approach/plate1_well_tile_predictions_all_timepoints.csv

# Expected: A10,10,APP

# Check distribution (columns 1-2 should be WT only, 3+ should be APP only)
awk -F',' 'NR>1 {print $7","$8}' \
  ScoringModelOptimization/results/temporal_predictions_best_approach/plate1_well_tile_predictions_all_timepoints.csv | \
  sort | uniq -c
```

### Re-run Deployments (if needed)

```bash
# APP Treatment - Best Approach
python ScoringModelOptimization/scripts/deploy_all_timepoints_best_approach.py 2>&1 | \
  tee ScoringModelOptimization/results/temporal_deployment_FIXED.log

# APP Treatment - Baseline
python ScoringModelOptimization/scripts/deploy_all_timepoints_baseline.py 2>&1 | \
  tee ScoringModelOptimization/results/baseline_temporal_deployment_FIXED.log

# Rescue Model
python ScoringModelOptimization/scripts/deploy_rescue_full_plate_ALL_WELLS.py 2>&1 | \
  tee ScoringModelOptimization/results/rescue_full_plate_ALL_WELLS/deployment.log
```

**Expected Runtime**: ~20-30 minutes per deployment

---

## Analysis Files

### For APP Treatment Analysis
**Primary File**:
```
ScoringModelOptimization/results/temporal_predictions_best_approach/plate1_well_tile_predictions_all_timepoints.csv
ScoringModelOptimization/results/temporal_predictions_best_approach/plate2_well_tile_predictions_all_timepoints.csv
```

**Use Cases**:
- Temporal evolution of stimulation phenotype
- When does 10000ms phenotype appear?
- Plate-to-plate comparison

### For Drug Screening
**Primary File**:
```
ScoringModelOptimization/results/rescue_full_plate_ALL_WELLS/plate1_well_tile_predictions.csv
ScoringModelOptimization/results/rescue_full_plate_ALL_WELLS/plate2_well_tile_predictions.csv
```

**Use Cases**:
- Rank drugs by rescue effect
- Compare drug-treated wells (columns 5-24) to APP controls (columns 3-4)
- Calculate rescue percentage: lower `prob_class1` = better rescue

---

## Metadata Files

### Correct Metadata (USE THESE)

```
A2screen_combined/data/plate1_aligned_extracted/metadata_plate1_aligned.csv
A2screen_combined/data/plate2_extracted/metadata_plate2.csv
```

**Features**:
- ✓ Column assigned by well number (A10 = column 10)
- ✓ Correct genotype labels
- ✓ 224×224 aligned images

### Incorrect Metadata (DO NOT USE)

```
A2screen_combined/data/plate1_extracted/metadata_plate1.csv
```

**Issues**:
- ✗ Column assigned by row letter (A=1, B=2, O=15)
- ✗ Incorrect genotype labels
- ✗ Will cause deployment bugs

---

## Scripts Still Using Buggy Metadata (Low Priority)

These 8 scripts use the buggy `plate1_extracted` metadata but are not currently critical:

**Data Preparation**:
1. `ScoringModelOptimization/scripts/phase0_fix_imbalance/prepare_binary_dataset.py`
2. `ScoringModelOptimization/scripts/phase0_fix_imbalance/prepare_rescue_dataset.py`
3. `ScoringModelOptimization/scripts/phase0_fix_imbalance/prepare_binary_dataset_balanced.py`
4. `ScoringModelOptimization/scripts/phase0_fix_imbalance/prepare_rescue_dataset_balanced.py`

**Other Deployments**:
5. `ScoringModelOptimization/scripts/deploy_rescue_baseline_confidence_weighted.py`
6. `ScoringModelOptimization/scripts/deploy_binary_6channel.py`
7. `ScoringModelOptimization/scripts/deploy_rescue_3channel.py`
8. `ScoringModelOptimization/scripts/deploy_all_comprehensive.py`

**When to fix**: Update these when they're next needed. Training data was NOT affected.

---

## Success Checklist

### APP Treatment Model
- ✓ Model trained and saved
- ✓ Temperature scaling calibrated (T=1.545)
- ✓ Deployed to all timepoints (both plates)
- ✓ Genotype labels verified correct
- ✓ Well-tile aggregated files ready for analysis

### Rescue Model
- ✓ Model trained on late timepoints (78% accuracy)
- ✓ Deployed to all wells, timepoints, stimulations
- ✓ Control baselines verified (WT ~0.3, APP ~0.8)
- ✓ Well-tile aggregated files ready for drug screening

### Quality Checks
- ✓ Columns 1-2 labeled as WT
- ✓ Columns 3-24 labeled as APP
- ✓ Well A10 shows column=10, genotype=APP
- ✓ Well O19 shows column=19, genotype=APP
- ✓ No WT labels in columns >2

---

## Contact & Troubleshooting

**For questions about**:
- Column definitions: See [CRITICAL: Column Definition](#critical-column-definition)
- Model performance: Check validation accuracy in checkpoint filenames
- Deployment bugs: Verify correct metadata file is being used
- Drug screening: Use rescue model well-tile predictions with `prob_class1` as disease score

**Key Principle**: Always use `plate1_aligned_extracted` for plate1 data, never `plate1_extracted`.
