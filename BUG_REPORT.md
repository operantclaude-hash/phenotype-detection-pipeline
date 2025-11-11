# Bug Audit Report: Training and Deployment Scripts

**Date:** 2025-11-11
**Audited by:** Claude Code Agent
**Scope:** Training scripts (train_reversion_3channel.py, train_threechannel_improved.py) and deployment scripts (deploy_and_score.py)

---

## Executive Summary

This audit identified **11 critical and high-severity bugs** in the phenotype detection pipeline, primarily affecting:
1. **Image and label processing** - Data corruption from stratification errors
2. **Model deployment** - Complete failure of deployment script with silent data corruption
3. **Multi-class support** - Hardcoded assumptions breaking generalization

**Severity Breakdown:**
- 🔴 **CRITICAL**: 5 bugs (data corruption, crashes, wrong results)
- 🟠 **HIGH**: 3 bugs (scientific validity, compatibility issues)
- 🟡 **MEDIUM**: 3 bugs (suboptimal design, inconsistencies)

---

## 🔴 CRITICAL BUGS

### BUG #1: Stratification Misalignment in Data Splitting
**File:** `src/train_lightning_threechannel_improved.py:216-223`
**Severity:** CRITICAL
**Status:** ❌ UNFIXED

**Description:**
The stratification labels are misaligned with the wells array. `unique_wells` is a numpy array in appearance order, but `well_labels` is a pandas Series with potentially different index order. When `train_test_split` converts the Series to array using `.values`, it uses the Series' index order, NOT the order of `unique_wells`.

**Example of failure:**
```python
# If unique_wells = ['C3', 'A1', 'B2']  (appearance order)
# And well_labels = Series(['classB', 'classA', 'classA'], index=['A1', 'B2', 'C3'])
# Then stratify receives ['classB', 'classA', 'classA']
# But this corresponds to wells ['A1', 'B2', 'C3'], NOT ['C3', 'A1', 'B2']
# Result: well 'C3' gets labeled as 'classB' for stratification instead of 'classA'!
```

**Impact:**
- Stratification applies to wrong wells, breaking balanced splits
- Can lead to severe class imbalance in train/val/test sets
- Compromises all model evaluation metrics
- Affects generalization and reproducibility

**Fix:**
```python
# Line 217: Fix first split
wells_train_val, wells_test = train_test_split(
    unique_wells, test_size=0.15, random_state=42,
    stratify=well_labels.loc[unique_wells].values
)

# Line 221-223: Fix second split
well_labels_train_val = well_labels.loc[wells_train_val]
wells_train, wells_val = train_test_split(
    wells_train_val, test_size=0.15/(1-0.15), random_state=42,
    stratify=well_labels_train_val.values
)
```

---

### BUG #2: Reversion Dataset Stratification Crashes
**File:** `src/train_lightning_threechannel_improved.py:214-217`
**Severity:** CRITICAL
**Status:** ❌ UNFIXED

**Description:**
For reversion tasks (T0 vs T15), each well contains samples from BOTH timepoints. The code uses `.groupby('well')['class_label'].first()` which assigns only the first label to each well. When metadata is ordered such that T0 appears first in all wells, ALL wells get labeled as T0, causing stratification to fail.

**Impact:**
- Script crashes immediately with ValueError: "The least populated class in y has only 1 member"
- Makes reversion training completely non-functional
- Blocks all T0 vs T15 experiments

**Fix:**
```python
# Option 1: Remove stratification for mixed-class wells
wells_train_val, wells_test = train_test_split(
    unique_wells, test_size=0.15, random_state=42
)

# Option 2: Stratify by well condition/genotype instead
if 'condition' in metadata.columns:
    well_conditions = metadata.groupby('well')['condition'].first()
    wells_train_val, wells_test = train_test_split(
        unique_wells, test_size=0.15, random_state=42,
        stratify=well_conditions.loc[unique_wells].values
    )
```

---

### BUG #3: Deployment Index Tracking Error
**File:** `scripts/deploy_and_score.py:68-71`
**Severity:** CRITICAL
**Status:** ❌ UNFIXED

**Description:**
The index calculation uses a hardcoded value of 32, but uses the actual batch size for the range:
```python
batch_size = images.shape[0]
start_idx = batch_idx * 32  # ← HARDCODED!
all_indices.extend(range(start_idx, start_idx + batch_size))
```

**Example failure:**
- Dataset: 100 samples, batch_size=32
- Batches: [0-31], [32-63], [64-95], [96-99]
- Batch 3 calculates: start_idx = 3 * 32 = 96, range(96, 100) ✓
- BUT if an earlier batch was smaller, indices will be wrong
- Last batch: tries to access indices 96-127 which don't exist!

**Impact:**
- IndexError or data misalignment
- Wrong predictions assigned to wrong neurons
- Deployment fails on datasets not divisible by 32

**Fix:**
```python
# Track actual cumulative index
current_idx = 0
for batch_idx, batch in enumerate(tqdm(data_module.test_dataloader())):
    images, labels = batch
    images = images.to(device)
    batch_size = images.shape[0]
    all_indices.extend(range(current_idx, current_idx + batch_size))
    current_idx += batch_size
```

---

### BUG #4: Deployment Uses Test Split Only
**File:** `scripts/deploy_and_score.py:55`
**Severity:** CRITICAL
**Status:** ❌ UNFIXED

**Description:**
The deployment script uses `data_module.test_dataloader()` which only returns the test split (15% of data by default). It should deploy on ALL data, not just the test subset.

**Impact:**
- **Only 15% of data deployed** - missing 85% of samples
- Downstream analysis has massive missing data
- Misleading results - appears to run successfully but silently drops most data

**Fix:**
```python
# Create full dataset loader instead of using test split
full_dataset = create_full_deployment_dataset(
    metadata_path=metadata_path,
    root_dir=root_dir,
    channel=channel
)
full_dataloader = DataLoader(
    full_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Use full_dataloader instead of data_module.test_dataloader()
for batch_idx, batch in enumerate(tqdm(full_dataloader)):
    ...
```

---

### BUG #5: Deployment Metadata Indexing Mismatch
**File:** `scripts/deploy_and_score.py:47, 78`
**Severity:** CRITICAL (SILENT DATA CORRUPTION)
**Status:** ❌ UNFIXED

**Description:**
The script loads full metadata but indexes it with indices from test dataloader:
```python
metadata = pd.read_csv(metadata_path)  # Full metadata: 1000 rows
# ... iterate through test dataloader (150 samples)
results = metadata.iloc[all_indices].copy()  # all_indices = [0, 1, 2, ..., 149]
```

The indices [0-149] point to the FIRST 150 rows of metadata, NOT the actual test samples (which are scattered throughout: rows 23, 45, 67, 89, ...).

**Impact:**
- **Complete data corruption** - predictions assigned to wrong neurons
- Wrong neuron IDs, wells, timepoints
- **Silent failure** - no error raised, results look valid but are completely incorrect
- All downstream analysis is invalid

**Fix:**
```python
# Option 1: Get actual test indices from data module
test_indices = data_module.test_dataset.metadata.index
results = metadata.loc[test_indices].copy()

# Option 2 (CORRECT): Use full dataset and natural indexing
metadata = pd.read_csv(metadata_path)
full_dataset = create_deployment_dataset(metadata, ...)
# Iterate and indices naturally align with metadata rows
```

---

## 🟠 HIGH SEVERITY BUGS

### BUG #6: ColorJitter Breaks Multi-Channel Relationships
**File:** `src/train_lightning_threechannel_improved.py:41`
**Severity:** HIGH (Scientific Validity)
**Status:** ❌ UNFIXED

**Description:**
`ColorJitter` applies random brightness/contrast independently to each channel. For microscopy data where RFP1, Halo, and Halo_masked represent different fluorescence signals, the relative intensities between channels are biologically meaningful.

**Example:**
```python
# Original: RFP1=100, Halo=50, Halo_masked=45 (consistent relationship)
# After ColorJitter: RFP1=130, Halo=35, Halo_masked=60 (relationship destroyed!)
```

**Impact:**
- Destroys biological relationship between fluorescence channels
- Model learns to ignore meaningful cross-channel features
- Scientifically invalid augmentation for multi-channel microscopy

**Fix:**
```python
# Option 1: Remove ColorJitter entirely
if is_train:
    self.augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # ColorJitter removed
    ])

# Option 2: Custom synchronized brightness/contrast
class SyncColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        # Apply SAME adjustment to ALL channels
        brightness_factor = random.uniform(1-self.brightness, 1+self.brightness)
        contrast_factor = random.uniform(1-self.contrast, 1+self.contrast)
        img = img * contrast_factor + (brightness_factor - 1)
        return img.clamp(0, 1)
```

---

### BUG #7: Hardcoded num_classes in Reversion Script
**File:** `scripts/train_reversion_3channel.py:36`
**Severity:** HIGH
**Status:** ❌ UNFIXED

**Description:**
Hard-codes `num_classes=2` instead of using `data_module.num_classes`:
```python
model = ThreeChannelClassifier(
    num_classes=2,  # Hard-coded!
    class_names=data_module.class_names,  # Could have ≠2 classes
    ...
)
```

**Impact:**
- Crashes if wrong metadata used (e.g., 3+ classes)
- Inconsistent with train_threechannel_improved.py
- Makes script fragile and less reusable

**Fix:**
```python
model = ThreeChannelClassifier(
    num_classes=data_module.num_classes,  # Dynamic
    class_names=data_module.class_names,
    ...
)
```

---

### BUG #8: Confusion Matrix Hardcoded for 2 Classes
**File:** `src/train_lightning_threechannel_improved.py:165-167`
**Severity:** HIGH
**Status:** ❌ UNFIXED

**Description:**
Confusion matrix printing assumes exactly 2 classes:
```python
print(f"{'':20s} {'Pred ' + self.class_names[0]:20s} {'Pred ' + self.class_names[1]:20s}")
for i, name in enumerate(self.class_names):
    print(f"{'True ' + name:20s} {cm[i, 0]:20d} {cm[i, 1]:20d}")
```

**Impact:**
- Crashes with 1 class (IndexError)
- For 3+ classes, only shows first 2 columns of confusion matrix
- Incomplete and misleading results for multi-class problems

**Fix:**
```python
cm = self.confusion_matrix.compute().cpu().numpy()
print("\nConfusion Matrix:")

# Dynamic header
header = f"{'':20s}"
for name in self.class_names:
    header += f" {'Pred ' + name:20s}"
print(header)

# Dynamic rows
for i, name in enumerate(self.class_names):
    row = f"{'True ' + name:20s}"
    for j in range(len(self.class_names)):
        row += f" {cm[i, j]:20d}"
    print(row)
```

---

## 🟡 MEDIUM SEVERITY ISSUES

### ISSUE #9: ImageNet Normalization on Microscopy Data
**File:** `src/train_lightning_threechannel_improved.py:28-32`
**Severity:** MEDIUM
**Status:** ❌ UNFIXED

**Description:**
Uses ImageNet normalization statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) on microscopy fluorescence images with completely different intensity distributions.

**Impact:**
- Suboptimal feature learning
- May limit model performance vs data-specific normalization
- Microscopy images have lower intensities and different contrast patterns

**Recommendation:**
Calculate normalization from actual training data:
```python
# Run once on training set
all_images = [train_dataset[i][0] for i in range(len(train_dataset))]
all_images = torch.stack(all_images)
mean = all_images.mean(dim=[0, 2, 3])
std = all_images.std(dim=[0, 2, 3])
print(f"Data statistics - Mean: {mean}, Std: {std}")
```

---

### ISSUE #10: Inconsistent Callback Monitoring
**File:** `scripts/train_reversion_3channel.py:43-51`
**Severity:** MEDIUM
**Status:** ❌ UNFIXED

**Description:**
`ModelCheckpoint` monitors `val_acc` (maximize) while `EarlyStopping` monitors `val_loss` (minimize), leading to conflicting behavior.

**Impact:**
- May stop too early or too late
- Best checkpoint may not correspond to when training stops
- Suboptimal training efficiency

**Fix:**
```python
# Both monitor same metric
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    ...
)
early_stop = EarlyStopping(monitor='val_acc', patience=15, mode='max')
```

---

### ISSUE #11: Missing Return Value
**File:** `scripts/train_threechannel_improved.py:57`
**Severity:** MEDIUM
**Status:** ❌ UNFIXED

**Description:**
The `train()` function doesn't return test accuracy, unlike `train_reversion()` which does.

**Impact:**
- Cannot use function programmatically
- Inconsistent API
- Harder to integrate into automated pipelines

**Fix:**
```python
# Add after line 57
return test_results[0]['test_acc']
```

---

## Additional Observations

### Relative Path Issues
**All training scripts:** Use `sys.path.insert(0, 'src')` which only works from project root.

**Fix:**
```python
from pathlib import Path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'src'))
```

### Deployment Architecture Mismatch
**File:** `scripts/deploy_and_score.py`
Uses `AggregationClassifier` (single-channel) but cannot deploy `ThreeChannelClassifier` models, causing runtime errors.

---

## Priority Fix Order

1. **BUG #5** (Deployment metadata indexing) - Silent data corruption
2. **BUG #4** (Deployment test split only) - Missing 85% of data
3. **BUG #3** (Deployment index tracking) - IndexError
4. **BUG #1** (Stratification misalignment) - Wrong splits
5. **BUG #2** (Reversion stratification crash) - Blocks reversion training
6. **BUG #6** (ColorJitter) - Scientific validity
7. **BUG #7** (Hardcoded num_classes) - Robustness
8. **BUG #8** (Confusion matrix) - Multi-class support
9. **Issues #9-11** - Optimization and consistency

---

## Testing Recommendations

See `tests/test_training_pipeline.py` for comprehensive unit tests covering:
- Data loading and preprocessing
- Label alignment
- Stratification correctness
- Index tracking
- Multi-class support
- Deployment pipeline

---

## Conclusion

The audit reveals **critical data integrity issues** that compromise both training and deployment:
1. Training scripts have stratification bugs causing imbalanced splits
2. Deployment script has catastrophic bugs causing data corruption
3. Scientific validity concerns with ColorJitter on multi-channel data

**All critical bugs should be fixed before using this pipeline for any experiments.**
