# Proposed Fixes for Critical Bugs

This document provides specific code fixes for all critical and high-severity bugs identified in the audit.

---

## Fix #1: Stratification Alignment in ThreeChannelDataModule

**File:** `src/train_lightning_threechannel_improved.py`
**Lines:** 216-224

### Current (Buggy) Code:
```python
unique_wells = metadata['well'].unique()
well_labels = metadata.groupby('well')['class_label'].first()

wells_train_val, wells_test = train_test_split(
    unique_wells, test_size=0.15, random_state=42, stratify=well_labels
)

well_labels_train_val = well_labels[wells_train_val]
wells_train, wells_val = train_test_split(
    wells_train_val, test_size=0.15/(1-0.15), random_state=42,
    stratify=well_labels_train_val
)
```

### Fixed Code:
```python
unique_wells = metadata['well'].unique()
well_labels = metadata.groupby('well')['class_label'].first()

# Check if wells have mixed classes (reversion case)
wells_with_multiple_classes = metadata.groupby('well')['class_label'].nunique()
has_mixed_wells = (wells_with_multiple_classes > 1).any()

if has_mixed_wells:
    # Cannot stratify by class_label if wells contain multiple classes
    # Option 1: No stratification
    wells_train_val, wells_test = train_test_split(
        unique_wells, test_size=0.15, random_state=42
    )
    wells_train, wells_val = train_test_split(
        wells_train_val, test_size=0.15/(1-0.15), random_state=42
    )
    print("⚠️ Wells contain multiple classes - stratification by class disabled")

    # Option 2: Stratify by condition/genotype if available
    # if 'condition' in metadata.columns:
    #     well_conditions = metadata.groupby('well')['condition'].first()
    #     wells_train_val, wells_test = train_test_split(
    #         unique_wells, test_size=0.15, random_state=42,
    #         stratify=well_conditions.loc[unique_wells].values
    #     )
else:
    # Normal case: stratify by class_label
    # FIX: Use .loc to align indices
    wells_train_val, wells_test = train_test_split(
        unique_wells, test_size=0.15, random_state=42,
        stratify=well_labels.loc[unique_wells].values
    )

    well_labels_train_val = well_labels.loc[wells_train_val]
    wells_train, wells_val = train_test_split(
        wells_train_val, test_size=0.15/(1-0.15), random_state=42,
        stratify=well_labels_train_val.values
    )
```

---

## Fix #2: Remove ColorJitter from Augmentation

**File:** `src/train_lightning_threechannel_improved.py`
**Lines:** 34-42

### Current (Buggy) Code:
```python
if is_train:
    self.augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # BREAKS CHANNELS
    ])
```

### Fixed Code (Option 1 - Remove):
```python
if is_train:
    self.augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # ColorJitter removed - preserves channel relationships
    ])
```

### Fixed Code (Option 2 - Synchronized ColorJitter):
```python
import random

class SynchronizedColorJitter:
    """Apply same brightness/contrast to all channels"""
    def __init__(self, brightness=0.3, contrast=0.3):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        # Sample factors once and apply to all channels
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        # Apply same transform to all channels
        mean = img.mean(dim=[1, 2], keepdim=True)
        img = (img - mean) * contrast_factor + mean
        img = img + (brightness_factor - 1)
        return img.clamp(0, 1)

# In __init__:
if is_train:
    self.augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    self.color_jitter = SynchronizedColorJitter(brightness=0.3, contrast=0.3)
else:
    self.augment = None
    self.color_jitter = None

# In __getitem__:
if self.augment:
    img = self.augment(img)
    if self.color_jitter:
        img = self.color_jitter(img)
```

---

## Fix #3: Confusion Matrix Dynamic Printing

**File:** `src/train_lightning_threechannel_improved.py`
**Lines:** 163-168

### Current (Buggy) Code:
```python
cm = self.confusion_matrix.compute().cpu().numpy()
print("\nConfusion Matrix:")
print(f"{'':20s} {'Pred ' + self.class_names[0]:20s} {'Pred ' + self.class_names[1]:20s}")
for i, name in enumerate(self.class_names):
    print(f"{'True ' + name:20s} {cm[i, 0]:20d} {cm[i, 1]:20d}")
```

### Fixed Code:
```python
cm = self.confusion_matrix.compute().cpu().numpy()
print("\nConfusion Matrix:")

# Dynamic header
header = f"{'':20s}"
for name in self.class_names:
    header += f" {'Pred ' + name:>15s}"
print(header)

# Dynamic rows
for i, true_name in enumerate(self.class_names):
    row = f"{'True ' + true_name:20s}"
    for j in range(self.num_classes):
        row += f" {cm[i, j]:>15d}"
    print(row)
print("="*70)
```

---

## Fix #4: Dynamic num_classes in Reversion Training

**File:** `scripts/train_reversion_3channel.py`
**Line:** 35-40

### Current (Buggy) Code:
```python
model = ThreeChannelClassifier(
    num_classes=2,  # HARDCODED
    class_names=data_module.class_names,
    class_weights=None,
    learning_rate=1e-4,
    architecture=architecture
)
```

### Fixed Code:
```python
model = ThreeChannelClassifier(
    num_classes=data_module.num_classes,  # DYNAMIC
    class_names=data_module.class_names,
    class_weights=None,
    learning_rate=1e-4,
    architecture=architecture
)

# Add validation
assert data_module.num_classes == 2, \
    f"Reversion model expects 2 classes, got {data_module.num_classes}"
```

---

## Fix #5: Complete Deployment Script Rewrite

**File:** `scripts/deploy_and_score.py`

The deployment script has multiple critical bugs. Here's a complete fixed version:

```python
#!/usr/bin/env python3
"""
Deploy trained models on new data (FIXED VERSION)
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class DeploymentDataset(Dataset):
    """Full dataset for deployment (no train/test split)"""
    def __init__(self, metadata_df, root_dir, channel):
        self.metadata = metadata_df.reset_index(drop=True)  # Reset index for clean indexing
        self.root_dir = Path(root_dir)
        self.channel = channel

        # Standard transforms (no augmentation)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load image based on channel
        img_path = self.root_dir / row[f'{self.channel.lower()}_path']
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, idx  # Return index for alignment


def deploy_model(checkpoint_path, metadata_path, root_dir, channel,
                 model_name, output_file, model_type='AggregationClassifier'):
    """
    Deploy one model and output scores (FIXED VERSION)
    """
    print(f"\n{'='*70}")
    print(f"DEPLOYING: {model_name} - {channel}")
    print(f"{'='*70}")

    # Load model with architecture detection
    if model_type == 'ThreeChannel':
        from train_lightning_threechannel_improved import ThreeChannelClassifier
        model = ThreeChannelClassifier.load_from_checkpoint(checkpoint_path)
    else:
        from train_lightning import AggregationClassifier
        model = AggregationClassifier.load_from_checkpoint(checkpoint_path)

    model.eval()

    # Load FULL metadata
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded {len(metadata)} samples from metadata")

    # Create full deployment dataset (NO SPLITTING)
    deploy_dataset = DeploymentDataset(metadata, root_dir, channel)
    deploy_loader = DataLoader(
        deploy_dataset,
        batch_size=32,
        shuffle=False,  # Keep order!
        num_workers=4
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Run inference
    all_confidences = []
    all_predictions = []
    all_indices = []

    with torch.no_grad():
        for batch in tqdm(deploy_loader, desc=f"Deploying {channel}"):
            images, indices = batch
            images = images.to(device)

            # Get softmax probabilities
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_confidences.append(probs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_indices.extend(indices.tolist())

    # Combine results
    confidences = np.vstack(all_confidences)
    predictions = np.concatenate(all_predictions)

    # Verify alignment
    assert len(all_indices) == len(metadata), \
        f"Index mismatch: {len(all_indices)} predictions vs {len(metadata)} metadata rows"
    assert all_indices == list(range(len(metadata))), \
        "Indices are not sequential - data order was not preserved!"

    # Create results dataframe with proper alignment
    results = metadata.copy()
    results['confidence_class0'] = confidences[:, 0]
    results['confidence_class1'] = confidences[:, 1]
    results['predicted_class'] = predictions

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print(f"✅ Saved: {output_file}")
    print(f"   Samples: {len(results)}")

    return results


def aggregate_by_tile(scores_df, output_file):
    """Aggregate neuron scores to tile level (FIXED VERSION)"""
    agg = scores_df.groupby(['well', 'tile', 'timepoint',
                             'condition', 'stimulation']).agg({
        'neuron_id': 'count',
        'confidence_class0': ['mean', 'std'],
        'confidence_class1': ['mean', 'std'],
        'predicted_class': 'mean'  # Mean of binary = fraction
    }).reset_index()

    # Flatten column names properly
    agg.columns = [
        'well', 'tile', 'timepoint', 'condition', 'stimulation',
        'n_neurons',
        'mean_confidence_class0', 'std_confidence_class0',
        'mean_confidence_class1', 'std_confidence_class1',
        'fraction_predicted_class1'  # Renamed for clarity
    ]

    agg.to_csv(output_file, index=False)
    print(f"✅ Aggregated: {output_file}")

    return agg
```

---

## Fix #6: Add Return Value to train_threechannel_improved.py

**File:** `scripts/train_threechannel_improved.py`
**Line:** After line 57

### Add:
```python
return test_results[0]['test_acc']
```

---

## Fix #7: Consistent Callback Monitoring

**File:** `scripts/train_reversion_3channel.py`
**Lines:** 43-51

### Current Code:
```python
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    ...
)
early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min')
```

### Fixed Code:
```python
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='best-{epoch:02d}-{val_acc:.3f}',
    monitor='val_acc',
    mode='max',
    save_top_k=1
)

# Monitor same metric as checkpoint
early_stop = EarlyStopping(monitor='val_acc', patience=15, mode='max')
```

---

## Fix #8: Robust Path Handling

**File:** All scripts in `scripts/`
**Line:** Near top of file

### Current (Fragile) Code:
```python
sys.path.insert(0, 'src')
```

### Fixed Code:
```python
from pathlib import Path
import sys

# Get project root regardless of where script is run from
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'src'))
```

---

## Testing After Fixes

After applying fixes, run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/test_training_pipeline.py -v

# Run specific test
pytest tests/test_training_pipeline.py::test_stratification_alignment -v

# Run with coverage
pytest tests/test_training_pipeline.py --cov=src --cov-report=html
```

---

## Verification Checklist

After applying fixes, verify:

- [ ] Stratification works for both normal and reversion datasets
- [ ] Confusion matrix displays correctly for 2+ classes
- [ ] Deployment processes ALL samples (100%, not just 15%)
- [ ] Deployment metadata aligns correctly with predictions
- [ ] ColorJitter is removed or synchronized
- [ ] All scripts work from any directory
- [ ] Model num_classes matches data
- [ ] Callbacks monitor consistent metrics
- [ ] All tests pass

---

## Migration Guide

1. **Backup existing code:**
   ```bash
   git checkout -b backup-before-fixes
   git commit -am "Backup before bug fixes"
   git checkout main
   ```

2. **Apply fixes in order:**
   - Fix #1 (stratification) - CRITICAL
   - Fix #5 (deployment) - CRITICAL
   - Fix #2 (ColorJitter) - HIGH
   - Fix #3 (confusion matrix) - HIGH
   - Fix #4 (num_classes) - HIGH
   - Fixes #6-8 - MEDIUM

3. **Test after each fix:**
   ```bash
   # After each fix, run relevant tests
   pytest tests/test_training_pipeline.py::<test_name> -v
   ```

4. **Retrain models:**
   - After fixing stratification and ColorJitter, retrain all models
   - Old models were trained with incorrect augmentation
   - Results may differ after fixes

5. **Rerun deployment:**
   - After fixing deployment script, rerun all deployments
   - Old deployment results have 85% missing data and wrong metadata

---

## Questions?

For questions about these fixes, refer to:
- `BUG_REPORT.md` for detailed bug descriptions
- `tests/test_training_pipeline.py` for test cases demonstrating bugs
- Individual test functions for bug reproduction examples
