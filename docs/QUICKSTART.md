# Quick Start Guide - 6-Class Phenotype Detection

**Complete walkthrough with copy-paste commands for the entire pipeline.**

---

## 📋 Prerequisites Checklist

Before starting:
- [ ] Python 3.8+ installed
- [ ] CUDA-capable GPU (optional but recommended)
- [ ] HDF5 neuron data organized in directories
- [ ] ~50GB free disk space
- [ ] List of bad wells identified

---

## 🚀 Step-by-Step Workflow

### Step 1: Install Dependencies (5 minutes)

```bash
# Create virtual environment (recommended)
conda create -n phenotype python=3.10
conda activate phenotype

# Install PyTorch (check https://pytorch.org for your CUDA version)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install pytorch-lightning pandas numpy matplotlib seaborn scikit-learn tqdm h5py Pillow
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Step 2: Prepare Your Data (10 minutes)

#### 2.1 Identify Bad Wells

Check your experimental log or quality control data for wells to exclude:
```bash
# Example bad wells list
BAD_WELLS="A1 A2 A3 A5 A12 B14 B15"
```

#### 2.2 Run Dataset Preparation

```bash
python scripts/1_prepare_dataset.py \
    --hdf5_dir "/path/to/your/hdf5_neurons" \
    --output_dir ./experiment1_dataset \
    --bad_wells $BAD_WELLS
```

**What this does:**
- Scans all HDF5 files
- Filters wells and timepoints
- Creates 6-class labels
- Extracts T0 and T16 images
- Saves metadata CSV

**Expected output:**
```
Processing wells: 100%|██████████| 96/96
Valid neurons: 9694
T0 samples: 9694
T16 samples: 9694
Total: 19388 image pairs

Class distribution:
Control_T0              4103
APPV717I_T0             5591
Control_0ms_T16         1653
Control_10000ms_T16     2450
APPV717I_0ms_T16        2402
APPV717I_10000ms_T16    3189

✅ Dataset saved to: ./experiment1_dataset
```

#### 2.3 Add class_label Column

```bash
python << 'EOF'
import pandas as pd
df = pd.read_csv('experiment1_dataset/metadata.csv')
df['class_label'] = df['label_6class']
df.to_csv('experiment1_dataset/metadata.csv', index=False)
print("✅ Added class_label column")
EOF
```

---

### Step 3: Train Models (15-20 minutes)

```bash
python scripts/2_train_models.py \
    --metadata experiment1_dataset/metadata.csv \
    --root_dir experiment1_dataset/ \
    --output_dir ./results \
    --tasks 6class \
    --channels RFP1 Halo \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num_workers 4
```

**Training parameters:**
- `--batch_size 32`: Adjust based on GPU memory (16/32/64)
- `--epochs 100`: Max epochs (early stopping at 15)
- `--lr 0.001`: Learning rate
- `--num_workers 4`: CPU threads for data loading

**GPU recommendations:**
- 8GB VRAM: batch_size=16
- 16GB VRAM: batch_size=32
- 24GB+ VRAM: batch_size=64

**Monitor training (optional):**
```bash
# In another terminal
tensorboard --logdir results/
# Open http://localhost:6006
```

**Expected output:**
```
Training RFP1...
Epoch 0: train_loss=1.79, val_acc=0.23
Epoch 10: train_loss=1.45, val_acc=0.38
Epoch 30: train_loss=1.15, val_acc=0.52
Epoch 45: train_loss=0.98, val_acc=0.55
Early stopping triggered!
✅ Best RFP1 accuracy: 0.548

Training Halo...
Epoch 0: train_loss=1.82, val_acc=0.21
Epoch 25: train_loss=1.28, val_acc=0.45
Epoch 50: train_loss=1.05, val_acc=0.50
✅ Best Halo accuracy: 0.505
```

**Outputs:**
- `results/RFP1/6class/checkpoints/` - Model weights
- `results/Halo/6class/checkpoints/` - Model weights
- `results/RFP1/6class/logs/` - TensorBoard logs

---

### Step 4: Evaluate Models (2 minutes)

#### 4.1 Evaluate RFP1

```bash
# Find best checkpoint
BEST_RFP1=$(ls results/RFP1/6class/checkpoints/best-*.ckpt | sort -t'=' -k3 -r | head -1)
echo "Using checkpoint: $BEST_RFP1"

python scripts/3_evaluate_models.py \
    --checkpoint "$BEST_RFP1" \
    --metadata experiment1_dataset/metadata.csv \
    --root_dir experiment1_dataset/ \
    --channel RFP1 \
    --task 6class \
    --output_dir ./evaluation/RFP1
```

#### 4.2 Evaluate Halo

```bash
# Find best checkpoint
BEST_HALO=$(ls results/Halo/6class/checkpoints/best-*.ckpt | sort -t'=' -k3 -r | head -1)
echo "Using checkpoint: $BEST_HALO"

python scripts/3_evaluate_models.py \
    --checkpoint "$BEST_HALO" \
    --metadata experiment1_dataset/metadata.csv \
    --root_dir experiment1_dataset/ \
    --channel Halo \
    --task 6class \
    --output_dir ./evaluation/Halo
```

**Expected output:**
```
Loading checkpoint...
Evaluating: 100%|████████| 91/91
Test accuracy: 0.548

Generating confusion matrix...
Saved: confusion_matrix.png
Saved: confusion_matrix_normalized.png
Saved: confusion_matrix.npy
Saved: metrics.json
```

**Check outputs:**
```bash
ls evaluation/RFP1/
# confusion_matrix.png
# confusion_matrix_normalized.png
# confusion_matrix.npy
# metrics.json

ls evaluation/Halo/
# (same files)
```

---

### Step 5: Comprehensive Analysis (< 1 minute)

**If you have 8-class results for comparison:**
```bash
python scripts/4_analyze_results.py \
    --rfp1_8class results_8class/RFP1/full/evaluation \
    --rfp1_6class evaluation/RFP1 \
    --halo_8class results_8class/Halo/full/evaluation \
    --halo_6class evaluation/Halo \
    --metadata experiment1_dataset/metadata.csv \
    --output_dir ./final_analysis
```

**If you only have 6-class results:**
```bash
# Skip 8-class comparison, just analyze 6-class
python << 'EOF'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load confusion matrices
rfp1_cm = np.load('evaluation/RFP1/confusion_matrix.npy')
halo_cm = np.load('evaluation/Halo/confusion_matrix.npy')

class_names = [
    'Control_T0', 'APPV717I_T0',
    'Control_0ms_T16', 'Control_10000ms_T16',
    'APPV717I_0ms_T16', 'APPV717I_10000ms_T16'
]

# Per-class accuracy
rfp1_acc = rfp1_cm.diagonal() / rfp1_cm.sum(axis=1)
halo_acc = halo_cm.diagonal() / halo_cm.sum(axis=1)

print("="*60)
print("PER-CLASS ACCURACY")
print("="*60)
print(f"{'Class':<30} {'RFP1':<10} {'Halo':<10}")
print("-"*60)
for i, name in enumerate(class_names):
    print(f"{name:<30} {rfp1_acc[i]:.3f}      {halo_acc[i]:.3f}")

print("\n" + "="*60)
print("TEMPORAL ANALYSIS")
print("="*60)
print(f"T0 average:  RFP1={rfp1_acc[:2].mean():.3f}, Halo={halo_acc[:2].mean():.3f}")
print(f"T16 average: RFP1={rfp1_acc[2:].mean():.3f}, Halo={halo_acc[2:].mean():.3f}")
print(f"Improvement: RFP1={rfp1_acc[2:].mean()-rfp1_acc[:2].mean():.3f}, Halo={halo_acc[2:].mean()-halo_acc[:2].mean():.3f}")

print("\n✅ Analysis complete!")
EOF
```

**Expected output:**
```
============================================================
8-CLASS vs 6-CLASS COMPARISON
============================================================
                    8-class    6-class    Change
RFP1 (Morphology)   0.434      0.548      +0.114
Halo (Reporter)     0.461      0.505      +0.044

⚠️  8-class accuracy LOWER - good! Not learning artifacts
✅ Performance above chance: 38.1% (RFP1), 33.8% (Halo)

============================================================
6-CLASS DETAILED ANALYSIS
============================================================
PER-CLASS ACCURACY:
Control_T0                     0.587      0.523
APPV717I_T0                    0.651      0.489
Control_0ms_T16                0.423      0.512
Control_10000ms_T16            0.578      0.601
APPV717I_0ms_T16               0.445      0.398
APPV717I_10000ms_T16           0.605      0.509

TIMEPOINT ANALYSIS:
T0 (pre-stim)                  0.619      0.506
T16 (post-stim)                0.513      0.505
```

---

## 📊 Interpreting Your Results

### ✅ Good Results

**1. Overall accuracy > 40%**
- Random chance is 16.7%
- 40-50% = good
- 50%+ = excellent

**2. T0 classes separate by cell line**
- Control_T0 vs APPV717I_T0 have different accuracies
- NOT separated by future stimulation (avoiding artifacts!)

**3. T16 stimulation effect visible**
- Control_10000ms_T16 > Control_0ms_T16 (or vice versa)
- Clear separation between stimulated vs unstimulated

**4. 6-class ≥ 8-class accuracy**
- Model not relying on artifacts
- Learning real biology!

### ⚠️ Red Flags

**1. T0 classes separate by stimulation**
```
Control_0ms_T0 acc = 0.70
Control_10000ms_T0 acc = 0.35
```
- Model still learning well position!
- Need stronger artifact mitigation

**2. T16 classes don't separate**
```
Control_0ms_T16 = Control_10000ms_T16 (both ~0.40)
```
- Stimulation phenotype weak or absent
- Try different channels/markers

**3. 8-class >> 6-class accuracy**
```
8-class: 0.65, 6-class: 0.35
```
- Previous model was mostly learning artifacts!
- Current 6-class is honest but lower

---

## 🐛 Common Issues & Solutions

### Issue 1: Training Loss Not Decreasing

```bash
# Try lower learning rate
python scripts/2_train_models.py \
    --lr 0.0001 \
    ... other args ...
```

### Issue 2: CUDA Out of Memory

```bash
# Reduce batch size
python scripts/2_train_models.py \
    --batch_size 16 \
    ... other args ...
```

### Issue 3: Imbalanced Classes

```python
# In prepare_dataset.py, add class balancing
# The DataModule already handles this automatically
# But check class distribution:
python -c "
import pandas as pd
df = pd.read_csv('experiment1_dataset/metadata.csv')
print(df['label_6class'].value_counts())
"
```

### Issue 4: Poor Generalization

- Check train vs val accuracy gap
- If large gap (>15%), model is overfitting
- Solutions:
  - More data augmentation
  - Reduce model size
  - More dropout
  - More regularization

---

## 📁 File Organization

After completing all steps:
```
experiment1/
├── experiment1_dataset/
│   ├── images/
│   │   ├── RFP1/          # 19,388 images
│   │   └── Halo/          # 19,388 images
│   └── metadata.csv       # Sample metadata
├── results/
│   ├── RFP1/6class/
│   │   ├── checkpoints/   # Best models
│   │   └── logs/          # TensorBoard logs
│   └── Halo/6class/
│       ├── checkpoints/
│       └── logs/
├── evaluation/
│   ├── RFP1/
│   │   ├── confusion_matrix.png
│   │   └── metrics.json
│   └── Halo/
│       ├── confusion_matrix.png
│       └── metrics.json
└── final_analysis/
    ├── 8class_vs_6class_comparison.png
    ├── 6class_detailed_analysis.png
    └── analysis_summary.json
```

---

## ⏱️ Time Estimates

| Step | CPU | GPU | 
|------|-----|-----|
| Dataset prep | 5 min | 5 min |
| Training RFP1 | 8 hours | 10 min |
| Training Halo | 8 hours | 10 min |
| Evaluation (both) | 10 min | 2 min |
| Analysis | 1 min | 1 min |
| **Total** | **~16 hours** | **~30 min** |

---

## 🎯 Next Steps

After completing the pipeline:

1. **Visualize confusion matrices**
   - Look for patterns in misclassifications
   - Which classes are confused?

2. **Compare channels**
   - Which detects phenotype better?
   - RFP1 for morphology, Halo for reporter

3. **Temporal analysis**
   - Plot T0 vs T16 accuracy
   - When does signal emerge?

4. **Export results**
   - Share confusion matrices with collaborators
   - Include metrics.json in papers/reports

5. **Iterate**
   - Try different architectures (ResNet50, EfficientNet)
   - Adjust hyperparameters
   - Add more timepoints

---

## 📞 Getting Help

If stuck:
1. Check error messages carefully
2. Verify file paths and data structure
3. Review METHODOLOGY.md for conceptual help
4. Check example outputs in examples/

**Common questions:**
- "What's a good accuracy?" → >40% for 6 classes
- "Why is 6-class lower than 8-class?" → Not learning artifacts anymore!
- "Which channel is better?" → Depends on phenotype. Try both!

**Good luck! 🚀**
