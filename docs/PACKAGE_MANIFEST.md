# 6-Class Phenotype Detection Pipeline - Package Manifest

**Complete artifact-free deep learning pipeline for biological phenotype detection**

---

## 📦 Package Contents

### Core Documentation
```
README.md                   # Overview and quick start
QUICKSTART.md              # Detailed step-by-step guide
METHODOLOGY.md             # Scientific rationale and theory
PACKAGE_MANIFEST.md        # This file
```

### Scripts (scripts/)
```
1_prepare_dataset.py       # Extract and prepare HDF5 data
2_train_models.py          # Train 6-class classification models
3_evaluate_models.py       # Generate confusion matrices and metrics
4_analyze_results.py       # Comprehensive analysis and visualization
```

### Examples (examples/)
```
example_commands.sh        # Copy-paste commands for full workflow
expected_outputs.md        # What to expect at each step
```

---

## 🚀 Quick Start Commands

```bash
# 1. Download the package
cd 6class_phenotype_pipeline/

# 2. Install dependencies
pip install torch torchvision pytorch-lightning pandas numpy matplotlib seaborn scikit-learn tqdm h5py Pillow

# 3. Prepare your data
python scripts/1_prepare_dataset.py \
    --hdf5_dir /path/to/hdf5_neurons \
    --output_dir ./dataset \
    --bad_wells [your bad wells]

# Add class_label column
python -c "
import pandas as pd
df = pd.read_csv('dataset/metadata.csv')
df['class_label'] = df['label_6class']
df.to_csv('dataset/metadata.csv', index=False)
"

# 4. Train models
python scripts/2_train_models.py \
    --metadata ./dataset/metadata.csv \
    --root_dir ./dataset \
    --output_dir ./results \
    --tasks 6class \
    --channels RFP1 Halo

# 5. Evaluate models
# RFP1
python scripts/3_evaluate_models.py \
    --checkpoint results/RFP1/6class/checkpoints/best-*.ckpt \
    --metadata ./dataset/metadata.csv \
    --root_dir ./dataset \
    --channel RFP1 \
    --output_dir ./evaluation/RFP1

# Halo
python scripts/3_evaluate_models.py \
    --checkpoint results/Halo/6class/checkpoints/best-*.ckpt \
    --metadata ./dataset/metadata.csv \
    --root_dir ./dataset \
    --channel Halo \
    --output_dir ./evaluation/Halo

# 6. Analyze results
python scripts/4_analyze_results.py \
    --rfp1_6class ./evaluation/RFP1 \
    --halo_6class ./evaluation/Halo \
    --metadata ./dataset/metadata.csv \
    --output_dir ./analysis
```

---

## 📋 Script Details

### 1_prepare_dataset.py

**Purpose**: Extract images from HDF5 files and create 6-class labels

**Key Features**:
- Filters bad wells and low-quality neurons
- Extracts T0 and T16 timepoints
- Creates 6-class labels (collapses T0 stimulation)
- Saves images as PNGs
- Generates metadata CSV

**Arguments**:
```bash
--hdf5_dir     # Input directory with HDF5 files
--output_dir   # Output directory for dataset
--bad_wells    # Space-separated list of wells to exclude
--min_timepoints # Minimum timepoints required (default: 16)
```

**Outputs**:
- `images/RFP1/*.png` - RFP1 channel images
- `images/Halo/*.png` - Halo channel images
- `metadata.csv` - Sample metadata with labels
- `dataset_summary.json` - Statistics

---

### 2_train_models.py

**Purpose**: Train deep learning models on both channels

**Key Features**:
- ResNet18 architecture with transfer learning
- Early stopping (patience=15)
- TensorBoard logging
- Automatic checkpointing
- GPU acceleration

**Arguments**:
```bash
--metadata     # Path to metadata CSV
--root_dir     # Root directory containing images/
--output_dir   # Output directory for models
--tasks        # Classification tasks (use '6class')
--channels     # Channels to train (RFP1, Halo, or both)
--batch_size   # Batch size (default: 32)
--epochs       # Max epochs (default: 100)
--lr           # Learning rate (default: 0.001)
--num_workers  # Data loading workers (default: 4)
```

**Outputs**:
- `{channel}/6class/checkpoints/*.ckpt` - Model weights
- `{channel}/6class/logs/` - TensorBoard logs

---

### 3_evaluate_models.py

**Purpose**: Generate confusion matrices and metrics

**Key Features**:
- Test set evaluation
- Confusion matrix visualization
- Per-class accuracy
- Classification report
- Saves numpy arrays for further analysis

**Arguments**:
```bash
--checkpoint   # Path to model checkpoint
--metadata     # Path to metadata CSV
--root_dir     # Root directory containing images/
--channel      # Channel (RFP1 or Halo)
--output_dir   # Output directory for evaluation
```

**Outputs**:
- `confusion_matrix.png` - Raw counts
- `confusion_matrix_normalized.png` - Normalized
- `confusion_matrix.npy` - Numpy array
- `confusion_matrix_normalized.npy` - Numpy array
- `metrics.json` - Performance metrics
- `classification_report.txt` - Detailed report

---

### 4_analyze_results.py

**Purpose**: Comprehensive analysis comparing channels and classes

**Key Features**:
- 8-class vs 6-class comparison
- Per-class accuracy breakdown
- Temporal analysis (T0 vs T16)
- Stimulation effect analysis
- Multi-panel visualizations

**Arguments**:
```bash
--rfp1_8class  # RFP1 8-class evaluation (optional)
--rfp1_6class  # RFP1 6-class evaluation (required)
--halo_8class  # Halo 8-class evaluation (optional)
--halo_6class  # Halo 6-class evaluation (required)
--metadata     # Path to 6-class metadata CSV
--output_dir   # Output directory for analysis
```

**Outputs**:
- `8class_vs_6class_comparison.png` - Accuracy comparison
- `6class_detailed_analysis.png` - 4-panel analysis
- `analysis_summary.json` - All metrics

---

## 🎯 Class Structure

### 6 Classes:
1. **Control_T0** - WT baseline (merged 0ms + 10000ms)
2. **APPV717I_T0** - APP mutant baseline (merged 0ms + 10000ms)
3. **Control_0ms_T16** - WT unstimulated at endpoint
4. **Control_10000ms_T16** - WT stimulated at endpoint
5. **APPV717I_0ms_T16** - APP mutant unstimulated at endpoint
6. **APPV717I_10000ms_T16** - APP mutant stimulated at endpoint

### Why This Design?

At T0, cells haven't been stimulated yet. Merging 0ms and 10000ms groups prevents the model from learning:
- Well position effects
- Future treatment assignments
- Other technical artifacts

The model must learn only the biological signal that emerges after stimulation at T16.

---

## 📊 Expected Performance

### Benchmarks

| Channel | Accuracy | Above Chance | Best For |
|---------|----------|--------------|----------|
| RFP1 (Morphology) | 50-55% | ~35% | Structural changes, late phenotypes |
| Halo (Reporter) | 45-52% | ~30% | Molecular events, early responses |

### Random Baseline
- **6 classes**: 16.7% (1/6)
- **Good performance**: >40% (2.4× baseline)
- **Excellent**: >50% (3× baseline)

---

## 🔧 Customization Guide

### For Different Experimental Designs

**1. Different conditions/treatments:**
Edit `scripts/1_prepare_dataset.py` around line 120:
```python
def extract_condition_info(hdf5_path):
    # Customize for your experiment
    condition = ...  # Your condition extraction
    cell_line = ...  # Your cell line logic
    treatment = ...  # Your treatment variable
```

**2. Different timepoints:**
Change T0 and T16 to your timepoints:
```python
# Line ~180 in prepare_dataset.py
t0_idx = 0  # Your baseline timepoint
t15_idx = 15  # Your endpoint timepoint
```

**3. Different class structure:**
Modify the label logic:
```python
if metadata['timepoint'] == 'T0':
    metadata['label_6class'] = f"{cell_line}_T0"
else:
    metadata['label_6class'] = f"{cell_line}_{drug}_{dose}_T16"
```

**4. Different channels:**
Add your channel to the list:
```python
# In train_models.py
--channels YourChannel1 YourChannel2
```

---

## 🐛 Common Issues

### "ModuleNotFoundError: No module named 'train_lightning'"

**Solution**: The scripts expect `train_lightning.py` in `/mnt/project/`. Update the import path:
```python
# In each script, change:
sys.path.insert(0, '/mnt/project')
# To your path:
sys.path.insert(0, '/path/to/your/train_lightning.py')
```

### "KeyError: 'class_label'"

**Solution**: Add the column:
```bash
python -c "
import pandas as pd
df = pd.read_csv('dataset/metadata.csv')
df['class_label'] = df['label_6class']
df.to_csv('dataset/metadata.csv', index=False)
"
```

### "CUDA out of memory"

**Solution**: Reduce batch size:
```bash
python scripts/2_train_models.py --batch_size 16 [other args]
```

### Low accuracy (<30%)

**Check**:
1. Data quality (focus, exposure, tracking)
2. Class balance (imbalanced classes?)
3. Phenotype strength (visible by eye?)
4. Hyperparameters (try different learning rates)

---

## 📚 Documentation Map

**Start here:**
1. `README.md` - Overview and philosophy
2. `QUICKSTART.md` - Step-by-step tutorial
3. `examples/example_commands.sh` - Copy-paste commands

**For understanding:**
4. `METHODOLOGY.md` - Scientific background
5. `examples/expected_outputs.md` - What to expect

**For customization:**
6. Script docstrings - Detailed implementation
7. This file - Complete reference

---

## 🎓 Citation

If you use this pipeline in your research, please cite:

```
[Your citation here]
```

---

## 📧 Support

For issues or questions:
1. Check QUICKSTART.md for common problems
2. Review expected_outputs.md to verify results
3. Consult METHODOLOGY.md for conceptual questions
4. Open an issue with error logs and data description

---

## ✅ Validation Checklist

Pipeline is working correctly if:
- [ ] Dataset preparation completes without errors
- [ ] Training loss decreases over epochs
- [ ] Val accuracy > train accuracy initially (not overfitting)
- [ ] Test accuracy > 40% (2.4× random chance)
- [ ] T0 classes separate by cell line, not stimulation
- [ ] T16 classes show treatment effect
- [ ] Confusion matrix has strong diagonal
- [ ] Results are biologically interpretable

---

## 🎉 Success Metrics

**Your pipeline is working well if:**
- ✅ Accuracy > 40% on both channels
- ✅ T16 stimulation separates (acc gap >10%)
- ✅ T0 doesn't leak future treatment info
- ✅ Results replicate across independent runs
- ✅ Findings match visual inspection
- ✅ Model learns biology, not artifacts!

**Congratulations on running artifact-free phenotype detection! 🔬**

---

*Version: 1.0 | Last updated: [Date] | License: [Your license]*
