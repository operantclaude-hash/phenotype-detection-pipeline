# Artifact-Free Phenotype Detection Pipeline

**Complete workflow for training deep learning models to detect biological phenotypes while avoiding technical artifacts.**

## 🎯 Overview

This pipeline implements a **6-class classification approach** that:
- ✅ **Collapses pre-stimulation timepoints** (T0) to prevent learning well position artifacts
- ✅ **Separates post-stimulation timepoints** (T16) to capture real biological phenotypes
- ✅ **Compares morphology (RFP1) vs reporter (Halo)** channels
- ✅ **Provides comprehensive analysis** of model performance and biological insights

### The Problem This Solves

Standard 8-class models can achieve high accuracy by learning **technical artifacts** rather than biology:
- Well position effects
- Imaging shadows/gradients
- Batch effects from experimental design

By collapsing pre-stimulation classes, we force the model to learn **only the biological signal** that emerges after treatment.

---

## 📁 Package Contents

```
6class_phenotype_pipeline/
├── README.md                          # This file
├── QUICKSTART.md                      # Quick start guide
├── METHODOLOGY.md                     # Scientific rationale
├── scripts/
│   ├── 1_prepare_dataset.py          # Extract and prepare data
│   ├── 2_train_models.py             # Train classification models
│   ├── 3_evaluate_models.py          # Generate confusion matrices
│   └── 4_analyze_results.py          # Comprehensive analysis
└── examples/
    ├── example_commands.sh            # Copy-paste commands
    └── expected_outputs.md            # What to expect
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- PyTorch with CUDA (for GPU training)
- Required packages: `pytorch-lightning`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `h5py`, `Pillow`

### Installation
```bash
# Install dependencies
pip install torch torchvision pytorch-lightning pandas numpy matplotlib seaborn scikit-learn tqdm h5py Pillow

# Clone or download this package
cd 6class_phenotype_pipeline
```

### Complete Workflow (30 minutes)

```bash
# 1. Prepare dataset with 6-class labels (5 min)
python scripts/1_prepare_dataset.py \
    --hdf5_dir /path/to/hdf5_neurons \
    --output_dir ./dataset \
    --bad_wells A1 A2 A3 [... list bad wells ...]

# 2. Train models on both channels (15-20 min)
python scripts/2_train_models.py \
    --metadata ./dataset/metadata.csv \
    --root_dir ./dataset \
    --output_dir ./results \
    --tasks 6class \
    --channels RFP1 Halo \
    --epochs 100

# 3. Evaluate trained models (2 min)
python scripts/3_evaluate_models.py \
    --rfp1_checkpoint ./results/RFP1/6class/checkpoints/best-*.ckpt \
    --halo_checkpoint ./results/Halo/6class/checkpoints/best-*.ckpt \
    --metadata ./dataset/metadata.csv \
    --root_dir ./dataset \
    --output_dir ./evaluation

# 4. Comprehensive analysis (< 1 min)
python scripts/4_analyze_results.py \
    --rfp1_8class ./results_8class/RFP1/full/evaluation \
    --rfp1_6class ./evaluation/RFP1 \
    --halo_8class ./results_8class/Halo/full/evaluation \
    --halo_6class ./evaluation/Halo \
    --metadata ./dataset/metadata.csv \
    --output_dir ./analysis
```

---

## 📊 Understanding the Results

### Key Metrics

1. **Overall Accuracy**
   - Random chance: 16.7% (1/6 classes)
   - Good performance: >40%
   - Excellent: >50%

2. **Per-Class Accuracy**
   - T0 classes: Should separate by **cell line** (not stimulation)
   - T16 classes: Should separate by **stimulation** (phenotype emerged)

3. **Artifact Contribution**
   - Compare 8-class vs 6-class accuracy
   - Drop indicates model was learning artifacts

### Expected Outputs

**Analysis generates:**
- `8class_vs_6class_comparison.png` - Accuracy comparison
- `6class_detailed_analysis.png` - 4-panel detailed view
- `confusion_matrix_rfp1.png` - RFP1 confusion matrix
- `confusion_matrix_halo.png` - Halo confusion matrix
- `analysis_summary.json` - All metrics

---

## 🔬 Scientific Interpretation

### Good Results Look Like:

✅ **At T0 (Pre-stimulation)**:
- Control_T0 vs APPV717I_T0 separate (baseline difference)
- NO separation by future stimulation (avoiding artifacts!)

✅ **At T16 (Post-stimulation)**:
- Control_0ms_T16 vs Control_10000ms_T16 separate (stimulation effect)
- APPV717I_0ms_T16 vs APPV717I_10000ms_T16 separate (stimulation effect)

✅ **Channel Comparison**:
- RFP1 (morphology): Often better at late phenotypes
- Halo (reporter): May detect earlier or subtler changes

### Red Flags:

⚠️ **8-class accuracy >> 6-class accuracy**
- Model was learning technical artifacts

⚠️ **T0 classes separate by stimulation**
- Well position effects not fully eliminated

⚠️ **T16 classes don't separate by stimulation**
- Phenotype may not be strong or detectable

---

## 🎓 Class Structure

### 6 Classes:
1. **Control_T0** - WT baseline (collapsed 0ms + 10000ms)
2. **APPV717I_T0** - APP mutant baseline (collapsed 0ms + 10000ms)
3. **Control_0ms_T16** - WT unstimulated at endpoint
4. **Control_10000ms_T16** - WT stimulated at endpoint
5. **APPV717I_0ms_T16** - APP mutant unstimulated at endpoint
6. **APPV717I_10000ms_T16** - APP mutant stimulated at endpoint

### Why Collapse T0?

At T0 (timepoint 0), cells haven't been stimulated yet. Any differences between 0ms and 10000ms groups are **purely technical artifacts**:
- The cells are identical biologically
- Differences come from well position, imaging conditions, etc.
- Model learns to cheat by memorizing which wells will be stimulated

By merging these classes, we force the model to ignore artifacts and learn only the **real biological signal** that develops after stimulation.

---

## 📈 Adapting to Your Experiment

### Required Data Structure

**HDF5 files** organized as:
```
hdf5_neurons/
  ├── A1/
  │   └── experiment_A1_tile1_neuron1.h5
  ├── A2/
  │   └── experiment_A2_tile1_neuron1.h5
  ...
```

**Each HDF5 contains:**
- `data/images/RFP1`: (T, H, W) array
- `data/images/Halo`: (T, H, W) array (optional)
- `metadata/`: condition, well, tile, cell line info

### Modify for Your Conditions

Edit `scripts/1_prepare_dataset.py`:

```python
# Change condition names (line ~120)
def extract_condition_info(hdf5_path):
    # Your custom logic here
    condition = ...  # e.g., "Drug_A", "Drug_B"
    cell_line = ...  # e.g., "WT", "KO"
    treatment = ...  # e.g., "0uM", "10uM"
    
# Change class definitions (line ~180)
if metadata['timepoint'] == 'T0':
    # Collapse pre-treatment
    metadata['label_6class'] = f"{condition}_T0"
else:
    # Separate post-treatment
    metadata['label_6class'] = f"{condition}_{treatment}_T16"
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "class_label column not found"**
```bash
# Add class_label column to metadata
python -c "
import pandas as pd
df = pd.read_csv('dataset/metadata.csv')
df['class_label'] = df['label_6class']
df.to_csv('dataset/metadata.csv', index=False)
"
```

**2. "Not enough timepoints"**
- Check your HDF5 files have ≥16 timepoints
- Adjust threshold in `prepare_dataset.py` if needed

**3. "CUDA out of memory"**
- Reduce batch size: `--batch_size 16`
- Use CPU: `--accelerator cpu`

**4. "Confusion matrix not saved"**
- Check evaluate script saves `.npy` files
- See QUICKSTART.md for fix

---

## 📚 Additional Resources

- **QUICKSTART.md** - Detailed step-by-step guide
- **METHODOLOGY.md** - Scientific background and rationale
- **examples/** - Example commands and expected outputs

---

## 🤝 Citation

If you use this pipeline, please cite:
```
[Your paper/method here]
```

---

## 📧 Support

For questions or issues:
1. Check QUICKSTART.md and METHODOLOGY.md
2. Review example outputs in examples/
3. Open an issue with error logs and data description

---

## ⚖️ License

[Your license here - MIT recommended]

---

## 🎉 Success Criteria

Your pipeline is working well if:
- ✅ 6-class accuracy > 40% (RFP1 or Halo)
- ✅ T16 accuracy > T0 accuracy (phenotype emerges)
- ✅ Stimulation separates at T16 but not at T0
- ✅ Model learns biology, not artifacts!

**Good luck with your experiments! 🔬**
