# Artifact-Free Phenotype Detection Pipeline

**Deep learning pipeline for detecting biological phenotypes while avoiding technical artifacts**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

Standard multi-class image classifiers often achieve high accuracy by learning **technical artifacts** (well positions, imaging shadows, batch effects) rather than true biological signals. This pipeline implements a **6-class design** that prevents artifact learning by collapsing pre-treatment timepoints.

### Key Innovation

**Problem**: At timepoint 0 (T0), cells haven't been treated yet. Any model accuracy distinguishing "will be treated" vs "won't be treated" groups must come from memorizing well positions - not biology.

**Solution**: Merge pre-treatment groups to force models to learn only post-treatment phenotypes.

**Result**: Lower accuracy, but honest - models learn real biology! 🔬

---

## ✨ Features

- ✅ **Prevents artifact learning** through intelligent class design
- ✅ **Multi-channel support** (morphology + reporter channels)
- ✅ **Comprehensive evaluation** with confusion matrices and temporal analysis
- ✅ **Production-ready** code with full documentation
- ✅ **GPU accelerated** training (~30 min vs ~16 hours on CPU)
- ✅ **Fully customizable** for different experimental designs

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended but not required)

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/phenotype-detection-pipeline.git
cd phenotype-detection-pipeline

# Create environment
conda create -n phenotype python=3.10
conda activate phenotype

# Install PyTorch (check https://pytorch.org for your CUDA version)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### Alternative: pip only

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Complete Workflow (30 minutes on GPU)

```bash
# 1. Prepare dataset (5 min)
python src/prepare_dataset.py \
    --hdf5_dir /path/to/hdf5_neurons \
    --output_dir ./dataset \
    --bad_wells A1 A2 A3

# Add required column
python -c "import pandas as pd; df=pd.read_csv('dataset/metadata.csv'); df['class_label']=df['label_6class']; df.to_csv('dataset/metadata.csv', index=False)"

# 2. Train models (15-20 min)
python src/train_models.py \
    --metadata dataset/metadata.csv \
    --root_dir dataset/ \
    --output_dir results/ \
    --tasks 6class \
    --channels RFP1 Halo \
    --batch_size 32 \
    --epochs 100

# 3. Evaluate models (2 min)
python src/evaluate_models.py \
    --checkpoint results/RFP1/6class/checkpoints/best-*.ckpt \
    --metadata dataset/metadata.csv \
    --root_dir dataset/ \
    --channel RFP1 \
    --output_dir evaluation/RFP1

# 4. Analyze results (< 1 min)
python src/analyze_results.py \
    --rfp1_6class evaluation/RFP1 \
    --halo_6class evaluation/Halo \
    --metadata dataset/metadata.csv \
    --output_dir analysis/
```

---

## 📊 Expected Results

### Performance Benchmarks

| Channel | Accuracy | Above Chance | Best For |
|---------|----------|--------------|----------|
| Morphology (RFP1) | 50-55% | ~35% | Structural changes, late phenotypes |
| Reporter (Halo) | 45-52% | ~30% | Molecular events, early responses |

**Random baseline**: 16.7% (1/6 classes)

### What You Get

- ✅ Confusion matrices showing per-class performance
- ✅ Temporal analysis (pre-treatment vs post-treatment)
- ✅ Treatment separability metrics
- ✅ Channel comparison visualizations
- ✅ Publication-ready figures

---

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Detailed step-by-step tutorial
- **[Methodology](docs/METHODOLOGY.md)** - Scientific background and rationale
- **[Usage Guide](docs/USAGE.md)** - Complete API reference
- **[Examples](examples/)** - Example commands and expected outputs

---

## 🧬 Class Structure

### 6-Class Design

**Pre-treatment (T0)** - Merged to prevent artifact learning:
1. Control_T0 (merged 0ms + 10000ms)
2. APPV717I_T0 (merged 0ms + 10000ms)

**Post-treatment (T16)** - Separated to detect phenotype:
3. Control_0ms_T16
4. Control_10000ms_T16
5. APPV717I_0ms_T16
6. APPV717I_10000ms_T16

This design forces models to:
- ❌ **Cannot** learn well positions or future treatments
- ✅ **Must** learn biological phenotypes that emerge after treatment

---

## 🛠️ Customization

### For Different Conditions

Edit `src/prepare_dataset.py`:

```python
# Customize condition extraction (line ~120)
def extract_condition_info(hdf5_path):
    condition = ...  # Your logic here
    cell_line = ...
    treatment = ...
    return condition, cell_line, treatment

# Customize class labels (line ~180)
if metadata['timepoint'] == 'T0':
    label = f"{cell_line}_T0"  # Collapse pre-treatment
else:
    label = f"{cell_line}_{treatment}_T16"  # Separate post-treatment
```

### For Different Architectures

```bash
python src/train_models.py \
    --architecture resnet50 \  # or efficientnet_b0
    ... other args
```

---

## 🧪 Validation

Pipeline is working correctly if:
- ✅ Overall accuracy > 40% (2.4× random chance)
- ✅ T0 classes separate by cell line (not future treatment)
- ✅ T16 classes separate by treatment
- ✅ Confusion matrix shows strong diagonal
- ✅ Results are biologically interpretable

---

## 📖 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{phenotype_detection_pipeline,
  author = {Your Name},
  title = {Artifact-Free Phenotype Detection Pipeline},
  year = {2024},
  url = {https://github.com/yourusername/phenotype-detection-pipeline}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🐛 Issues & Support

- **Bug reports**: [Open an issue](https://github.com/yourusername/phenotype-detection-pipeline/issues)
- **Questions**: Check [docs/](docs/) or open a discussion
- **Feature requests**: [Open an issue](https://github.com/yourusername/phenotype-detection-pipeline/issues) with "enhancement" label

---

## 🎓 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Inspired by challenges in high-content imaging screens
- Developed for [your lab/institution]

---

## 📊 Results Preview

### Confusion Matrix Example

<img src="docs/images/confusion_matrix_example.png" width="600">

*Example confusion matrix showing strong diagonal and biological signal separation*

### Temporal Analysis

<img src="docs/images/temporal_analysis_example.png" width="600">

*Phenotype emergence from pre-treatment (T0) to post-treatment (T16)*

---

**⭐ Star this repo if it helps your research!**
