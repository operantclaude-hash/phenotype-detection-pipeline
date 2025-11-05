# Experiment Management Guide

**Complete workflow for organizing, tracking, and comparing experiments**

---

## 🎯 Philosophy

**Problem**: Running multiple experiments leads to:
- Overwritten results
- Lost code versions
- Unclear comparisons
- Difficult reproduction

**Solution**: Structured experiment management with:
- Git branches for code versions
- Organized directories for outputs
- Config files for reproducibility
- Comparison tools for analysis

---

## 📁 Directory Structure

```
phenotype-detection-pipeline/
├── src/                          # Source code (stable)
│   ├── prepare_dataset.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── analyze_results.py
│
├── experiments/                  # All experiments here
│   ├── 2024-11-04_6class_t0-t16/      # Experiment 1
│   │   ├── config.yaml                 # Experiment config
│   │   ├── README.md                   # Documentation
│   │   ├── dataset/                    # Dataset for this exp
│   │   │   ├── images/
│   │   │   └── metadata.csv
│   │   ├── results/                    # Model checkpoints
│   │   │   ├── RFP1/
│   │   │   └── Halo/
│   │   ├── evaluation/                 # Evaluation outputs
│   │   │   ├── RFP1/
│   │   │   └── Halo/
│   │   ├── analysis/                   # Final analysis
│   │   └── logs/                       # Training logs
│   │
│   ├── 2024-11-04_6class_t0-t15/      # Experiment 2
│   │   └── ... (same structure)
│   │
│   ├── 2024-11-05_binary_app-vs-wt/   # Experiment 3
│   │   └── ... (same structure)
│   │
│   └── EXPERIMENTS_INDEX.md            # Index of all experiments
│
├── comparison_reports/           # Cross-experiment comparisons
│   ├── 20241104_6class_comparison.md
│   └── 20241104_6class_comparison.png
│
├── config/                       # Config templates
│   └── experiment_template.yaml
│
├── scripts/                      # Helper scripts
│   ├── new_experiment.sh         # Create new experiment
│   └── compare_experiments.py    # Compare experiments
│
└── README.md                     # Main project README
```

---

## 🚀 Workflow

### 1. Starting a New Experiment

```bash
# Create experiment structure
./scripts/new_experiment.sh binary_app_vs_wt "Binary: APP vs WT at T15"

# This creates:
# experiments/2024-11-05_binary_app_vs_wt/
#   ├── config.yaml
#   ├── README.md
#   ├── dataset/
#   ├── results/
#   ├── evaluation/
#   ├── analysis/
#   └── logs/

# Create git branch for code changes
git checkout -b experiment/binary_app_vs_wt
```

### 2. Configure Experiment

Edit `experiments/2024-11-05_binary_app_vs_wt/config.yaml`:

```yaml
experiment:
  name: "binary_app_vs_wt"
  description: "Binary classification: APP vs WT at T15 10000ms"

data:
  timepoints:
    endpoint: "T15"  # Changed from T16
  
  classification:
    type: "binary"   # Changed from 6class
    classes:
      - APPV717I_10000ms_T15
      - Control_10000ms_T15

training:
  channels:
    - RFP1          # Only RFP1, not Halo
```

### 3. Modify Code (if needed)

```bash
# Make changes to src/prepare_dataset.py for binary classification
# All changes are tracked in this branch

# Example: Add binary classification support
nano src/prepare_dataset.py
# ... add binary label logic ...

git add src/prepare_dataset.py
git commit -m "Add binary classification for APP vs WT at T15"
```

### 4. Run Experiment

```bash
# Set experiment directory
EXP_DIR="experiments/2024-11-05_binary_app_vs_wt"

# Prepare dataset
python src/prepare_dataset.py \
    --hdf5_dir "/path/to/hdf5" \
    --output_dir "$EXP_DIR/dataset" \
    --bad_wells A1 A2 A3 \
    | tee "$EXP_DIR/logs/prepare.log"

# Train model
python src/train_models.py \
    --metadata "$EXP_DIR/dataset/metadata.csv" \
    --root_dir "$EXP_DIR/dataset/" \
    --output_dir "$EXP_DIR/results/" \
    --tasks binary \
    --channels RFP1 \
    | tee "$EXP_DIR/logs/training.log"

# Evaluate
BEST_CKPT=$(ls "$EXP_DIR/results/RFP1/binary/checkpoints/best-"*.ckpt | head -1)
python src/evaluate_models.py \
    --checkpoint "$BEST_CKPT" \
    --metadata "$EXP_DIR/dataset/metadata.csv" \
    --root_dir "$EXP_DIR/dataset/" \
    --channel RFP1 \
    --output_dir "$EXP_DIR/evaluation/RFP1" \
    | tee "$EXP_DIR/logs/evaluation.log"
```

### 5. Document Results

Edit `experiments/2024-11-05_binary_app_vs_wt/README.md`:

```markdown
## Results

### Training Results
- **RFP1**: 0.872 accuracy

### Key Findings
- Binary classification much easier than 6-class (87% vs 54%)
- Clear separation between APP and WT at T15 with stimulation
- RFP1 channel sufficient for this task

### Conclusions
APP vs WT phenotype is highly detectable at T15 with 10000ms stimulation.
```

### 6. Compare with Previous Experiments

```bash
# Compare 6-class vs binary
python scripts/compare_experiments.py \
    experiments/2024-11-04_6class_t0-t16/ \
    experiments/2024-11-05_binary_app_vs_wt/ \
    --output_dir comparison_reports

# Generates:
# - comparison_reports/comparison_20241105_143022.png
# - comparison_reports/comparison_20241105_143022.md
# - comparison_reports/comparison_20241105_143022.csv
```

### 7. Merge Back (if successful)

```bash
# If experiment successful and you want to keep changes
git checkout main
git merge experiment/binary_app_vs_wt
git push

# Tag this version
git tag -a v1.1-binary -m "Add binary classification support"
git push --tags
```

---

## 📊 Tracking Multiple Experiments

### experiments/EXPERIMENTS_INDEX.md

```markdown
# Experiments Index

## Active Experiments

| Date | Name | Status | RFP1 Acc | Halo Acc | Notes |
|------|------|--------|----------|----------|-------|
| 2024-11-04 | 6class_t0-t16 | ✅ Complete | 0.543 | 0.514 | Baseline |
| 2024-11-04 | 6class_t0-t15 | ✅ Complete | 0.521 | 0.498 | T15 endpoint |
| 2024-11-05 | binary_app_vs_wt | 🔄 Running | - | - | Binary test |

## Completed

### 6class_t0-t16 (Baseline)
- **Date**: 2024-11-04
- **Goal**: Establish artifact-free baseline
- **Result**: RFP1=0.543, Halo=0.514
- **Key Finding**: Prevents well position artifact learning
- **Branch**: `main`

### 6class_t0-t15  
- **Date**: 2024-11-04
- **Goal**: Test T15 vs T16 endpoint
- **Result**: RFP1=0.521, Halo=0.498
- **Key Finding**: T16 slightly better than T15
- **Branch**: `experiment/t15-endpoint`

## Planned

- [ ] Binary classification: Control 0ms vs 10000ms at T16
- [ ] 4-class: Cell line × Stimulation (collapse timepoints)
- [ ] RFP1-only training to save compute
- [ ] Ensemble: RFP1 + Halo predictions

## Failed/Abandoned

None yet
```

---

## 🔄 Common Workflows

### Experiment 1: New Classification Task

```bash
# 1. Create experiment
./scripts/new_experiment.sh new_task "Description"

# 2. Create branch
git checkout -b experiment/new_task

# 3. Modify code
# ... edit src files ...

# 4. Run pipeline
# ... prepare, train, evaluate ...

# 5. Document
# ... update README.md ...

# 6. Compare
python scripts/compare_experiments.py exp1 exp2

# 7. Decide: keep or discard
git merge experiment/new_task  # Keep
# OR
git checkout main  # Discard
```

### Experiment 2: Same Code, Different Data

```bash
# 1. Create experiment (no code changes needed)
./scripts/new_experiment.sh different_data "Testing new plate"

# 2. Stay on main branch (no code changes)

# 3. Run with new data path
EXP_DIR="experiments/2024-11-05_different_data"
python src/prepare_dataset.py \
    --hdf5_dir "/path/to/NEW/data" \
    --output_dir "$EXP_DIR/dataset"

# 4. Compare results
python scripts/compare_experiments.py \
    experiments/2024-11-04_6class_t0-t16/ \
    experiments/2024-11-05_different_data/
```

### Experiment 3: Hyperparameter Sweep

```bash
# Test different learning rates
for LR in 0.0001 0.001 0.01; do
    EXP_DIR="experiments/2024-11-05_lr_${LR}"
    mkdir -p "$EXP_DIR"
    
    python src/train_models.py \
        --lr $LR \
        --output_dir "$EXP_DIR/results" \
        ...
done

# Compare all
python scripts/compare_experiments.py experiments/2024-11-05_lr_*/
```

---

## 🎓 Best Practices

### Do's ✅

1. **Always create new experiment directory** - Never overwrite
2. **Document immediately** - Update README.md after each step
3. **Use git branches** for code changes
4. **Save all commands** in experiment README
5. **Compare before deciding** what to keep
6. **Tag successful experiments** with git tags

### Don'ts ❌

1. **Don't mix data from experiments** - Keep isolated
2. **Don't commit large files** - Use .gitignore
3. **Don't delete experiments** - Archive instead
4. **Don't skip documentation** - Future you will thank you
5. **Don't merge untested code** to main

---

## 🔍 Finding Past Results

```bash
# Find all experiments with accuracy > 0.5
grep -r "test_accuracy" experiments/*/evaluation/*/metrics.json | grep "0\.[5-9]"

# List all experiments by date
ls -lt experiments/

# Find which experiment used specific hyperparameters
grep -r "learning_rate: 0.001" experiments/*/config.yaml

# See what was trained on a specific date
ls experiments/2024-11-04*/
```

---

## 📦 Archiving Experiments

```bash
# Archive old experiments (keep results, delete large files)
./scripts/archive_experiment.sh experiments/2024-11-04_6class_t0-t16

# This:
# 1. Removes dataset/images/ (can regenerate)
# 2. Removes results/*.ckpt (keep only best)
# 3. Keeps evaluation/ and analysis/
# 4. Creates experiments/archive/2024-11-04_6class_t0-t16.tar.gz
```

---

## 🎯 Example: Your Binary Classification Experiment

```bash
# 1. Setup
./scripts/new_experiment.sh binary_app_vs_wt_t15 \
    "Binary classification: APP vs WT at T15 with 10000ms stimulation, RFP1 only"

git checkout -b experiment/binary_app_vs_wt

# 2. Modify prepare_dataset.py
cat >> src/prepare_dataset.py << 'EOF'

# Add binary classification logic
if args.binary_app_vs_wt:
    # Filter for T15 and 10000ms only
    df = df[(df['timepoint'] == 'T15') & (df['stimulation'] == '10000ms')]
    df['label_binary'] = df['cell_line']  # APPV717I or Control
    df['class_label'] = df['label_binary']
EOF

# 3. Run experiment
EXP="experiments/$(date +%Y-%m-%d)_binary_app_vs_wt_t15"

python src/prepare_dataset.py \
    --hdf5_dir "/gladstone/finkbeiner/linsley/GXYTMPS/Jeremy/GXYTMP-A2screenPlate2/hdf5_neurons" \
    --output_dir "$EXP/dataset" \
    --binary_app_vs_wt \
    --bad_wells A1 A2 A3 ...

python src/train_models.py \
    --metadata "$EXP/dataset/metadata.csv" \
    --root_dir "$EXP/dataset/" \
    --output_dir "$EXP/results/" \
    --tasks binary \
    --channels RFP1 \
    --epochs 100

# 4. Compare
python scripts/compare_experiments.py \
    experiments/2024-11-04_6class_t0-t16/ \
    "$EXP"

# 5. Merge if good
git add src/prepare_dataset.py
git commit -m "Add binary APP vs WT classification"
git checkout main
git merge experiment/binary_app_vs_wt
git tag -a v1.1-binary -m "Binary classification support"
```

---

## 🎉 Summary

**This system gives you:**

✅ **Organization**: Each experiment isolated
✅ **Reproducibility**: Config files + git branches
✅ **Comparison**: Easy to compare across experiments
✅ **History**: Never lose results
✅ **Collaboration**: Share experiments via git

**Key principle**: *Experiments are cheap, insights are valuable. Keep everything organized so you can learn from all your work.*

---

*Questions? Check examples in experiments/ or ask!* 🚀
