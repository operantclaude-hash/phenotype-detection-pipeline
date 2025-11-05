#!/bin/bash
# Setup experiment management system
# Run this once to initialize the structure

set -e

echo "=========================================="
echo "SETTING UP EXPERIMENT MANAGEMENT SYSTEM"
echo "=========================================="
echo ""

# Create main directories
echo "📁 Creating directory structure..."
mkdir -p experiments
mkdir -p comparison_reports
mkdir -p config
mkdir -p scripts

# Copy helper scripts
echo "📝 Installing helper scripts..."

# Copy experiment template
cat > config/experiment_template.yaml << 'EOF'
experiment:
  name: "experiment_name"
  date: "YYYY-MM-DD"
  description: |
    What this experiment tests
  researcher: "your_name"
  hypothesis: |
    What you expect to find

data:
  hdf5_dir: "/path/to/hdf5_neurons"
  output_dir: "./experiments/YYYY-MM-DD_name/dataset"
  bad_wells: []
  timepoints:
    baseline: "T0"
    endpoint: "T16"
  classification:
    type: "6class"
    classes: []

training:
  channels: ["RFP1", "Halo"]
  architecture: "resnet18"
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.001
  num_workers: 4

evaluation:
  metrics:
    - accuracy
    - confusion_matrix
    - temporal_analysis

notes: |
  Any additional notes
EOF

# Create experiments index
cat > experiments/EXPERIMENTS_INDEX.md << 'EOF'
# Experiments Index

Track all experiments here.

## Active

| Date | Name | Status | RFP1 | Halo | Notes |
|------|------|--------|------|------|-------|
| | | | | | |

## Completed

### [Experiment Name]
- **Date**: 
- **Goal**: 
- **Result**: 
- **Key Finding**: 
- **Branch**: 

## Planned

- [ ] 
- [ ] 

## Abandoned

- 
EOF

# Make scripts executable
chmod +x scripts/*.sh 2>/dev/null || true

# Create .gitignore additions
cat >> .gitignore << 'EOF'

# Experiment data (already in .gitignore but being explicit)
experiments/*/dataset/images/
experiments/*/results/*.ckpt
experiments/*/*.h5
experiments/*/*.hdf5

# Keep these experiment files
!experiments/*/config.yaml
!experiments/*/README.md
!experiments/*/metadata.csv
!experiments/*/logs/*.log
!experiments/*/evaluation/**/*.png
!experiments/*/evaluation/**/*.json
!experiments/*/analysis/**/*.png
!experiments/*/analysis/**/*.json
EOF

echo "✅ Directory structure created"
echo "✅ Config template installed"
echo "✅ Index file created"
echo ""

echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Your structure:"
echo "  experiments/          - All experiments go here"
echo "  comparison_reports/   - Cross-experiment comparisons"
echo "  config/              - Config templates"
echo "  scripts/             - Helper scripts"
echo ""
echo "Next steps:"
echo "  1. Copy helper scripts:"
echo "     cp /path/to/new_experiment.sh scripts/"
echo "     cp /path/to/compare_experiments.py scripts/"
echo ""
echo "  2. Create your first experiment:"
echo "     ./scripts/new_experiment.sh my_first_exp 'Testing the pipeline'"
echo ""
echo "  3. Read the guide:"
echo "     cat EXPERIMENT_MANAGEMENT.md"
echo ""
echo "  4. Move your existing results:"
echo "     mv A2screen_dataset_6class experiments/2024-11-04_6class_baseline/dataset"
echo "     mv results_6class experiments/2024-11-04_6class_baseline/results"
echo "     mv evaluation_6class experiments/2024-11-04_6class_baseline/evaluation"
echo ""
echo "Happy experimenting! 🚀"
