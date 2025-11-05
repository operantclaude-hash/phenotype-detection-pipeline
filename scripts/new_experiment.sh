#!/bin/bash
# Create a new experiment with proper structure
# Usage: ./new_experiment.sh experiment_name "description"

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <description>"
    echo "Example: $0 binary_app_vs_wt 'Binary classification: APP vs WT at T15 10000ms'"
    exit 1
fi

EXPERIMENT_NAME=$1
DESCRIPTION=$2
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR="experiments/${DATE}_${EXPERIMENT_NAME}"

echo "=========================================="
echo "Creating New Experiment"
echo "=========================================="
echo "Name: $EXPERIMENT_NAME"
echo "Date: $DATE"
echo "Directory: $EXPERIMENT_DIR"
echo ""

# Create directory structure
mkdir -p "$EXPERIMENT_DIR"/{dataset,results,evaluation,analysis,logs}

# Create experiment README
cat > "$EXPERIMENT_DIR/README.md" << EOF
# Experiment: ${EXPERIMENT_NAME}

**Date**: ${DATE}
**Status**: In Progress

## Description

${DESCRIPTION}

## Hypothesis

[Your hypothesis here]

## Methods

### Data Preparation
\`\`\`bash
# Command used
python src/prepare_dataset.py \\
    --hdf5_dir /path/to/data \\
    --output_dir ${EXPERIMENT_DIR}/dataset \\
    --bad_wells [wells]
\`\`\`

### Training
\`\`\`bash
# Command used
python src/train_models.py \\
    --metadata ${EXPERIMENT_DIR}/dataset/metadata.csv \\
    --root_dir ${EXPERIMENT_DIR}/dataset/ \\
    --output_dir ${EXPERIMENT_DIR}/results/ \\
    --tasks [task] \\
    --channels [channels]
\`\`\`

### Evaluation
\`\`\`bash
# Commands used
[To be filled]
\`\`\`

## Results

### Training Results
- **RFP1**: [accuracy]
- **Halo**: [accuracy]

### Key Findings
- [Finding 1]
- [Finding 2]

### Figures
- \`analysis/confusion_matrix_rfp1.png\`
- \`analysis/confusion_matrix_halo.png\`
- \`analysis/temporal_analysis.png\`

## Conclusions

[Your conclusions]

## Next Steps

- [ ] [Next experiment or analysis]
- [ ] [Follow-up questions]

## Files

\`\`\`
${EXPERIMENT_DIR}/
├── dataset/           # Prepared dataset
├── results/           # Model checkpoints
├── evaluation/        # Evaluation outputs
├── analysis/          # Final analysis
├── logs/             # Training logs
└── README.md         # This file
\`\`\`

## Notes

[Any additional notes]
EOF

# Copy config template
cp config/experiment_template.yaml "$EXPERIMENT_DIR/config.yaml"

# Create .gitignore for this experiment
cat > "$EXPERIMENT_DIR/.gitignore" << EOF
# Don't commit large files
dataset/images/
*.h5
*.hdf5
*.ckpt
*.pth

# Keep these
!dataset/metadata.csv
!config.yaml
!README.md
EOF

# Create experiment log
cat > "$EXPERIMENT_DIR/experiment_log.txt" << EOF
Experiment Log: ${EXPERIMENT_NAME}
Created: ${DATE}
=====================================

$(date): Experiment created

EOF

echo "✅ Experiment directory created: $EXPERIMENT_DIR"
echo ""
echo "Next steps:"
echo "1. Edit $EXPERIMENT_DIR/config.yaml"
echo "2. Update $EXPERIMENT_DIR/README.md with your hypothesis"
echo "3. Run your preparation/training commands"
echo "4. Document results in README.md"
echo ""
echo "Create git branch:"
echo "  git checkout -b experiment/${EXPERIMENT_NAME}"
