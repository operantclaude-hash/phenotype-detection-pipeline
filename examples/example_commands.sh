#!/bin/bash
# Example commands for the 6-class phenotype detection pipeline
# Copy and modify these for your experiment

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data
HDF5_DIR="/path/to/your/hdf5_neurons"
OUTPUT_DIR="./my_experiment"

# Bad wells (customize for your plate)
BAD_WELLS="A1 A2 A3 A5 A12 A13 A14 A15 B14 B15 B16"

# Training parameters
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.001

# ============================================================================
# STEP 1: PREPARE DATASET
# ============================================================================

echo "Step 1: Preparing dataset..."

python scripts/1_prepare_dataset.py \
    --hdf5_dir "$HDF5_DIR" \
    --output_dir "$OUTPUT_DIR/dataset" \
    --bad_wells $BAD_WELLS

# Add class_label column (required by data module)
python << 'EOF'
import pandas as pd
df = pd.read_csv('my_experiment/dataset/metadata.csv')
df['class_label'] = df['label_6class']
df.to_csv('my_experiment/dataset/metadata.csv', index=False)
print("✅ Added class_label column")
EOF

echo "✅ Dataset ready!"

# ============================================================================
# STEP 2: TRAIN MODELS
# ============================================================================

echo "Step 2: Training models on both channels..."

python scripts/2_train_models.py \
    --metadata "$OUTPUT_DIR/dataset/metadata.csv" \
    --root_dir "$OUTPUT_DIR/dataset/" \
    --output_dir "$OUTPUT_DIR/models" \
    --tasks 6class \
    --channels RFP1 Halo \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --num_workers 4

echo "✅ Training complete!"

# ============================================================================
# STEP 3: EVALUATE MODELS
# ============================================================================

echo "Step 3: Evaluating trained models..."

# Find best checkpoints
BEST_RFP1=$(ls "$OUTPUT_DIR/models/RFP1/6class/checkpoints/best-"*.ckpt | sort -t'=' -k3 -r | head -1)
BEST_HALO=$(ls "$OUTPUT_DIR/models/Halo/6class/checkpoints/best-"*.ckpt | sort -t'=' -k3 -r | head -1)

echo "Using RFP1 checkpoint: $BEST_RFP1"
echo "Using Halo checkpoint: $BEST_HALO"

# Evaluate RFP1
python scripts/3_evaluate_models.py \
    --checkpoint "$BEST_RFP1" \
    --metadata "$OUTPUT_DIR/dataset/metadata.csv" \
    --root_dir "$OUTPUT_DIR/dataset/" \
    --channel RFP1 \
    --output_dir "$OUTPUT_DIR/evaluation/RFP1"

# Evaluate Halo
python scripts/3_evaluate_models.py \
    --checkpoint "$BEST_HALO" \
    --metadata "$OUTPUT_DIR/dataset/metadata.csv" \
    --root_dir "$OUTPUT_DIR/dataset/" \
    --channel Halo \
    --output_dir "$OUTPUT_DIR/evaluation/Halo"

echo "✅ Evaluation complete!"

# ============================================================================
# STEP 4: ANALYZE RESULTS
# ============================================================================

echo "Step 4: Running comprehensive analysis..."

# If you have 8-class results to compare:
# python scripts/4_analyze_results.py \
#     --rfp1_8class "$OUTPUT_DIR/models_8class/RFP1/full/evaluation" \
#     --rfp1_6class "$OUTPUT_DIR/evaluation/RFP1" \
#     --halo_8class "$OUTPUT_DIR/models_8class/Halo/full/evaluation" \
#     --halo_6class "$OUTPUT_DIR/evaluation/Halo" \
#     --metadata "$OUTPUT_DIR/dataset/metadata.csv" \
#     --output_dir "$OUTPUT_DIR/analysis"

# Simple 6-class analysis (without 8-class comparison)
python << 'EOF'
import numpy as np
import pandas as pd

# Load results
rfp1_cm = np.load('my_experiment/evaluation/RFP1/confusion_matrix.npy')
halo_cm = np.load('my_experiment/evaluation/Halo/confusion_matrix.npy')

class_names = [
    'Control_T0', 'APPV717I_T0',
    'Control_0ms_T16', 'Control_10000ms_T16',
    'APPV717I_0ms_T16', 'APPV717I_10000ms_T16'
]

# Calculate accuracies
rfp1_acc = rfp1_cm.diagonal() / rfp1_cm.sum(axis=1)
halo_acc = halo_cm.diagonal() / halo_cm.sum(axis=1)

print("="*70)
print("PER-CLASS ACCURACY")
print("="*70)
print(f"{'Class':<30} {'RFP1':>10} {'Halo':>10} {'Best':>10}")
print("-"*70)
for i, name in enumerate(class_names):
    best = 'RFP1' if rfp1_acc[i] > halo_acc[i] else 'Halo'
    print(f"{name:<30} {rfp1_acc[i]:>10.3f} {halo_acc[i]:>10.3f} {best:>10}")

print("\n" + "="*70)
print("TEMPORAL ANALYSIS")
print("="*70)
rfp1_t0 = rfp1_acc[:2].mean()
rfp1_t16 = rfp1_acc[2:].mean()
halo_t0 = halo_acc[:2].mean()
halo_t16 = halo_acc[2:].mean()

print(f"T0 (pre-stim):   RFP1={rfp1_t0:.3f}, Halo={halo_t0:.3f}")
print(f"T16 (post-stim): RFP1={rfp1_t16:.3f}, Halo={halo_t16:.3f}")
print(f"Improvement:     RFP1={rfp1_t16-rfp1_t0:+.3f}, Halo={halo_t16-halo_t0:+.3f}")

print("\n✅ Analysis complete!")
print(f"View confusion matrices in: {OUTPUT_DIR}/evaluation/")
EOF

echo "✅ All done! Check results in $OUTPUT_DIR/"

# ============================================================================
# QUICK CHECKS
# ============================================================================

echo ""
echo "Quick checks:"
echo "  1. View RFP1 confusion matrix:"
echo "     open $OUTPUT_DIR/evaluation/RFP1/confusion_matrix_normalized.png"
echo ""
echo "  2. View Halo confusion matrix:"
echo "     open $OUTPUT_DIR/evaluation/Halo/confusion_matrix_normalized.png"
echo ""
echo "  3. Check training logs:"
echo "     tensorboard --logdir $OUTPUT_DIR/models/"
echo ""
echo "  4. Review metrics:"
echo "     cat $OUTPUT_DIR/evaluation/RFP1/metrics.json"
echo "     cat $OUTPUT_DIR/evaluation/Halo/metrics.json"
