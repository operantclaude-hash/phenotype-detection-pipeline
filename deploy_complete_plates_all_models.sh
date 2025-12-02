#!/bin/bash
################################################################################
# Deploy All Models to Complete Plates - CORRECT NORMALIZATION
################################################################################
# This script deploys all 4 model types to complete plates (all wells, cols 1-24)
# ALL models use IDENTICAL normalization:
#   1. Percentile clipping [p1, p99]
#   2. Normalize to [0, 1]
#   3. ImageNet normalization
#
# Model types:
#   1. RESCUE @ 0ms (3-channel): APP vs WT at baseline T0
#   2. RESCUE @ 10000ms (3-channel): APP vs WT post-stimulation T15
#   3. APP Treatment (6-channel): APP_T0 vs APP_T15
#   4. WT Treatment (6-channel): WT_T0 vs WT_T15
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "DEPLOYING ALL MODELS TO COMPLETE PLATES - CORRECT NORMALIZATION"
echo "================================================================================"
echo ""
echo "Started: $(date)"
echo ""

# Output directory
OUTPUT_DIR="A2screen_combined/predictions_complete_plates_CORRECT_NORM"
mkdir -p "$OUTPUT_DIR"

################################################################################
# 1. RESCUE @ 0ms (APP vs WT at baseline T0)
################################################################################
echo "================================================================================"
echo "1. RESCUE @ 0ms: APP vs WT at baseline (single timepoint T0)"
echo "================================================================================"
echo ""

# Use best checkpoint from replicate 01
RESCUE_0MS_CKPT="A2screen_combined/models/rescue_0ms_replicates_CORRECT_NORM/replicate_01/checkpoints/best*.ckpt"
RESCUE_0MS_CKPT=$(ls $RESCUE_0MS_CKPT 2>/dev/null | head -1)

if [ -z "$RESCUE_0MS_CKPT" ]; then
    echo "⚠ RESCUE @ 0ms checkpoint not found. Skipping..."
else
    echo "Checkpoint: $RESCUE_0MS_CKPT"

    # TODO: Create metadata CSV for complete plate @ 0ms
    # This would include ALL wells from columns 1-24 at T0
    echo "TODO: Create complete plate metadata for 0ms (T0 timepoint)"
    echo "  Should include: plate1-T0 and plate2-T0, all wells, cols 1-24"
fi

echo ""

################################################################################
# 2. RESCUE @ 10000ms (APP vs WT post-stimulation T15)
################################################################################
echo "================================================================================"
echo "2. RESCUE @ 10000ms: APP vs WT post-stimulation (single timepoint T15)"
echo "================================================================================"
echo ""

# Use best checkpoint from replicate 01
RESCUE_10000MS_CKPT="A2screen_combined/models/rescue_10000ms_replicates_CORRECT_NORM/replicate_01/checkpoints/best*.ckpt"
RESCUE_10000MS_CKPT=$(ls $RESCUE_10000MS_CKPT 2>/dev/null | head -1)

if [ -z "$RESCUE_10000MS_CKPT" ]; then
    echo "⚠ RESCUE @ 10000ms checkpoint not found. Skipping..."
else
    echo "Checkpoint: $RESCUE_10000MS_CKPT"

    # TODO: Create metadata CSV for complete plate @ 10000ms
    # This would include ALL wells from columns 1-24 at T15
    echo "TODO: Create complete plate metadata for 10000ms (T15 timepoint)"
    echo "  Should include: plate1-T14 and plate2-T15, all wells, cols 1-24"
fi

echo ""

################################################################################
# 3. APP Treatment (APP_T0 vs APP_T15)
################################################################################
echo "================================================================================"
echo "3. APP Treatment: Temporal change detection in APP neurons (T0 vs T15)"
echo "================================================================================"
echo ""

# Use best checkpoint from replicate 01
APP_TREATMENT_CKPT="A2screen_combined/models/app_treatment_binary_FIXED_MASK/checkpoints/best*.ckpt"
APP_TREATMENT_CKPT=$(ls $APP_TREATMENT_CKPT 2>/dev/null | head -1)

if [ -z "$APP_TREATMENT_CKPT" ]; then
    echo "⚠ APP Treatment checkpoint not found. Skipping..."
else
    echo "Checkpoint: $APP_TREATMENT_CKPT"

    # TODO: Create metadata CSV for complete plate APP neurons (T0 + T15)
    # This would include ALL APP wells from columns 1-24 with both T0 and T15
    echo "TODO: Create complete plate metadata for APP treatment"
    echo "  Should include: APP wells only, both T0 and T15, cols 1-24"
fi

echo ""

################################################################################
# 4. WT Treatment (WT_T0 vs WT_T15)
################################################################################
echo "================================================================================"
echo "4. WT Treatment: Temporal change detection in WT neurons (T0 vs T15)"
echo "================================================================================"
echo ""

# Use best checkpoint from replicate 01
WT_TREATMENT_CKPT="A2screen_combined/models/wt_treatment_binary_FIXED_MASK/checkpoints/best*.ckpt"
WT_TREATMENT_CKPT=$(ls $WT_TREATMENT_CKPT 2>/dev/null | head -1)

if [ -z "$WT_TREATMENT_CKPT" ]; then
    echo "⚠ WT Treatment checkpoint not found. Skipping..."
else
    echo "Checkpoint: $WT_TREATMENT_CKPT"

    # TODO: Create metadata CSV for complete plate WT neurons (T0 + T15)
    # This would include ALL WT wells from columns 1-24 with both T0 and T15
    echo "TODO: Create complete plate metadata for WT treatment"
    echo "  Should include: WT wells only, both T0 and T15, cols 1-24"
fi

echo ""

################################################################################
# Summary
################################################################################
echo "================================================================================"
echo "DEPLOYMENT STATUS"
echo "================================================================================"
echo ""
echo "Finished: $(date)"
echo ""
echo "Next steps:"
echo "  1. Wait for RESCUE model retraining to complete"
echo "  2. Create metadata CSVs for complete plates (all wells, cols 1-24)"
echo "  3. Run this script again to deploy all models"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "================================================================================"
