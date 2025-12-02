#!/bin/bash
# Train 10 replicate models each for APP and WT treatment using only last timepoints

echo "================================================================================"
echo "TRAINING LAST-TIMEPOINT REPLICATES: 10 APP + 10 WT = 20 MODELS"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - 0ms: plate1 T14 only"
echo "  - 10000ms: plate2 T15 only"
echo "  - 10 replicates per model (different random seeds)"
echo "  - Total: 20 models"
echo ""
echo "Started: $(date)"
echo "================================================================================"
echo ""

# Train APP replicates
echo "STAGE 1: Training APP treatment replicates (1-10)..."
echo "================================================================================"
for i in {1..10}; do
    echo ""
    echo ">>> Starting APP replicate $i/10 at $(date)"
    python3 scripts/train_app_last_timepoint_replicates.py $i
    if [ $? -eq 0 ]; then
        echo "✓ APP replicate $i completed successfully"
    else
        echo "✗ APP replicate $i FAILED"
        exit 1
    fi
done

echo ""
echo "================================================================================"
echo "APP TREATMENT REPLICATES COMPLETE - Moving to WT treatment"
echo "================================================================================"
echo ""

# Train WT replicates
echo "STAGE 2: Training WT treatment replicates (1-10)..."
echo "================================================================================"
for i in {1..10}; do
    echo ""
    echo ">>> Starting WT replicate $i/10 at $(date)"
    python3 scripts/train_wt_last_timepoint_replicates.py $i
    if [ $? -eq 0 ]; then
        echo "✓ WT replicate $i completed successfully"
    else
        echo "✗ WT replicate $i FAILED"
        exit 1
    fi
done

echo ""
echo "================================================================================"
echo "ALL TRAINING COMPLETE"
echo "================================================================================"
echo "Finished: $(date)"
echo ""
echo "Results:"
echo "  - APP replicates: A2screen_combined/models/app_treatment_LAST_TIMEPOINT_replicates/"
echo "  - WT replicates: A2screen_combined/models/wt_treatment_LAST_TIMEPOINT_replicates/"
echo ""
echo "Next steps: Run analysis script to compare performance"
echo "================================================================================"
