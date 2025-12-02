#!/bin/bash
# Train 20 RESCUE model replicates with CORRECT normalization
# 10 replicates @ 0ms (baseline) + 10 replicates @ 10000ms (post-stim)

echo "================================================================================"
echo "TRAINING RESCUE MODELS - CORRECT NORMALIZATION"
echo "================================================================================"
echo ""
echo "Using EXACT same normalization as 6-channel models:"
echo "  1. Percentile clipping [p1, p99]"
echo "  2. Normalize to [0, 1]"
echo "  3. ImageNet normalization"
echo ""
echo "Total: 20 models (10 @ 0ms + 10 @ 10000ms)"
echo "================================================================================"
echo ""
echo "Started: $(date)"
echo ""

# Stage 1: Train 10 replicates @ 0ms
echo "================================================================================"
echo "STAGE 1: RESCUE @ 0ms (APP vs WT at baseline T0)"
echo "================================================================================"

for i in {1..10}; do
    SEED=$((42 + i))
    OUTPUT_DIR="A2screen_combined/models/rescue_0ms_replicates_CORRECT_NORM/replicate_$(printf '%02d' $i)"

    echo ""
    echo ">>> Training replicate $i/10 @ 0ms (seed=$SEED)"

    python3 scripts/train_rescue_3channel_CORRECT_NORM.py \
        --data_dir A2screen_combined/data/rescue_3channel_0ms_SIZE_MATCHED \
        --output_dir "$OUTPUT_DIR" \
        --epochs 100 \
        --batch_size 32 \
        --seed $SEED

    if [ $? -eq 0 ]; then
        echo "✓ Replicate $i @ 0ms completed"
    else
        echo "✗ Replicate $i @ 0ms FAILED"
        exit 1
    fi
done

echo ""
echo "================================================================================"
echo "STAGE 2: RESCUE @ 10000ms (APP vs WT post-stimulation T15)"
echo "================================================================================"

for i in {1..10}; do
    SEED=$((42 + i))
    OUTPUT_DIR="A2screen_combined/models/rescue_10000ms_replicates_CORRECT_NORM/replicate_$(printf '%02d' $i)"

    echo ""
    echo ">>> Training replicate $i/10 @ 10000ms (seed=$SEED)"

    python3 scripts/train_rescue_3channel_CORRECT_NORM.py \
        --data_dir A2screen_combined/data/rescue_3channel_10000ms_SIZE_MATCHED \
        --output_dir "$OUTPUT_DIR" \
        --epochs 100 \
        --batch_size 32 \
        --seed $SEED

    if [ $? -eq 0 ]; then
        echo "✓ Replicate $i @ 10000ms completed"
    else
        echo "✗ Replicate $i @ 10000ms FAILED"
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
echo "  - 0ms replicates: A2screen_combined/models/rescue_0ms_replicates_CORRECT_NORM/"
echo "  - 10000ms replicates: A2screen_combined/models/rescue_10000ms_replicates_CORRECT_NORM/"
echo ""
echo "================================================================================"
