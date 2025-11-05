# Experiment: app_vs_wt_t15_10000ms

**Date**: 2025-11-05
**Status**: In Progress

## Description

Binary: APP vs WT at T15 with 10000ms stimulation - for drug rescue scoring

## Hypothesis

[Your hypothesis here]

## Methods

### Data Preparation
```bash
# Command used
python src/prepare_dataset.py \
    --hdf5_dir /path/to/data \
    --output_dir experiments/2025-11-05_app_vs_wt_t15_10000ms/dataset \
    --bad_wells [wells]
```

### Training
```bash
# Command used
python src/train_models.py \
    --metadata experiments/2025-11-05_app_vs_wt_t15_10000ms/dataset/metadata.csv \
    --root_dir experiments/2025-11-05_app_vs_wt_t15_10000ms/dataset/ \
    --output_dir experiments/2025-11-05_app_vs_wt_t15_10000ms/results/ \
    --tasks [task] \
    --channels [channels]
```

### Evaluation
```bash
# Commands used
[To be filled]
```

## Results

### Training Results
- **RFP1**: [accuracy]
- **Halo**: [accuracy]

### Key Findings
- [Finding 1]
- [Finding 2]

### Figures
- `analysis/confusion_matrix_rfp1.png`
- `analysis/confusion_matrix_halo.png`
- `analysis/temporal_analysis.png`

## Conclusions

[Your conclusions]

## Next Steps

- [ ] [Next experiment or analysis]
- [ ] [Follow-up questions]

## Files

```
experiments/2025-11-05_app_vs_wt_t15_10000ms/
├── dataset/           # Prepared dataset
├── results/           # Model checkpoints
├── evaluation/        # Evaluation outputs
├── analysis/          # Final analysis
├── logs/             # Training logs
└── README.md         # This file
```

## Notes

[Any additional notes]
