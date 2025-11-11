# Archive Directory

This directory contains experimental and deprecated code that is no longer part of the production pipeline.

## Purpose

As the project matured, we identified a core set of production scripts for the phenotype detection pipeline. All experimental, analysis, and legacy code has been archived here to maintain a clean and focused production codebase.

## Contents

### old_scripts/
Contains 19 Python scripts that were used during development, experimentation, and analysis phases:

- **Rescue experiments** (legacy approach):
  - `train_rescue_multitimepoint.py`
  - `train_rescue_dualchannel.py`
  - `train_rescue_masked.py`
  - `deploy_both_rescue_models.py`
  - `deploy_rescue_models_full.py`

- **Deployment scripts** (replaced by production versions):
  - `deploy_and_score.py`
  - `deploy_and_score_ALL.py`
  - `deploy_mixed_models.py`
  - `deploy_dualchannel.py`

- **Analysis and evaluation**:
  - `compare_experiments.py`
  - `evaluate_binary.py`
  - `filter_binary.py`
  - `analyze_per_timepoint.py`
  - `analyze_channels_per_timepoint.py`
  - `analyze_single_channel_timepoint.py`

- **Ablation studies**:
  - `test_channel_ablation.py`
  - `test_threechannel_ablation.py`

- **Legacy training**:
  - `train_threechannel.py` (replaced by `train_threechannel_improved.py`)

- **Dataset preparation variants**:
  - `create_reversion_datasets_controlled.py` (replaced by v2 and stratified versions)

## Why Archive?

These scripts were valuable during development but are no longer needed for production workflows. They have been archived rather than deleted to:

1. Preserve project history
2. Allow reference to experimental approaches
3. Enable reproduction of earlier results if needed
4. Maintain git history

## Production Scripts

For current production scripts, see `scripts/README_PRODUCTION.md`

---

*Last updated: 2025-11-11*
