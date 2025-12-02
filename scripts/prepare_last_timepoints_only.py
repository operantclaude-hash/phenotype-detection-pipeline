#!/usr/bin/env python3
"""
Create datasets for APP and WT treatment using only last timepoints:
- 0ms: plate1 T14 only
- 10000ms: plate2 T15 only
"""

import pandas as pd
import os
from pathlib import Path

# Base directories
base_dir = Path("A2screen_combined/data")
output_dir_app = base_dir / "app_treatment_LAST_TIMEPOINT"
output_dir_wt = base_dir / "wt_treatment_LAST_TIMEPOINT"

# Create output directories
output_dir_app.mkdir(exist_ok=True)
output_dir_wt.mkdir(exist_ok=True)

print("=" * 80)
print("CREATING LAST-TIMEPOINT-ONLY DATASETS")
print("=" * 80)

# Load existing datasets
app_train = pd.read_csv(base_dir / "app_treatment_binary_FIXED_MASK/train.csv")
app_val = pd.read_csv(base_dir / "app_treatment_binary_FIXED_MASK/val.csv")
app_test = pd.read_csv(base_dir / "app_treatment_binary_FIXED_MASK/test.csv")

wt_train = pd.read_csv(base_dir / "wt_treatment_binary_FIXED_MASK/train.csv")
wt_val = pd.read_csv(base_dir / "wt_treatment_binary_FIXED_MASK/val.csv")
wt_test = pd.read_csv(base_dir / "wt_treatment_binary_FIXED_MASK/test.csv")

print(f"\nOriginal APP dataset sizes:")
print(f"  Train: {len(app_train)}, Val: {len(app_val)}, Test: {len(app_test)}")
print(f"\nOriginal WT dataset sizes:")
print(f"  Train: {len(wt_train)}, Val: {len(wt_val)}, Test: {len(wt_test)}")

# Filter function
def filter_last_timepoint(df):
    """Keep only plate1 T14 (0ms) and plate2 T15 (10000ms)"""
    # For 0ms neurons (plate1), keep only T14
    mask_0ms = (df['stimulation'] == '0ms') & (df['timepoint'] == 14)
    # For 10000ms neurons (plate2), keep only T15
    mask_10000ms = (df['stimulation'] == '10000ms') & (df['timepoint'] == 15)

    filtered = df[mask_0ms | mask_10000ms].copy()
    return filtered

# Filter APP datasets
print("\n" + "=" * 80)
print("FILTERING APP TREATMENT DATA")
print("=" * 80)

app_train_filtered = filter_last_timepoint(app_train)
app_val_filtered = filter_last_timepoint(app_val)
app_test_filtered = filter_last_timepoint(app_test)

print(f"\nFiltered APP dataset sizes:")
print(f"  Train: {len(app_train_filtered)} (from {len(app_train)})")
print(f"  Val: {len(app_val_filtered)} (from {len(app_val)})")
print(f"  Test: {len(app_test_filtered)} (from {len(app_test)})")

print(f"\nAPP class distribution (train):")
print(app_train_filtered['class_binary'].value_counts())
print(f"\nAPP stimulation distribution (train):")
print(app_train_filtered['stimulation'].value_counts())

# Save APP datasets
app_train_filtered.to_csv(output_dir_app / "train.csv", index=False)
app_val_filtered.to_csv(output_dir_app / "val.csv", index=False)
app_test_filtered.to_csv(output_dir_app / "test.csv", index=False)

# Filter WT datasets
print("\n" + "=" * 80)
print("FILTERING WT TREATMENT DATA")
print("=" * 80)

wt_train_filtered = filter_last_timepoint(wt_train)
wt_val_filtered = filter_last_timepoint(wt_val)
wt_test_filtered = filter_last_timepoint(wt_test)

print(f"\nFiltered WT dataset sizes:")
print(f"  Train: {len(wt_train_filtered)} (from {len(wt_train)})")
print(f"  Val: {len(wt_val_filtered)} (from {len(wt_val)})")
print(f"  Test: {len(wt_test_filtered)} (from {len(wt_test)})")

print(f"\nWT class distribution (train):")
print(wt_train_filtered['class_binary'].value_counts())
print(f"\nWT stimulation distribution (train):")
print(wt_train_filtered['stimulation'].value_counts())

# Save WT datasets
wt_train_filtered.to_csv(output_dir_wt / "train.csv", index=False)
wt_val_filtered.to_csv(output_dir_wt / "val.csv", index=False)
wt_test_filtered.to_csv(output_dir_wt / "test.csv", index=False)

print("\n" + "=" * 80)
print("DATASET CREATION COMPLETE")
print("=" * 80)
print(f"\nAPP datasets saved to: {output_dir_app}")
print(f"WT datasets saved to: {output_dir_wt}")
