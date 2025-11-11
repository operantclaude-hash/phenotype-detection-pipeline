#!/usr/bin/env python3
"""Demonstrate the stratification bug in train_lightning_threechannel_improved.py"""

import pandas as pd
import numpy as np

# Create sample metadata
metadata = pd.DataFrame({
    'well': ['C3', 'A1', 'B2', 'C3', 'A1', 'B2', 'D4', 'D4'],
    'class_label': ['classA', 'classB', 'classA', 'classA', 'classB', 'classA', 'classB', 'classB']
})

print("Sample metadata:")
print(metadata)
print()

# Replicate the code from train_lightning_threechannel_improved.py
unique_wells = metadata['well'].unique()
well_labels = metadata.groupby('well')['class_label'].first()

print(f"unique_wells (numpy array): {unique_wells}")
print(f"Order of wells in unique_wells: {list(unique_wells)}")
print()

print(f"well_labels (pandas Series):")
print(well_labels)
print(f"Index of well_labels: {list(well_labels.index)}")
print(f"Values of well_labels: {list(well_labels.values)}")
print()

print("=" * 70)
print("THE BUG:")
print("=" * 70)
print("When passed to train_test_split, well_labels will be converted to an array")
print("using well_labels.values, which gives values in the order of the Series index.")
print()
print(f"well_labels.values = {well_labels.values}")
print(f"well_labels.index  = {list(well_labels.index)}")
print()
print("But unique_wells has a different order:")
print(f"unique_wells = {unique_wells}")
print()
print("So the stratification labels will be MISALIGNED with the wells!")
print()
print("For example:")
for i, well in enumerate(unique_wells):
    series_idx = list(well_labels.index).index(well)
    print(f"  unique_wells[{i}] = '{well}' -> should have label '{well_labels[well]}'")
    print(f"  but stratify array[{i}] = '{well_labels.values[i]}' (from well '{well_labels.index[i]}')")
    if well_labels[well] != well_labels.values[i]:
        print(f"    ❌ MISALIGNMENT! Expected '{well_labels[well]}' but got '{well_labels.values[i]}'")
    else:
        print(f"    ✓ Happens to match (by luck)")
    print()

print("=" * 70)
print("THE FIX:")
print("=" * 70)
print("Use: stratify=well_labels.loc[unique_wells].values")
print("or:  stratify=[well_labels[w] for w in unique_wells]")
print()
correct_stratify = well_labels.loc[unique_wells].values
print(f"Correct stratify array: {correct_stratify}")
print("This ensures the stratification labels match the order of unique_wells")
