#!/usr/bin/env python3
"""Filter dataset for binary classification"""
import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: filter_binary.py <dataset_dir>")
    sys.exit(1)

dataset_dir = Path(sys.argv[1])
metadata_file = dataset_dir / 'metadata.csv'

print(f"Loading {metadata_file}...")
df = pd.read_csv(metadata_file)
print(f"Original samples: {len(df)}")

# Filter: only T15 (or T16) + 10000ms stimulation
df = df[(df['timepoint'] == 'T15') & (df['stimulation'] == '10000ms')]

# Create binary labels
df['label_binary'] = df['cell_line']
df['class_label'] = df['label_binary']

# Save
df.to_csv(metadata_file, index=False)

print(f"Binary samples: {len(df)}")
print("\nClass distribution:")
print(df['class_label'].value_counts())
