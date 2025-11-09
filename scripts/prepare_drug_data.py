#!/usr/bin/env python3
"""
Process drug-treated wells for model deployment
Just a wrapper around prepare_dataset.py
"""
import subprocess
import sys
import pandas as pd
from pathlib import Path

def prepare_drug_wells(hdf5_dir, output_dir, bad_wells):
    """
    Process ALL wells except bad_wells
    Calls prepare_dataset.py directly
    """
    # Build command
    cmd = [
        'python', 'src/prepare_dataset.py',
        '--hdf5_dir', hdf5_dir,
        '--output_dir', output_dir,
        '--bad_wells'
    ] + bad_wells
    
    print("="*70)
    print("PROCESSING DRUG WELLS")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print("This will take ~4 hours...")
    print("="*70)
    
    # Run
    result = subprocess.run(cmd, check=True)
    
    # Verify
    metadata_file = Path(output_dir) / 'metadata.csv'
    if metadata_file.exists():
        df = pd.read_csv(metadata_file)
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Total samples: {len(df)}")
        print(f"Wells: {sorted(df['well'].unique())}")
        print(f"Timepoints: {sorted(df['timepoint'].unique())}")
        print(f"Output: {metadata_file}")
        print("="*70)
    
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bad_wells', nargs='+', required=True)
    
    args = parser.parse_args()
    
    prepare_drug_wells(args.hdf5_dir, args.output_dir, args.bad_wells)
