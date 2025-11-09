#!/usr/bin/env python3
"""
Extract all timepoints T0-T15 for DMSO controls
"""
import sys
sys.path.insert(0, 'src')

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import re

def extract_timecourse(hdf5_dir, output_dir, wells):
    """Extract all 16 timepoints for specified wells"""
    output_dir = Path(output_dir)
    (output_dir / 'images' / 'RFP1').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'Halo').mkdir(parents=True, exist_ok=True)
    
    # Get HDF5 files from well subdirectories
    hdf5_files = []
    for well in wells:
        well_dir = Path(hdf5_dir) / well
        if well_dir.exists():
            hdf5_files.extend(well_dir.glob('*.h5'))
    
    print(f"Found {len(hdf5_files)} HDF5 files in {len(wells)} wells")
    
    metadata_list = []
    
    for hdf5_path in tqdm(hdf5_files):
        # Parse filename
        match = re.match(r'A2screenPlate2_([A-P]\d+)_(\d+)_(\d+)\.h5', hdf5_path.name)
        if not match:
            continue
        
        well = match.group(1)
        tile = int(match.group(2))
        neuron_num = int(match.group(3))
        
        # Determine metadata
        row = well[0]
        condition = 'Control' if row in ['A', 'B'] else 'APPV717I'
        stimulation = '10000ms' if tile in [2, 3] else '0ms'
        
        neuron_id = f"{well}_tile{tile}_n{neuron_num}"
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'images' not in f:
                    continue
                
                images = f['images'][:]
                
                if images.shape[0] < 16:
                    continue
                
                # Extract all timepoints T0-T15
                for t in range(16):
                    timepoint = f"T{t}"
                    
                    # RFP1
                    rfp1_img = images[t, 0, :, :]
                    rfp1_path = output_dir / 'images' / 'RFP1' / f"{neuron_id}_{timepoint}.png"
                    Image.fromarray(normalize_image(rfp1_img)).save(rfp1_path)
                    
                    # Halo
                    halo_img = images[t, 1, :, :]
                    halo_path = output_dir / 'images' / 'Halo' / f"{neuron_id}_{timepoint}.png"
                    Image.fromarray(normalize_image(halo_img)).save(halo_path)
                    
                    # Metadata
                    metadata_list.append({
                        'neuron_id': neuron_id,
                        'well': well,
                        'tile': tile,
                        'condition': condition,
                        'stimulation': stimulation,
                        'timepoint': timepoint,
                        'timepoint_idx': t,
                        'rfp1_path': f"images/RFP1/{neuron_id}_{timepoint}.png",
                        'halo_path': f"images/Halo/{neuron_id}_{timepoint}.png"
                    })
        
        except Exception as e:
            continue
    
    # Save metadata
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"TIME COURSE EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"Neurons: {df['neuron_id'].nunique()}")
    print(f"Wells: {sorted(df['well'].unique())}")
    print(f"Timepoints: {sorted(df['timepoint'].unique())}")
    print(f"{'='*70}")

def normalize_image(img):
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, [1, 99])
    img = np.clip(img, p_low, p_high)
    if p_high > p_low:
        img = (img - p_low) / (p_high - p_low) * 255
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--wells', nargs='+', required=True)
    
    args = parser.parse_args()
    extract_timecourse(args.hdf5_dir, args.output_dir, args.wells)
