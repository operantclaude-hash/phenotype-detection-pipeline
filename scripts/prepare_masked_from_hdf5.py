#!/usr/bin/env python3
"""
Extract RFP1, Halo, and Masks from HDF5, then create masked images
Channel 0: RFP1
Channel 1: Halo
Channel 2: Mask (neuron intensity = neuron number)
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import re

def expand_mask(mask, expansion_percent=15):
    """Expand mask by 15% using morphological dilation"""
    if mask.max() == 0:
        return mask
    
    mask_size = np.sum(mask > 0)
    expansion_pixels = int(np.sqrt(mask_size) * (expansion_percent / 100))
    expansion_pixels = max(1, expansion_pixels)
    
    y, x = np.ogrid[-expansion_pixels:expansion_pixels+1, -expansion_pixels:expansion_pixels+1]
    circle = x**2 + y**2 <= expansion_pixels**2
    
    expanded = ndimage.binary_dilation(mask > 0, structure=circle)
    return expanded.astype(np.uint8) * 255

def apply_neuron_mask(image, mask, neuron_num):
    """Apply neuron-specific mask (expand 15%, zero outside)"""
    neuron_mask = (mask == neuron_num)
    
    if not neuron_mask.any():
        return np.zeros_like(image)
    
    expanded_mask = expand_mask(neuron_mask, expansion_percent=15)
    masked_image = image.copy()
    masked_image[~expanded_mask.astype(bool)] = 0
    
    return masked_image

def normalize_image(img):
    """Normalize with 0.01/99.99 percentile - minimal clipping"""
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, [0.01, 99.99])
    img = np.clip(img, p_low, p_high)
    if p_high > p_low:
        img = (img - p_low) / (p_high - p_low) * 255
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)

def process_hdf5_with_masks(hdf5_dir, output_dir, wells):
    """Extract and apply masks from HDF5"""
    output_dir = Path(output_dir)
    (output_dir / 'images' / 'RFP1').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'Halo').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'RFP1_masked').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'Halo_masked').mkdir(parents=True, exist_ok=True)
    
    metadata_list = []
    
    # Get HDF5 files from well subdirectories
    hdf5_files = []
    for well in wells:
        well_dir = Path(hdf5_dir) / well
        if well_dir.exists():
            hdf5_files.extend(well_dir.glob('*.h5'))
    
    print(f"Found {len(hdf5_files)} HDF5 files in {len(wells)} wells")
    
    for hdf5_path in tqdm(hdf5_files):
        match = re.match(r'A2screenPlate2_([A-P]\d+)_(\d+)_(\d+)\.h5', hdf5_path.name)
        if not match:
            continue
        
        well = match.group(1)
        tile = int(match.group(2))
        neuron_num = int(match.group(3))
        
        row = well[0]
        condition = 'Control' if row in ['A', 'B'] else 'APPV717I'
        stimulation = '10000ms' if tile in [2, 3] else '0ms'
        neuron_id = f"{well}_tile{tile}_n{neuron_num}"
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'images' not in f:
                    continue
                
                images = f['images'][:]
                
                if images.shape[0] < 16 or images.shape[1] < 3:
                    continue
                
                for t in range(16):
                    timepoint = f"T{t}"
                    
                    rfp1 = images[t, 0, :, :]
                    halo = images[t, 1, :, :]
                    mask = images[t, 2, :, :]
                    
                    rfp1_norm = normalize_image(rfp1)
                    halo_norm = normalize_image(halo)
                    
                    rfp1_masked = apply_neuron_mask(rfp1_norm, mask, neuron_num)
                    halo_masked = apply_neuron_mask(halo_norm, mask, neuron_num)
                    
                    rfp1_path = output_dir / 'images' / 'RFP1' / f"{neuron_id}_{timepoint}.png"
                    halo_path = output_dir / 'images' / 'Halo' / f"{neuron_id}_{timepoint}.png"
                    rfp1_masked_path = output_dir / 'images' / 'RFP1_masked' / f"{neuron_id}_{timepoint}.png"
                    halo_masked_path = output_dir / 'images' / 'Halo_masked' / f"{neuron_id}_{timepoint}.png"
                    
                    Image.fromarray(rfp1_norm).save(rfp1_path)
                    Image.fromarray(halo_norm).save(halo_path)
                    Image.fromarray(rfp1_masked).save(rfp1_masked_path)
                    Image.fromarray(halo_masked).save(halo_masked_path)
                    
                    metadata_list.append({
                        'neuron_id': neuron_id,
                        'well': well,
                        'tile': tile,
                        'condition': condition,
                        'stimulation': stimulation,
                        'timepoint': timepoint,
                        'timepoint_idx': t,
                        'rfp1_path': f"images/RFP1/{neuron_id}_{timepoint}.png",
                        'halo_path': f"images/Halo/{neuron_id}_{timepoint}.png",
                        'rfp1_masked_path': f"images/RFP1_masked/{neuron_id}_{timepoint}.png",
                        'halo_masked_path': f"images/Halo_masked/{neuron_id}_{timepoint}.png"
                    })
        
        except Exception as e:
            continue
    
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"MASKED DATASET CREATED")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"Neurons: {df['neuron_id'].nunique()}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--wells', nargs='+', required=True)
    
    args = parser.parse_args()
    process_hdf5_with_masks(args.hdf5_dir, args.output_dir, args.wells)
