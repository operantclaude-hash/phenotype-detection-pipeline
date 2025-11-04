#!/usr/bin/env python3
"""
Extract HDF5 cropped neurons for A2screenPlate2 dataset
Filters for neurons with both T0 and T15 timepoints (16 total)
Creates datasets for multiple classification tasks
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import json


class A2ScreenDatasetBuilder:
    """
    Extract single-neuron HDF5 files and organize for training
    
    Expected HDF5 structure per neuron file:
    - 'images': shape (T, C, H, W) where C=2 (RFP1, Halo)
    - timepoints: should have at least 16 (indices 0-15)
    """
    
    def __init__(self, hdf5_dir, output_dir, bad_wells=None, metadata_csv=None):
        """
        Args:
            hdf5_dir: Directory containing HDF5 files
            output_dir: Output directory for extracted images
            bad_wells: List of well IDs to exclude (e.g., ['B03', 'C04'])
            metadata_csv: Optional CSV with metadata
        """
        self.hdf5_dir = Path(hdf5_dir)
        self.output_dir = Path(output_dir)
        self.bad_wells = set(bad_wells) if bad_wells else set()
        
        # Load external metadata if provided
        self.external_metadata = None
        if metadata_csv:
            self.external_metadata = pd.read_csv(metadata_csv)
            print(f"Loaded external metadata: {len(self.external_metadata)} entries")
        
        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'RFP1').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'Halo').mkdir(parents=True, exist_ok=True)
        
        self.metadata = []
        
    def parse_metadata_from_hdf5(self, hdf5_path):
        """
        Extract experimental metadata from HDF5 filename
        
        Parses filename format: A2screenPlate2_{WELL}_{TILE}_{NEURON}.h5
        Example: A2screenPlate2_B12_2_470.h5
        
        Condition mapping:
        - A* or B* → Control
        - C* or D* → APPV717I
        
        Only includes wells: A1-A24, B1-B24, C1-C24, D1-D24 (all DMSO)
        
        Tile mapping:
        - Tile 1 or 4 → 0ms (no stimulation)
        - Tile 2 or 3 → 10000ms stimulation
        """
        import re
        
        try:
            # Parse filename: A2screenPlate2_WELL_TILE_NEURON.h5
            filename = hdf5_path.name
            match = re.match(r'A2screenPlate2_([A-Z]\d+)_(\d)_(\d+)', 
                           filename.replace('.h5', '').replace('.hdf5', ''))
            
            if not match:
                return None
            
            well = match.group(1)   # e.g., B12
            tile = int(match.group(2))  # e.g., 2
            neuron_id = int(match.group(3))  # e.g., 470
            
            # Check if well is in valid range (A-D rows, 1-24 columns)
            row = well[0]
            if row not in ['A', 'B', 'C', 'D']:
                return None  # Skip wells outside A-D rows
            
            col_match = re.match(r'[A-Z](\d+)', well)
            if col_match:
                col = int(col_match.group(1))
                if col < 1 or col > 24:
                    return None  # Skip columns outside 1-24
            
            # Determine condition from well row
            if row in ['A', 'B']:
                condition = 'Control'
            elif row in ['C', 'D']:
                condition = 'APPV717I'
            else:
                return None
            
            # All A-D rows are DMSO for this experiment
            name = 'DMSO'
            
            # Determine stimulation from tile
            if tile in [2, 3]:
                stimulation = '10000ms'
            elif tile in [1, 4]:
                stimulation = '0ms'
            else:
                return None
            
            return {
                'condition': condition,
                'name': name,
                'well': well,
                'tile': tile,
                'neuron_id': neuron_id,
                'stimulation': stimulation
            }
            
        except Exception as e:
            return None
    
    def normalize_image(self, img):
        """Normalize image to 0-255 uint8 range"""
        if img.dtype == np.uint8:
            return img
        
        # Handle different bit depths
        img = img.astype(np.float32)
        
        # Percentile normalization for robustness
        p_low, p_high = np.percentile(img, [1, 99])
        img = np.clip(img, p_low, p_high)
        
        # Scale to 0-255
        if p_high > p_low:
            img = (img - p_low) / (p_high - p_low) * 255
        else:
            img = np.zeros_like(img)
        
        return img.astype(np.uint8)
    
    def process_single_hdf5(self, hdf5_path):
        """
        Process one HDF5 file: extract T0 and T15, save as PNG
        Only include if both T0 and T15 exist (16 timepoints minimum)
        
        Returns: True if successful, False otherwise
        """
        # Parse metadata from filename
        metadata_info = self.parse_metadata_from_hdf5(hdf5_path)
        
        if metadata_info is None:
            return False
        
        # Skip if not DMSO
        if metadata_info['name'].upper() != 'DMSO':
            return False
        
        # Skip bad wells
        if metadata_info['well'] in self.bad_wells:
            return False
        
        # Read HDF5 file
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # New format: images with shape (T, C, H, W)
                if 'images' in f:
                    images = f['images'][:]  # (timepoints, channels, height, width)
                    num_timepoints = images.shape[0]
                    
                    # Check if we have at least 16 timepoints (T0-T15, indices 0-15)
                    if num_timepoints < 16:
                        return False
                    
                    # Check if we have 2 channels
                    if images.shape[1] < 2:
                        return False
                    
                    # Extract channels: Ch0=RFP1, Ch1=Halo
                    rfp1_data = images[:, 0, :, :]  # (T, H, W)
                    halo_data = images[:, 1, :, :]  # (T, H, W)
                    
                # Old format fallback: separate ch0, ch1 keys
                elif 'ch0' in f and 'ch1' in f:
                    rfp1_data = f['ch0'][:]  # (T, H, W)
                    halo_data = f['ch1'][:]  # (T, H, W)
                    num_timepoints = rfp1_data.shape[0]
                    
                    if num_timepoints < 16:
                        return False
                else:
                    return False
                
                # Extract T0 (index 0) and T15 (index 15)
                # T0 = first timepoint, T15 = 16th timepoint (0-indexed)
                t0_idx = 0
                t15_idx = 15
                
                rfp1_t0 = self.normalize_image(rfp1_data[t0_idx])
                rfp1_t15 = self.normalize_image(rfp1_data[t15_idx])
                halo_t0 = self.normalize_image(halo_data[t0_idx])
                halo_t15 = self.normalize_image(halo_data[t15_idx])
                
        except Exception as e:
            return False
        
        # Create unique neuron ID from parsed metadata
        neuron_id = f"{metadata_info['well']}_tile{metadata_info['tile']}_n{metadata_info['neuron_id']}"
        
        # Save images as PNG
        rfp1_t0_path = self.output_dir / 'images' / 'RFP1' / f"{neuron_id}_T0.png"
        rfp1_t15_path = self.output_dir / 'images' / 'RFP1' / f"{neuron_id}_T15.png"
        halo_t0_path = self.output_dir / 'images' / 'Halo' / f"{neuron_id}_T0.png"
        halo_t15_path = self.output_dir / 'images' / 'Halo' / f"{neuron_id}_T15.png"
        
        Image.fromarray(rfp1_t0).save(rfp1_t0_path)
        Image.fromarray(rfp1_t15).save(rfp1_t15_path)
        Image.fromarray(halo_t0).save(halo_t0_path)
        Image.fromarray(halo_t15).save(halo_t15_path)
        
        # Create metadata entries
        base_metadata = {
            'neuron_id': neuron_id,
            'condition': metadata_info['condition'],
            'cell_line': metadata_info['condition'],
            'drug': metadata_info['name'],
            'well': metadata_info['well'],
            'tile': metadata_info['tile'],
            'stimulation': metadata_info['stimulation'],
            'original_hdf5': str(hdf5_path),
            'num_timepoints': num_timepoints
        }
        
        # T0 metadata
        t0_metadata = {
            **base_metadata,
            'timepoint': 'T0',
            'timepoint_idx': t0_idx,
            'rfp1_path': str(rfp1_t0_path.relative_to(self.output_dir)),
            'halo_path': str(halo_t0_path.relative_to(self.output_dir))
        }
        
        # T15 metadata (calling it T16 in the label for user clarity)
        t15_metadata = {
            **base_metadata,
            'timepoint': 'T16',  # User calls it T16 (16th timepoint)
            'timepoint_idx': t15_idx,
            'rfp1_path': str(rfp1_t15_path.relative_to(self.output_dir)),
            'halo_path': str(halo_t15_path.relative_to(self.output_dir))
        }
        
        # Add class labels for different classification tasks
        for metadata in [t0_metadata, t15_metadata]:
            # Individual classifiers
            metadata['label_stimulation'] = metadata_info['stimulation']
            metadata['label_cell_line'] = metadata_info['condition']
            metadata['label_timepoint'] = metadata['timepoint']
            
            # Combined classifiers (2-way)
            metadata['label_stim_time'] = f"{metadata_info['stimulation']}_{metadata['timepoint']}"
            metadata['label_cell_time'] = f"{metadata_info['condition']}_{metadata['timepoint']}"
            metadata['label_cell_stim'] = f"{metadata_info['condition']}_{metadata_info['stimulation']}"
            
            # Full 3-way classifier (8 classes total)
            metadata['label_all'] = f"{metadata_info['condition']}_{metadata_info['stimulation']}_{metadata['timepoint']}"
            
            # 6-class: Collapse T0 (pre-stim), separate T16 (post-stim)
            # At T0: cells haven't been stimulated yet, so 0ms and 10000ms should be identical
            # At T16: phenotype has emerged, so keep stimulation separate
            if metadata['timepoint'] == 'T0':
                metadata['label_6class'] = f"{metadata_info['condition']}_T0"
            else:  # T16
                metadata['label_6class'] = f"{metadata_info['condition']}_{metadata_info['stimulation']}_T16"
            
        self.metadata.append(t0_metadata)
        self.metadata.append(t15_metadata)
        
        return True
    
    def build_dataset(self, max_files=None):
        """
        Process all HDF5 files in directory
        
        Args:
            max_files: For testing, limit number of files processed
        """
        # Find all HDF5 files
        hdf5_files = list(self.hdf5_dir.rglob('*.h5')) + list(self.hdf5_dir.rglob('*.hdf5'))
        
        if max_files:
            hdf5_files = hdf5_files[:max_files]
        
        print(f"Found {len(hdf5_files)} HDF5 files")
        print(f"Processing all files (filtering during processing)")
        
        # Process each file
        success_count = 0
        for hdf5_path in tqdm(hdf5_files, desc="Processing HDF5 files"):
            if self.process_single_hdf5(hdf5_path):
                success_count += 1
        
        print(f"\nSuccessfully processed {success_count}/{len(hdf5_files)} files")
        
        # Create metadata dataframe
        if not self.metadata:
            raise ValueError("No valid files were processed! Check that files have >=16 timepoints and are in A-D rows.")
        
        df = pd.DataFrame(self.metadata)
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.csv'
        df.to_csv(metadata_path, index=False)
        print(f"Saved metadata to {metadata_path}")
        
        # Print dataset statistics
        self.print_statistics(df)
        
        return df
    
    def print_statistics(self, df):
        """Print dataset statistics"""
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        
        print(f"\nTotal samples: {len(df)}")
        print(f"Unique neurons: {df['neuron_id'].nunique()}")
        
        print("\n--- Individual Classifications ---")
        print("\nStimulation distribution:")
        print(df.groupby(['label_stimulation', 'timepoint']).size().unstack(fill_value=0))
        
        print("\nCell line distribution:")
        print(df.groupby(['label_cell_line', 'timepoint']).size().unstack(fill_value=0))
        
        print("\nTimepoint distribution:")
        print(df['label_timepoint'].value_counts().sort_index())
        
        print("\n--- Full Classification (8 classes) ---")
        print(df['label_all'].value_counts().sort_index())
        
        print("\n--- 6-Class (Collapse T0 pre-stim) ---")
        print(df['label_6class'].value_counts().sort_index())
        
        print("\n--- Data Quality ---")
        print(f"\nWells represented: {df['well'].nunique()}")
        print("\nSamples per well:")
        print(df.groupby('well')['neuron_id'].nunique().describe())
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare A2screenPlate2 dataset for classification training'
    )
    parser.add_argument(
        '--hdf5_dir',
        type=str,
        required=True,
        help='Directory containing HDF5 files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--bad_wells',
        type=str,
        nargs='*',
        default=[],
        help='List of bad wells to exclude'
    )
    parser.add_argument(
        '--metadata_csv',
        type=str,
        default=None,
        help='Optional CSV with metadata'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("A2 SCREEN PLATE 2 - DATASET PREPARATION")
    print("="*80)
    print(f"\nInput directory: {args.hdf5_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.bad_wells:
        print(f"Excluding bad wells: {len(args.bad_wells)} wells")
    if args.metadata_csv:
        print(f"Using external metadata: {args.metadata_csv}")
    print()
    print("Requirements:")
    print("  - Filename format: A2screenPlate2_{WELL}_{TILE}_{NEURON}.h5")
    print("  - Wells: A1-A24, B1-B24, C1-C24, D1-D24 only")
    print("  - Timepoints: >=16 (T0 to T15)")
    print("  - Channels: 2 (RFP1 and Halo)")
    print("  - Condition: A/B=Control, C/D=APPV717I")
    print("  - Stimulation: Tile 2/3=10000ms, Tile 1/4=0ms")
    print()
    
    # Build dataset
    builder = A2ScreenDatasetBuilder(
        hdf5_dir=args.hdf5_dir,
        output_dir=args.output_dir,
        bad_wells=args.bad_wells,
        metadata_csv=args.metadata_csv
    )
    
    df = builder.build_dataset(max_files=args.max_files)
    
    print("\n✅ Dataset preparation complete!")
    print(f"Images saved to: {args.output_dir}/images/")
    print(f"Metadata saved to: {args.output_dir}/metadata.csv")
    
    # Save a summary JSON for reference
    summary = {
        'total_samples': len(df),
        'unique_neurons': df['neuron_id'].nunique(),
        'cell_lines': df['label_cell_line'].unique().tolist(),
        'stimulations': df['label_stimulation'].unique().tolist(),
        'timepoints': df['label_timepoint'].unique().tolist(),
        'bad_wells_excluded': args.bad_wells,
        'wells_included': df['well'].unique().tolist()
    }
    
    summary_path = Path(args.output_dir) / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
