#!/usr/bin/env python3
"""
Deploy both rescue models on ALL data (no splits)
- Model 1: T15 10000ms (72.5% accuracy)
- Model 2: T0-T15 balanced (68% accuracy)
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from train_lightning_threechannel_improved import ThreeChannelClassifier

class ThreeChannelDeployDataset(Dataset):
    """Load all 3 channels for deployment"""
    def __init__(self, metadata_df, root_dir):
        self.metadata = metadata_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        rfp1_path = self.root_dir / row['rfp1_path']
        halo_path = self.root_dir / row['halo_path']
        halo_masked_path = self.root_dir / row['halo_masked_path']
        
        rfp1 = Image.open(rfp1_path).convert('L')
        halo = Image.open(halo_path).convert('L')
        halo_masked = Image.open(halo_masked_path).convert('L')
        
        rfp1_t = transforms.ToTensor()(rfp1)
        halo_t = transforms.ToTensor()(halo)
        halo_masked_t = transforms.ToTensor()(halo_masked)
        
        img = torch.cat([rfp1_t, halo_t, halo_masked_t], dim=0)
        img = self.normalize(img)
        
        return img, 0  # dummy label

def deploy_model(checkpoint_path, metadata_path, root_dir, output_file, model_name):
    """Deploy single model on all data"""
    print(f"\n{'='*70}")
    print(f"Deploying: {model_name}")
    print(f"{'='*70}")
    
    # Load model
    model = ThreeChannelClassifier.load_from_checkpoint(
        checkpoint_path,
        num_classes=2,
        class_names=['APPV717I', 'Control']
    )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Load ALL data
    metadata = pd.read_csv(metadata_path)
    print(f"Total samples: {len(metadata)}")
    print(f"Wells: {len(metadata['well'].unique())}")
    print(f"Timepoints: {sorted(metadata['timepoint'].unique(), key=lambda x: int(x[1:]))}")
    
    # Create dataset
    dataset = ThreeChannelDeployDataset(metadata, root_dir)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    
    # Run inference
    all_probs = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f"Scoring {model_name}"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # Combine results
    probs = np.vstack(all_probs)
    
    # Add predictions to metadata
    metadata['confidence_APPV717I'] = probs[:, 0]
    metadata['confidence_Control'] = probs[:, 1]
    metadata['predicted_class'] = np.where(probs[:, 0] > probs[:, 1], 'APPV717I', 'Control')
    metadata['prediction_confidence'] = np.max(probs, axis=1)
    
    # Save
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved {len(metadata)} predictions to {output_file}")
    print(f"Predicted classes:")
    print(metadata['predicted_class'].value_counts())
    print(f"Mean confidence: {metadata['prediction_confidence'].mean():.3f}")
    print(f"{'='*70}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Deploy both rescue models on full dataset')
    parser.add_argument('--data_dir', required=True, help='Root directory with images')
    parser.add_argument('--metadata', required=True, help='Metadata CSV with ALL samples')
    parser.add_argument('--t15_checkpoint', required=True, help='T15 10000ms model checkpoint')
    parser.add_argument('--t0_t15_checkpoint', required=True, help='T0-T15 balanced model checkpoint')
    parser.add_argument('--output_dir', required=True, help='Output directory for predictions')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Deploy Model 1: T15 10000ms (highest accuracy)
    deploy_model(
        checkpoint_path=args.t15_checkpoint,
        metadata_path=args.metadata,
        root_dir=args.data_dir,
        output_file=output_dir / 'rescue_T15_10000ms_predictions.csv',
        model_name='T15 10000ms Model (72.5% accuracy)'
    )
    
    # Deploy Model 2: T0-T15 balanced (temporal evolution)
    deploy_model(
        checkpoint_path=args.t0_t15_checkpoint,
        metadata_path=args.metadata,
        root_dir=args.data_dir,
        output_file=output_dir / 'rescue_T0-T15_balanced_predictions.csv',
        model_name='T0-T15 Balanced Model (68% accuracy)'
    )
    
    print("\n" + "="*70)
    print("DEPLOYMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("\nOutput files:")
    print(f"  - rescue_T15_10000ms_predictions.csv")
    print(f"  - rescue_T0-T15_balanced_predictions.csv")
    print("\nColumns in output:")
    print("  - All original metadata columns")
    print("  - confidence_APPV717I: Probability of APP class")
    print("  - confidence_Control: Probability of Control class")
    print("  - predicted_class: APPV717I or Control")
    print("  - prediction_confidence: Max probability (0-1)")
    print("="*70)

if __name__ == '__main__':
    main()
