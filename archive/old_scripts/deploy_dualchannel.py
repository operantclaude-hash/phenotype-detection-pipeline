#!/usr/bin/env python3
"""Deploy dual-channel model"""
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
from train_lightning_dualchannel import DualChannelClassifier

class DualChannelDeployDataset(Dataset):
    """Load both RFP1 and Halo for deployment"""
    def __init__(self, metadata_df, root_dir):
        self.metadata = metadata_df
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load both channels
        rfp1_path = self.root_dir / row['rfp1_path']
        halo_path = self.root_dir / row['halo_path']
        
        rfp1 = Image.open(rfp1_path).convert('L')
        halo = Image.open(halo_path).convert('L')
        
        # Convert to tensors
        rfp1_t = transforms.ToTensor()(rfp1)
        halo_t = transforms.ToTensor()(halo)
        
        # Stack as 2-channel
        img = torch.cat([rfp1_t, halo_t], dim=0)
        img = transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])(img)
        
        return img, 0

def deploy_dualchannel(checkpoint_path, metadata_path, root_dir, output_file):
    """Deploy dual-channel model on ALL data"""
    print(f"\n{'='*70}")
    print(f"Deploying Dual-Channel Model")
    print(f"{'='*70}")
    
    # Load model
    model = DualChannelClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load data
    metadata = pd.read_csv(metadata_path)
    dataset = DualChannelDeployDataset(metadata, root_dir)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    
    # Run inference
    all_probs = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Scoring"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # Save results
    probs = np.vstack(all_probs)
    metadata['confidence_class0'] = probs[:, 0]
    metadata['confidence_class1'] = probs[:, 1]
    metadata['predicted_class'] = np.argmax(probs, axis=1)
    
    metadata.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved {len(metadata)} samples to {output_file}")
    print(f"{'='*70}")
    return metadata

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Dual-channel checkpoint')
    parser.add_argument('--metadata', required=True, help='Metadata CSV')
    parser.add_argument('--root_dir', required=True, help='Root directory')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    deploy_dualchannel(args.checkpoint, args.metadata, args.root_dir, args.output)
