#!/usr/bin/env python3
"""Deploy Model 1 and Model 2 reversion models on full dataset"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_lightning_threechannel_improved import ThreeChannelClassifier

# Copy ThreeChannelDeployDataset from earlier
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class ThreeChannelDeployDataset(Dataset):
    def __init__(self, metadata_df, root_dir):
        self.metadata = metadata_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
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

        return img, idx  # Return index for explicit tracking

def deploy_reversion(checkpoint_path, metadata_path, root_dir, output_file, class_names, prefix):
    print(f"\n{'='*70}")
    print(f"Deploying: {prefix}")
    print(f"{'='*70}")
    
    model = ThreeChannelClassifier.load_from_checkpoint(
        checkpoint_path, num_classes=2, class_names=class_names
    )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    metadata = pd.read_csv(metadata_path)
    print(f"Total samples: {len(metadata)}")
    
    dataset = ThreeChannelDeployDataset(metadata, root_dir)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)

    all_probs = []
    all_indices = []
    with torch.no_grad():
        for images, indices in tqdm(dataloader, desc=f"Scoring {prefix}"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_indices.append(indices.cpu().numpy())

    probs = np.vstack(all_probs)
    indices = np.concatenate(all_indices)

    # Verify index alignment (critical safety check)
    assert len(indices) == len(metadata), \
        f"Sample count mismatch! Expected {len(metadata)}, got {len(indices)}"
    assert np.array_equal(indices, np.arange(len(metadata))), \
        "Index order changed! Predictions would be misaligned with neurons."

    print(f"✓ Index alignment verified: {len(indices)} samples processed in correct order")

    # Add predictions with prefix (using verified indices)
    metadata[f'{prefix}_conf_class0'] = probs[:, 0]
    metadata[f'{prefix}_conf_class1'] = probs[:, 1]
    metadata[f'{prefix}_predicted'] = np.where(probs[:, 0] > probs[:, 1], class_names[0], class_names[1])
    metadata[f'{prefix}_confidence'] = np.max(probs, axis=1)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_file, index=False)
    
    print(f"✅ Saved to {output_file}")
    print(f"Predictions: {metadata[f'{prefix}_predicted'].value_counts().to_dict()}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--model1_ckpt', required=True)
    parser.add_argument('--model2_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Deploy Model 1
    deploy_reversion(
        checkpoint_path=args.model1_ckpt,
        metadata_path=args.metadata,
        root_dir=args.root_dir,
        output_file=output_dir / 'reversion_model1_predictions.csv',
        class_names=['Non-aggregated', 'Aggregated'],  # Fixed: match training order
        prefix='rev_m1'
    )
    
    # Deploy Model 2
    deploy_reversion(
        checkpoint_path=args.model2_ckpt,
        metadata_path=args.metadata,
        root_dir=args.root_dir,
        output_file=output_dir / 'reversion_model2_predictions.csv',
        class_names=['APP_Non-aggregated', 'APP_Aggregated'],  # Fixed: match training order
        prefix='rev_m2'
    )
    
    print("\n" + "="*70)
    print("REVERSION DEPLOYMENT COMPLETE")
    print("="*70)
    print(f"Model 1 (84.8%): Aggregation detector (any genotype)")
    print(f"Model 2 (88.2%): APP-specific aggregation detector")
    print("="*70)
