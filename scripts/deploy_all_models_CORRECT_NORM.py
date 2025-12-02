#!/usr/bin/env python3
"""
Unified Deployment Script - CORRECT NORMALIZATION
Deploys all models with EXACT SAME normalization:
1. RESCUE (3-channel): APP vs WT at single timepoints
2. APP Treatment (6-channel): APP_T0 vs APP_T15
3. WT Treatment (6-channel): WT_T0 vs WT_T15
4. Binary 6-channel: General temporal classification

ALL models use identical 3-stage normalization:
  1. Percentile clipping [p1, p99]
  2. Normalize to [0, 1]
  3. ImageNet normalization
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import argparse
import json
import warnings
warnings.filterwarnings('ignore')


class ThreeChannelDataset(Dataset):
    """
    3-channel dataset for RESCUE models with CORRECT normalization
    Uses EXACT SAME normalization as 6-channel models
    """
    def __init__(self, metadata_df, percentiles_path='A2screen_combined/data/intensity_percentiles.json', root_dir='A2screen_combined/data'):
        self.metadata = metadata_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)

        # Load intensity percentiles
        with open(percentiles_path) as f:
            self.percentiles = json.load(f)

        # ImageNet normalization (SAME AS 6-CHANNEL)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.resize = transforms.Resize((224, 224))

    def load_and_normalize_channel(self, path, channel_name):
        """
        CORRECT normalization - SAME AS TRAINING:
        1. Load 16-bit image
        2. Clip to percentile bounds [p1, p99]
        3. Normalize to [0, 1]: (img - p1) / (p99 - p1)
        4. Return as tensor
        """
        # Load 16-bit image as numpy array
        img = np.array(Image.open(path), dtype=np.float32)

        # Get percentile bounds for this channel
        # Map halo_masked to halo percentiles
        perc_key = 'halo' if 'halo_masked' in channel_name or channel_name == 'halo_masked' else channel_name
        p1 = self.percentiles[perc_key]['p1']
        p99 = self.percentiles[perc_key]['p99']

        # Step 1: Clip to percentile bounds
        img_clipped = np.clip(img, p1, p99)

        # Step 2: Normalize to [0, 1]
        if p99 > p1:
            img_normalized = (img_clipped - p1) / (p99 - p1)
        else:
            img_normalized = img_clipped

        # Convert to PIL for transforms, then to tensor
        img_pil = Image.fromarray((img_normalized * 255).astype(np.uint8))
        img_pil = self.resize(img_pil)

        # Convert to tensor (now in [0, 1])
        img_tensor = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).unsqueeze(0)

        return img_tensor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load 3 channels with CORRECT normalization
        rfp1_path = self.root_dir / row['rfp1_path']
        halo_path = self.root_dir / row['halo_path']
        halo_masked_path = self.root_dir / row['halo_masked_path']

        rfp1 = self.load_and_normalize_channel(rfp1_path, 'rfp1')
        halo = self.load_and_normalize_channel(halo_path, 'halo')
        halo_masked = self.load_and_normalize_channel(halo_masked_path, 'halo_masked')

        # Concatenate to 3-channel image
        img = torch.cat([rfp1, halo, halo_masked], dim=0)

        # Apply ImageNet normalization (SAME AS 6-CHANNEL)
        img = self.normalize(img)

        return img, idx


class SixChannelDataset(Dataset):
    """
    6-channel dataset for treatment models with CORRECT normalization
    Uses EXACT SAME normalization as 3-channel models
    """
    def __init__(self, metadata_df, percentiles_path='A2screen_combined/data/intensity_percentiles.json', root_dir='A2screen_combined/data'):
        self.metadata = metadata_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)

        # Load intensity percentiles
        with open(percentiles_path) as f:
            self.percentiles = json.load(f)

        # ImageNet normalization for 6 channels (2 timepoints × 3 channels)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406] * 2,
            std=[0.229, 0.224, 0.225] * 2
        )

        self.resize = transforms.Resize((224, 224))

    def load_and_normalize_channel(self, path, channel_name):
        """
        CORRECT normalization - SAME AS TRAINING:
        1. Load 16-bit image
        2. Clip to percentile bounds [p1, p99]
        3. Normalize to [0, 1]: (img - p1) / (p99 - p1)
        4. Return as tensor
        """
        # Load 16-bit image as numpy array
        img = np.array(Image.open(path), dtype=np.float32)

        # Get percentile bounds for this channel
        # Map halo_masked to halo percentiles
        perc_key = 'halo' if 'halo_masked' in channel_name or channel_name == 'halo_masked' else channel_name
        p1 = self.percentiles[perc_key]['p1']
        p99 = self.percentiles[perc_key]['p99']

        # Step 1: Clip to percentile bounds
        img_clipped = np.clip(img, p1, p99)

        # Step 2: Normalize to [0, 1]
        if p99 > p1:
            img_normalized = (img_clipped - p1) / (p99 - p1)
        else:
            img_normalized = img_clipped

        # Convert to PIL for transforms, then to tensor
        img_pil = Image.fromarray((img_normalized * 255).astype(np.uint8))
        img_pil = self.resize(img_pil)

        # Convert to tensor (now in [0, 1])
        img_tensor = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).unsqueeze(0)

        return img_tensor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load T0 channels
        rfp1_t0_path = self.root_dir / row['rfp1_t0_path']
        halo_t0_path = self.root_dir / row['halo_t0_path']
        halo_masked_t0_path = self.root_dir / row['halo_masked_t0_path']

        # Load T15 channels
        rfp1_t15_path = self.root_dir / row['rfp1_t15_path']
        halo_t15_path = self.root_dir / row['halo_t15_path']
        halo_masked_t15_path = self.root_dir / row['halo_masked_t15_path']

        # Load and normalize all 6 channels with CORRECT normalization
        rfp1_t0 = self.load_and_normalize_channel(rfp1_t0_path, 'rfp1')
        halo_t0 = self.load_and_normalize_channel(halo_t0_path, 'halo')
        halo_masked_t0 = self.load_and_normalize_channel(halo_masked_t0_path, 'halo_masked')

        rfp1_t15 = self.load_and_normalize_channel(rfp1_t15_path, 'rfp1')
        halo_t15 = self.load_and_normalize_channel(halo_t15_path, 'halo')
        halo_masked_t15 = self.load_and_normalize_channel(halo_masked_t15_path, 'halo_masked')

        # Concatenate to 6-channel image (T0 + T15)
        img = torch.cat([rfp1_t0, halo_t0, halo_masked_t0,
                        rfp1_t15, halo_t15, halo_masked_t15], dim=0)

        # Apply ImageNet normalization (SAME AS 3-CHANNEL)
        img = self.normalize(img)

        return img, idx


class SimpleClassifier(nn.Module):
    """Simple wrapper for loaded models"""
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes, in_chans=in_channels)

    def forward(self, x):
        return self.model(x)


def load_model(checkpoint_path, num_classes=2, in_channels=3):
    """Load a trained model from checkpoint"""
    model = SimpleClassifier(num_classes=num_classes, in_channels=in_channels)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict (handle different checkpoint formats)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()
                     if not k.startswith('train_acc') and not k.startswith('val_acc') and not k.startswith('criterion')}
    else:
        state_dict = ckpt

    model.model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def deploy_model(model, dataset, device='cuda', batch_size=64):
    """Deploy model and get predictions"""
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_probs = []
    all_preds = []
    all_indices = []

    with torch.no_grad():
        for imgs, indices in tqdm(loader, desc="Predicting"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_indices.append(indices.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_indices = np.concatenate(all_indices)

    return all_preds, all_probs, all_indices


def main():
    parser = argparse.ArgumentParser(description='Deploy models with CORRECT normalization')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rescue_0ms', 'rescue_10000ms', 'app_treatment', 'wt_treatment', 'binary'],
                       help='Which model to deploy')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata CSV')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--percentiles', type=str, default='A2screen_combined/data/intensity_percentiles.json',
                       help='Path to intensity percentiles JSON')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"DEPLOYING MODEL: {args.model_type}")
    print(f"NORMALIZATION: Percentile clipping → [0,1] → ImageNet")
    print(f"{'='*80}\n")

    # Load metadata
    print(f"Loading metadata from: {args.metadata}")
    metadata = pd.read_csv(args.metadata)
    print(f"  Total samples: {len(metadata)}")

    # Create dataset with CORRECT normalization
    if args.model_type in ['rescue_0ms', 'rescue_10000ms']:
        dataset = ThreeChannelDataset(metadata, percentiles_path=args.percentiles)
        in_channels = 3
    else:
        dataset = SixChannelDataset(metadata, percentiles_path=args.percentiles)
        in_channels = 6

    print(f"  Channels: {in_channels}")
    print(f"  Using percentiles from: {args.percentiles}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    model = load_model(args.checkpoint, num_classes=2, in_channels=in_channels)

    # Deploy
    print(f"\nDeploying model...")
    preds, probs, indices = deploy_model(model, dataset, device=device, batch_size=args.batch_size)

    # Add predictions to metadata
    results = metadata.copy()
    results['predicted_class'] = preds
    results['prob_class_0'] = probs[:, 0]
    results['prob_class_1'] = probs[:, 1]
    results['model_type'] = args.model_type
    results['checkpoint'] = args.checkpoint

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\n✓ Predictions saved to: {args.output}")
    print(f"  Total predictions: {len(results)}")
    print(f"  Class 0: {(preds == 0).sum()} ({(preds == 0).sum()/len(preds)*100:.1f}%)")
    print(f"  Class 1: {(preds == 1).sum()} ({(preds == 1).sum()/len(preds)*100:.1f}%)")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
