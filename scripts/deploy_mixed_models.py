#!/usr/bin/env python3
"""Deploy models that may have different architectures"""
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

# Import both versions
from train_lightning import AggregationClassifier as OldClassifier
from train_lightning_wells import AggregationClassifier as NewClassifier

class SimpleDataset(Dataset):
    def __init__(self, metadata_df, root_dir, channel):
        self.metadata = metadata_df
        self.root_dir = Path(root_dir)
        self.channel = channel
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = self.root_dir / row[f'{self.channel.lower()}_path']
        img = Image.open(img_path).convert('L')
        return self.transform(img), 0

def deploy_all(checkpoint_path, metadata_path, root_dir, channel, output_file, use_old=False):
    """Deploy - try both model architectures"""
    print(f"Deploying {channel}...")
    
    # Try old architecture first, then new
    if use_old:
        try:
            model = OldClassifier.load_from_checkpoint(checkpoint_path, num_classes=2, class_names=['c0', 'c1'])
            print(f"  Loaded with OLD architecture (backbone)")
        except Exception as e:
            model = NewClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
            print(f"  Loaded with NEW architecture")
    else:
        try:
            model = NewClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
            print(f"  Loaded with NEW architecture")
        except Exception as e:
            model = OldClassifier.load_from_checkpoint(checkpoint_path, num_classes=2, class_names=['c0', 'c1'])
            print(f"  Loaded with OLD architecture (backbone)")
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    metadata = pd.read_csv(metadata_path)
    dataset = SimpleDataset(metadata, root_dir, channel)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=channel):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    probs = np.vstack(all_probs)
    metadata['confidence_class0'] = probs[:, 0]
    metadata['confidence_class1'] = probs[:, 1]
    metadata['predicted_class'] = np.argmax(probs, axis=1)
    
    metadata.to_csv(output_file, index=False)
    print(f"✅ Saved {len(metadata)} samples")
    return metadata

if len(sys.argv) != 8:
    print("Usage: script.py metadata root_dir rev_rfp1 rev_halo resc_rfp1 resc_halo output_dir")
    sys.exit(1)

metadata, root_dir = sys.argv[1:3]
rev_rfp1, rev_halo, resc_rfp1, resc_halo = sys.argv[3:7]
output_dir = Path(sys.argv[7])
output_dir.mkdir(parents=True, exist_ok=True)

# Old models (reversion)
deploy_all(rev_rfp1, metadata, root_dir, 'RFP1', output_dir / 'reversion_rfp1_ALL.csv', use_old=True)
deploy_all(rev_halo, metadata, root_dir, 'Halo', output_dir / 'reversion_halo_ALL.csv', use_old=True)

# New models (rescue)
deploy_all(resc_rfp1, metadata, root_dir, 'RFP1', output_dir / 'rescue_rfp1_ALL.csv', use_old=False)
deploy_all(resc_halo, metadata, root_dir, 'Halo', output_dir / 'rescue_halo_ALL.csv', use_old=False)

print("\n✅ ALL DONE")
