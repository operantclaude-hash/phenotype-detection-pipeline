#!/usr/bin/env python3
"""Test if dual-channel model uses both channels by ablating one"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from pathlib import Path
from train_lightning_dualchannel_masked import DualChannelClassifier, DualChannelMaskedDataModule

def test_ablation(checkpoint_path, metadata_path, root_dir):
    """Test with: both channels, RFP1 only, Halo only"""
    
    # Load model and data
    model = DualChannelClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    data_module = DualChannelMaskedDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    test_loader = data_module.test_dataloader()
    
    # Test 1: Both channels (normal)
    correct_both = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct_both += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc_both = correct_both / total
    print(f"Both channels: {acc_both:.3f}")
    
    # Test 2: RFP1 only (zero out Halo = channel 1)
    correct_rfp1 = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            images[:, 1, :, :] = 0  # Zero out Halo channel
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct_rfp1 += (preds == labels).sum().item()
    
    acc_rfp1 = correct_rfp1 / total
    print(f"RFP1 only (Halo zeroed): {acc_rfp1:.3f}")
    
    # Test 3: Halo only (zero out RFP1 = channel 0)
    correct_halo = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            images[:, 0, :, :] = 0  # Zero out RFP1 channel
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct_halo += (preds == labels).sum().item()
    
    acc_halo = correct_halo / total
    print(f"Halo only (RFP1 zeroed): {acc_halo:.3f}")
    
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    if acc_rfp1 > acc_both * 0.95 or acc_halo > acc_both * 0.95:
        print("⚠️  Model is mostly using ONE channel!")
        if acc_rfp1 > acc_halo:
            print("   Dominated by RFP1")
        else:
            print("   Dominated by Halo")
    else:
        print("✅ Model is using BOTH channels")
    print(f"{'='*60}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    
    args = parser.parse_args()
    test_ablation(args.checkpoint, args.metadata, args.root_dir)
