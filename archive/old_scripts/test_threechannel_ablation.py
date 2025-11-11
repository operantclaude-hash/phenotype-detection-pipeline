#!/usr/bin/env python3
"""Test if 3-channel model uses all channels"""
import sys
sys.path.insert(0, 'src')

import torch
from train_lightning_threechannel_improved import ThreeChannelClassifier, ThreeChannelDataModule

def test_ablation(checkpoint_path, metadata_path, root_dir):
    """Test with all channels vs removing each one"""
    
    model = ThreeChannelClassifier.load_from_checkpoint(
        checkpoint_path, 
        num_classes=2, 
        class_names=['APPV717I', 'Control']
    )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    data_module = ThreeChannelDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    def compute_accuracy(zero_channel=None):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                if zero_channel is not None:
                    images[:, zero_channel, :, :] = 0  # Zero out channel
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total
    
    # Test all combinations
    acc_all = compute_accuracy(zero_channel=None)
    acc_no_rfp1 = compute_accuracy(zero_channel=0)
    acc_no_halo = compute_accuracy(zero_channel=1)
    acc_no_halo_masked = compute_accuracy(zero_channel=2)
    
    print("\n" + "="*70)
    print("CHANNEL ABLATION TEST")
    print("="*70)
    print(f"All channels (RFP1 + Halo + Halo_masked):  {acc_all:.3f}")
    print(f"No RFP1 (Halo + Halo_masked):              {acc_no_rfp1:.3f}")
    print(f"No Halo (RFP1 + Halo_masked):              {acc_no_halo:.3f}")
    print(f"No Halo_masked (RFP1 + Halo):              {acc_no_halo_masked:.3f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    drops = {
        'RFP1': acc_all - acc_no_rfp1,
        'Halo': acc_all - acc_no_halo,
        'Halo_masked': acc_all - acc_no_halo_masked
    }
    
    for channel, drop in drops.items():
        if abs(drop) < 0.02:
            print(f"⚠️  {channel}: Minimal impact ({drop:+.3f}) - may not be used")
        else:
            print(f"✅ {channel}: Important ({drop:+.3f} accuracy drop when removed)")
    print("="*70)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    
    args = parser.parse_args()
    test_ablation(args.checkpoint, args.metadata, args.root_dir)
