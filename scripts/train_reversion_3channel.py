#!/usr/bin/env python3
"""
Train Reversion model: T0 vs T15 (same neurons, pre vs post-stimulation)
Uses 3-channel approach: RFP1 + Halo + Halo_masked
NO class weighting - data is pre-balanced
"""
import sys
sys.path.insert(0, 'src')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from train_lightning_threechannel_improved import ThreeChannelClassifier, ThreeChannelDataModule
from pathlib import Path

def train_reversion(metadata_path, root_dir, output_dir, architecture='resnet18', epochs=100):
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / f'Reversion_3channel_{architecture}' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"REVERSION MODEL: T0 vs T15 ({architecture})")
    print("3-channel: RFP1 + Halo + Halo_masked")
    print("Task: Detect pre-stim vs post-stim on SAME neurons")
    print("="*70)
    
    data_module = ThreeChannelDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    # Don't use class weights - data is already balanced
    model = ThreeChannelClassifier(
        num_classes=2,
        class_names=data_module.class_names,
        class_weights=None,  # No weighting!
        learning_rate=1e-4,
        architecture=architecture
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min')
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10
    )
    
    trainer.fit(model, data_module)
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    print(f"\n✅ Overall Test Accuracy: {test_results[0]['test_acc']:.3f}")
    return test_results[0]['test_acc']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--architecture', default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    train_reversion(args.metadata, args.root_dir, args.output_dir, 
                   args.architecture, args.epochs)
