#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from train_lightning_threechannel_improved import ThreeChannelClassifier, ThreeChannelDataModule
from pathlib import Path

def train(metadata_path, root_dir, output_dir, architecture='resnet18', epochs=100):
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / f'ThreeChannel_{architecture}' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"THREE-CHANNEL MODEL: {architecture}")
    print("Features: Class weighting + Aggressive augmentation + Per-class metrics")
    print("="*70)
    
    data_module = ThreeChannelDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    model = ThreeChannelClassifier(
        num_classes=data_module.num_classes,
        class_names=data_module.class_names,
        class_weights=data_module.class_weights,  # Use computed weights
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
    train(args.metadata, args.root_dir, args.output_dir, args.architecture, args.epochs)
