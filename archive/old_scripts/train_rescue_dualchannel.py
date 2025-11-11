#!/usr/bin/env python3
"""Train dual-channel rescue model"""
import sys
sys.path.insert(0, 'src')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from train_lightning_dualchannel import DualChannelClassifier, DualChannelDataModule
from pathlib import Path

def train_dualchannel(metadata_path, root_dir, output_dir, epochs=100):
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / 'DualChannel' / 'binary' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("DUAL-CHANNEL RESCUE MODEL (RFP1 + Halo)")
    print("="*70)
    
    data_module = DualChannelDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    model = DualChannelClassifier(
        num_classes=data_module.num_classes,
        learning_rate=1e-4
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
    
    print(f"\n✅ Test Accuracy: {test_results[0]['test_acc']:.3f}")
    return test_results[0]['test_acc']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    train_dualchannel(args.metadata, args.root_dir, args.output_dir, args.epochs)
