#!/usr/bin/env python3
"""Train models on masked data for comparison"""
import sys
sys.path.insert(0, 'src')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from train_lightning_wells import AggregationClassifier, AggregationDataModuleWells
from train_lightning_dualchannel import DualChannelClassifier, DualChannelDataModule
from train_lightning_dualchannel_masked import DualChannelClassifier as MaskedClassifier
from train_lightning_dualchannel_masked import DualChannelMaskedDataModule
from pathlib import Path

def train_single_channel(metadata_path, root_dir, output_dir, channel, epochs=100):
    """Train single channel model"""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / channel / 'binary' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training: {channel}")
    print(f"{'='*70}")
    
    data_module = AggregationDataModuleWells(
        metadata_path=metadata_path,
        root_dir=root_dir,
        channel=channel,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    model = AggregationClassifier(num_classes=2, learning_rate=1e-4)
    
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
    
    acc = test_results[0]['test_acc']
    print(f"\n✅ {channel} Test Accuracy: {acc:.3f}")
    return acc

def train_dualchannel(metadata_path, root_dir, output_dir, name, masked=False, epochs=100):
    """Train dual-channel model"""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / name / 'binary' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"{'='*70}")
    
    if masked:
        data_module = DualChannelMaskedDataModule(
            metadata_path=metadata_path,
            root_dir=root_dir,
            batch_size=32,
            num_workers=4
        )
        model_class = MaskedClassifier
    else:
        data_module = DualChannelDataModule(
            metadata_path=metadata_path,
            root_dir=root_dir,
            batch_size=32,
            num_workers=4
        )
        model_class = DualChannelClassifier
    
    data_module.setup()
    model = model_class(num_classes=2, learning_rate=1e-4)
    
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
    
    acc = test_results[0]['test_acc']
    print(f"\n✅ {name} Test Accuracy: {acc:.3f}")
    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--mode', required=True, 
                       choices=['single_rfp1', 'single_halo', 'dual', 'dual_masked'])
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    if args.mode == 'single_rfp1':
        train_single_channel(args.metadata, args.root_dir, args.output_dir, 'RFP1', args.epochs)
    elif args.mode == 'single_halo':
        train_single_channel(args.metadata, args.root_dir, args.output_dir, 'Halo', args.epochs)
    elif args.mode == 'dual':
        train_dualchannel(args.metadata, args.root_dir, args.output_dir, 'DualChannel', False, args.epochs)
    elif args.mode == 'dual_masked':
        train_dualchannel(args.metadata, args.root_dir, args.output_dir, 'DualChannel_Masked', True, args.epochs)
