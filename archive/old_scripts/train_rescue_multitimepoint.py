#!/usr/bin/env python3
"""Train rescue model with well-based splits"""
import sys
sys.path.insert(0, 'src')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from train_lightning_wells import AggregationClassifier, AggregationDataModuleWells
from pathlib import Path

def train_rescue_model(metadata_path, root_dir, output_dir, channel, epochs=100):
    """Train one rescue model with well-based splits"""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / channel / 'binary' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training Rescue Model: {channel}")
    print(f"{'='*70}")
    
    # Data module with well splits
    data_module = AggregationDataModuleWells(
        metadata_path=metadata_path,
        root_dir=root_dir,
        channel=channel,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    # Model
    model = AggregationClassifier(
        num_classes=data_module.num_classes,
        learning_rate=1e-4
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    print(f"\n✅ {channel} Test Accuracy: {test_results[0]['test_acc']:.3f}")
    
    return test_results[0]['test_acc']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--channels', nargs='+', default=['RFP1', 'Halo'])
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    results = {}
    for channel in args.channels:
        acc = train_rescue_model(
            args.metadata, args.root_dir, args.output_dir, channel, args.epochs
        )
        results[channel] = acc
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    for channel, acc in results.items():
        print(f"{channel}: {acc:.3f}")
