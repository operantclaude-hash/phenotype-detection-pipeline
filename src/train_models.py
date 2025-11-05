#!/usr/bin/env python3
"""
Train classification models for A2screenPlate2 dataset
Supports multiple classification tasks including 6-class that collapses pre-stim T0
"""

import argparse
import sys
from pathlib import Path

# Import the original training module
sys.path.insert(0, 'src')
from train_lightning import AggregationClassifier, AggregationDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


# Configuration presets for different classification tasks
CLASSIFICATION_CONFIGS = {
    'stimulation': {
        'label_column': 'label_stimulation',
        'num_classes': 2,
        'class_names': ['0ms', '10000ms'],
        'description': 'Binary: Stimulation (0ms vs 10000ms)'
    },
    'cell_line': {
        'label_column': 'label_cell_line',
        'num_classes': 2,
        'class_names': ['Control', 'APPV717I'],
        'description': 'Binary: Cell line (Control vs APPV717I)'
    },
    'timepoint': {
        'label_column': 'label_timepoint',
        'num_classes': 2,
        'class_names': ['T0', 'T16'],
        'description': 'Binary: Timepoint (T0 vs T16)'
    },
    'stim_time': {
        'label_column': 'label_stim_time',
        'num_classes': 4,
        'class_names': ['0ms_T0', '0ms_T16', '10000ms_T0', '10000ms_T16'],
        'description': '4-class: Stimulation × Timepoint'
    },
    'cell_time': {
        'label_column': 'label_cell_time',
        'num_classes': 4,
        'class_names': ['Control_T0', 'Control_T16', 'APPV717I_T0', 'APPV717I_T16'],
        'description': '4-class: Cell line × Timepoint'
    },
    'cell_stim': {
        'label_column': 'label_cell_stim',
        'num_classes': 4,
        'class_names': ['Control_0ms', 'Control_10000ms', 'APPV717I_0ms', 'APPV717I_10000ms'],
        'description': '4-class: Cell line × Stimulation'
    },
    'full': {
        'label_column': 'label_all',
        'num_classes': 8,
        'class_names': [
            'Control_0ms_T0', 'Control_0ms_T16',
            'Control_10000ms_T0', 'Control_10000ms_T16',
            'APPV717I_0ms_T0', 'APPV717I_0ms_T16',
            'APPV717I_10000ms_T0', 'APPV717I_10000ms_T16'
        ],
        'description': '8-class: Cell line × Stimulation × Timepoint (full model)'
    },
    'binary': {
        'label_column': 'label_binary',
        'num_classes': 2,
        'class_names': ['Control', 'APPV717I'],
        'description': 'Binary: Control vs APPV717I'
    },
        'binary': {
        'label_column': 'label_binary',
        'num_classes': 2,
        'class_names': ['Control', 'APPV717I'],
        'description': 'Binary: Control vs APPV717I'
    },
        '6class': {
        'label_column': 'label_6class',
        'num_classes': 6,
        'class_names': [
            'Control_T0',               # Collapse 0ms and 10000ms at T0
            'APPV717I_T0',              # Collapse 0ms and 10000ms at T0
            'Control_0ms_T16',          # Separate at T16 (phenotype emerged)
            'Control_10000ms_T16',
            'APPV717I_0ms_T16',
            'APPV717I_10000ms_T16'
        ],
        'description': '6-class: Collapse T0 pre-stim (avoids technical artifacts), separate T16 post-stim (real phenotype)'
    }
}


def train_single_model(
    metadata_path,
    root_dir,
    output_dir,
    channel,
    task,
    batch_size=32,
    epochs=100,
    lr=0.001,
    architecture='resnet18',
    num_workers=4,
    accelerator='auto',
    devices='auto'
):
    """
    Train a single classification model
    
    Args:
        metadata_path: Path to metadata CSV
        root_dir: Root directory containing images/
        output_dir: Output directory for this model
        channel: 'RFP1' or 'Halo'
        task: Classification task (key from CLASSIFICATION_CONFIGS)
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        lr: Learning rate
        architecture: Model architecture (resnet18, resnet50, etc.)
        num_workers: Number of data loading workers
        accelerator: 'auto', 'gpu', 'cpu'
        devices: Number of devices to use
    
    Returns:
        dict with test results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get task configuration
    task_config = CLASSIFICATION_CONFIGS[task]
    
    print("="*80)
    print(f"TRAINING: {task_config['description']}")
    print(f"Channel: {channel}")
    print(f"Classes: {task_config['num_classes']}")
    print("="*80)
    
    # Create data module
    data_module = AggregationDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        channel=channel,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=224,
        stratify_by=task_config['label_column']
    )
    data_module.setup()
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(data_module.train_dataset)} samples")
    print(f"  Val:   {len(data_module.val_dataset)} samples")
    print(f"  Test:  {len(data_module.test_dataset)} samples")
    print("Class distribution:")
    # Note: can't easily print class distribution without accessing internal dataframes
    
    # Create model
    model = AggregationClassifier(
        num_classes=task_config['num_classes'],
        learning_rate=lr,
        architecture=architecture,
        class_names=task_config['class_names']
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=15,
        mode='max',
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / 'logs',
        name=f'{channel}_{task}'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.fit(model, data_module)
    
    # Test on best model
    print("\n📊 Evaluating on test set...")
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    print(f"\n✅ Training complete!")
    print(f"Best model saved to: {output_dir / 'checkpoints'}")
    print(f"Test accuracy: {test_results[0]['test_acc']:.3f}")
    
    return test_results[0]


def main():
    parser = argparse.ArgumentParser(
        description='Train A2screenPlate2 classification models'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata CSV'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        required=True,
        help='Root directory containing images/'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for models'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        choices=list(CLASSIFICATION_CONFIGS.keys()),
        default=['6class'],
        help='Classification tasks to train (default: 6class)'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        choices=['RFP1', 'Halo'],
        default=['RFP1', 'Halo'],
        help='Channels to train (default: both)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum epochs (default: 100)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='resnet18',
        help='Model architecture (default: resnet18)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=['auto', 'gpu', 'cpu'],
        help='Accelerator type (default: auto)'
    )
    parser.add_argument(
        '--devices',
        default='auto',
        help='Number of devices (default: auto)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("A2 SCREEN PLATE 2 - MODEL TRAINING")
    print("="*80)
    print(f"Tasks to train: {args.tasks}")
    print(f"Channels: {args.channels}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Train all combinations
    results = {}
    for channel in args.channels:
        results[channel] = {}
        for task in args.tasks:
            print(f"\n{'='*80}")
            print(f"Training: {channel} - {task}")
            print("="*80)
            
            output_dir = Path(args.output_dir) / channel / task
            
            test_results = train_single_model(
                metadata_path=args.metadata,
                root_dir=args.root_dir,
                output_dir=output_dir,
                channel=channel,
                task=task,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                architecture=args.architecture,
                num_workers=args.num_workers,
                accelerator=args.accelerator,
                devices=args.devices
            )
            
            results[channel][task] = test_results['test_acc']
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for channel, task_results in results.items():
        print(f"\n{channel} Channel:")
        for task, acc in task_results.items():
            print(f"  {task:15s}: {acc:.3f}")
    
    print("\n✅ All training complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
