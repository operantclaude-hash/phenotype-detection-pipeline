#!/usr/bin/env python3
"""
PyTorch Lightning training module for optogenetic aggregation classification

Modern, clean implementation with automatic GPU handling, logging, and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import json


class NeuronDataset(Dataset):
    """
    Dataset for single-channel neuron images
    """
    def __init__(
        self, 
        metadata_df: pd.DataFrame,
        channel: str,  # 'RFP1' or 'Halo'
        root_dir: Path,
        transform=None,
        target_transform=None
    ):
        """
        Args:
            metadata_df: DataFrame with image paths and labels
            channel: Which channel to load ('RFP1' or 'Halo')
            root_dir: Root directory containing the data
            transform: Image transformations
            target_transform: Label transformations
        """
        self.metadata = metadata_df.copy()
        self.channel = channel
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Create label encoding
        self.class_names = sorted(self.metadata['class_label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Get image paths
        if channel == 'RFP1':
            self.image_paths = self.metadata['rfp1_path'].values
        elif channel == 'Halo':
            self.image_paths = self.metadata['halo_path'].values
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        self.labels = self.metadata['class_label'].map(self.class_to_idx).values
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.root_dir / self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def get_transforms(image_size=224, augment=True):
    """
    Get image transforms for training and validation
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation
    
    Returns:
        train_transform, val_transform
    """
    # Normalization based on ImageNet (if using pretrained) or compute from your data
    # For now, using standard normalization for grayscale
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


class ResNetClassifier(nn.Module):
    """
    ResNet18 backbone adapted for single-channel input and custom number of classes
    """
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for single channel (grayscale)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Initialize new conv1 weights
        if pretrained:
            # Average the RGB weights to initialize grayscale conv
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class CustomCNN(nn.Module):
    """
    Custom CNN architecture (lighter than ResNet for smaller datasets/faster training)
    """
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AggregationClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for optogenetic aggregation classification
    
    Handles training, validation, testing, and logging automatically
    """
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        architecture: str = 'resnet18',  # 'resnet18' or 'custom_cnn'
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.class_names = class_names
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize model
        if architecture == 'resnet18':
            self.model = ResNetClassifier(num_classes, pretrained, dropout)
        elif architecture == 'custom_cnn':
            self.model = CustomCNN(num_classes, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # For tracking predictions
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions for confusion matrix
        self.validation_step_outputs.append({
            'preds': preds,
            'labels': labels
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        # Compute confusion matrix at end of validation epoch
        if len(self.validation_step_outputs) > 0:
            all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
            
            # Clear stored outputs
            self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        # Store predictions
        self.test_step_outputs.append({
            'preds': preds.cpu(),
            'labels': labels.cpu(),
            'probs': F.softmax(outputs, dim=1).cpu()
        })
        
        return loss
    
    def on_test_epoch_end(self):
        # Compute final test metrics and confusion matrix
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
        
        # Save predictions
        results = {
            'predictions': all_preds.numpy(),
            'labels': all_labels.numpy(),
            'probabilities': all_probs.numpy(),
            'class_names': self.class_names
        }
        
        # Clear stored outputs
        self.test_step_outputs.clear()
        
        return results
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class AggregationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for organizing data loading
    
    Handles train/val/test splits and DataLoader creation
    """
    def __init__(
        self,
        metadata_path: str,
        root_dir: str,
        channel: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.15,
        image_size: int = 224,
        augment: bool = True,
        random_state: int = 42,
        stratify_by: str = 'class_label',  # or 'well' to avoid batch effects
        balance_classes: bool = True
    ):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.root_dir = Path(root_dir)
        self.channel = channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.image_size = image_size
        self.augment = augment
        self.random_state = random_state
        self.stratify_by = stratify_by
        self.balance_classes = balance_classes
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
    
    def prepare_data(self):
        """
        Download or prepare data (called on single GPU/process)
        Not needed here as data is already prepared
        """
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing
        
        Called on every GPU in distributed training
        """
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)
        
        print(f"Loaded metadata: {len(metadata)} samples")
        print(f"Channel: {self.channel}")
        
        # Split data at neuron level (keep T0 and Tn together)
        unique_neurons = metadata['neuron_id'].unique()
        
        # Check if we have enough samples for stratified splitting
        # Need at least 2 samples per class for stratification to work
        if self.stratify_by:
            class_counts = metadata.groupby('neuron_id')[self.stratify_by].first().value_counts()
            min_class_count = class_counts.min()
            samples_needed_per_split = int(np.ceil(len(unique_neurons) * self.test_split))
            
            if min_class_count < 2 or samples_needed_per_split < class_counts.nunique():
                print(f"⚠ Warning: Dataset too small for stratified splitting ({min_class_count} min samples/class)")
                print(f"   Using random split instead")
                stratify_test = None
                stratify_val = None
            else:
                stratify_test = metadata.groupby('neuron_id')[self.stratify_by].first().values
                stratify_val = stratify_test
        else:
            stratify_test = None
            stratify_val = None
        
        # First split: separate test set
        neurons_train_val, neurons_test = train_test_split(
            unique_neurons,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=stratify_test
        )
        
        # Second split: separate validation from training
        val_size_adjusted = self.val_split / (1 - self.test_split)
        
        # Check stratification for val split
        if stratify_val is not None:
            stratify_val_subset = metadata[metadata['neuron_id'].isin(neurons_train_val)].groupby('neuron_id')[self.stratify_by].first().values
        else:
            stratify_val_subset = None
        
        neurons_train, neurons_val = train_test_split(
            neurons_train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_val_subset
        )
        
        # Create dataframes
        train_df = metadata[metadata['neuron_id'].isin(neurons_train)].copy()
        val_df = metadata[metadata['neuron_id'].isin(neurons_val)].copy()
        test_df = metadata[metadata['neuron_id'].isin(neurons_test)].copy()
        
        # Balance classes if requested (cutoff method)
        if self.balance_classes:
            train_df = self._balance_classes(train_df)
        
        # Compute class weights for loss function
        self.class_weights = self._compute_class_weights(train_df)
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_df)} samples ({len(neurons_train)} neurons)")
        print(f"  Val:   {len(val_df)} samples ({len(neurons_val)} neurons)")
        print(f"  Test:  {len(test_df)} samples ({len(neurons_test)} neurons)")
        
        # Get transforms
        train_transform, val_transform = get_transforms(
            image_size=self.image_size,
            augment=self.augment
        )
        
        # Create datasets
        self.train_dataset = NeuronDataset(
            train_df, self.channel, self.root_dir, transform=train_transform
        )
        self.val_dataset = NeuronDataset(
            val_df, self.channel, self.root_dir, transform=val_transform
        )
        self.test_dataset = NeuronDataset(
            test_df, self.channel, self.root_dir, transform=val_transform
        )
        
        # Store class names for model
        self.class_names = self.train_dataset.class_names
    
    def _balance_classes(self, df):
        """Balance classes using cutoff method"""
        class_counts = df.groupby('class_label').size()
        min_count = class_counts.min()
        
        balanced_dfs = []
        for class_label in df['class_label'].unique():
            class_df = df[df['class_label'] == class_label]
            if len(class_df) > min_count:
                class_df = class_df.sample(n=min_count, random_state=self.random_state)
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs).reset_index(drop=True)
    
    def _compute_class_weights(self, df):
        """Compute inverse frequency class weights"""
        class_counts = df.groupby('class_label').size()
        total = len(df)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights.values)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def train_model(
    metadata_path: str,
    root_dir: str,
    channel: str,
    output_dir: str,
    architecture: str = 'resnet18',
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    gpus: int = 1,
    precision: str = '16-mixed',  # Use mixed precision for faster training
    early_stop_patience: int = 15,
    **kwargs
):
    """
    Main training function
    
    Args:
        metadata_path: Path to metadata CSV
        root_dir: Root directory containing images
        channel: 'RFP1' or 'Halo'
        output_dir: Where to save model checkpoints and logs
        architecture: 'resnet18' or 'custom_cnn'
        batch_size: Batch size
        max_epochs: Maximum training epochs
        learning_rate: Learning rate
        gpus: Number of GPUs (0 for CPU)
        precision: Training precision ('32' or '16-mixed')
        early_stop_patience: Early stopping patience
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    data_module = AggregationDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        channel=channel,
        batch_size=batch_size,
        **kwargs
    )
    data_module.setup()
    
    # Initialize model
    model = AggregationClassifier(
        num_classes=len(data_module.class_names),
        class_names=data_module.class_names,
        architecture=architecture,
        learning_rate=learning_rate,
        class_weights=data_module.class_weights
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename=f'{channel}_{architecture}_{{epoch}}_{{val_acc:.3f}}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
        mode='min',
        
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Loggers
    tb_logger = TensorBoardLogger(
        save_dir=output_dir / 'logs',
        name=f'{channel}_{architecture}'
    )
    
    csv_logger = CSVLogger(
        save_dir=output_dir / 'logs',
        name=f'{channel}_{architecture}'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 'auto',
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training {architecture} on {channel} channel")
    print(f"{'='*60}\n")
    
    trainer.fit(model, data_module)
    
    # Test with best model
    print(f"\n{'='*60}")
    print("Testing best model...")
    print(f"{'='*60}\n")
    
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    return model, trainer, test_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train aggregation classifier')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata.csv')
    parser.add_argument('--root_dir', type=str, required=True, help='Root data directory')
    parser.add_argument('--channel', type=str, required=True, choices=['RFP1', 'Halo'])
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--architecture', type=str, default='resnet18', choices=['resnet18', 'custom_cnn'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=str, default='16-mixed')
    
    args = parser.parse_args()
    
    train_model(
        metadata_path=args.metadata,
        root_dir=args.root_dir,
        channel=args.channel,
        output_dir=args.output_dir,
        architecture=args.architecture,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        gpus=args.gpus,
        precision=args.precision
    )
