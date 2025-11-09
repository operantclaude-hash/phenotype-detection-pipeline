"""
Training with WELL-BASED splits (no data leakage across wells)
Based on train_lightning.py but splits by wells instead of neurons
"""
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

class NeuronDataset(Dataset):
    """Dataset that loads neuron images"""
    def __init__(self, metadata, root_dir, channel, transform=None):
        self.metadata = metadata
        self.root_dir = Path(root_dir)
        self.channel = channel
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = self.root_dir / row[f'{self.channel.lower()}_path']
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        label = row['label_idx']
        return image, label

class AggregationClassifier(pl.LightningModule):
    """ResNet18-based classifier"""
    def __init__(self, num_classes, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # ResNet18 with 1 input channel
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class AggregationDataModuleWells(pl.LightningDataModule):
    """Data module with WELL-BASED splits"""
    def __init__(self, metadata_path, root_dir, channel, batch_size=32, num_workers=4,
                 test_size=0.15, val_size=0.15):
        super().__init__()
        self.metadata_path = metadata_path
        self.root_dir = root_dir
        self.channel = channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        
    def setup(self, stage=None):
        """Split by WELLS, not neurons"""
        metadata = pd.read_csv(self.metadata_path)
        
        print(f"Loaded metadata: {len(metadata)} samples")
        print(f"Channel: {self.channel}")
        
        # Create label encoding
        unique_labels = sorted(metadata['class_label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        metadata['label_idx'] = metadata['class_label'].map(label_to_idx)
        
        self.num_classes = len(unique_labels)
        self.class_names = unique_labels
        
        print(f"Classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        
        # CRITICAL: Split by WELLS to avoid leakage
        unique_wells = metadata['well'].unique()
        
        # Get class distribution per well for stratification
        well_labels = metadata.groupby('well')['class_label'].first()
        
        # Test split (by wells)
        wells_train_val, wells_test = train_test_split(
            unique_wells,
            test_size=self.test_size,
            random_state=42,
            stratify=well_labels
        )
        
        # Val split (by wells)
        well_labels_train_val = well_labels[wells_train_val]
        val_size_adjusted = self.val_size / (1 - self.test_size)
        
        wells_train, wells_val = train_test_split(
            wells_train_val,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=well_labels_train_val
        )
        
        print(f"\nWell-based splits:")
        print(f"  Train wells: {len(wells_train)}")
        print(f"  Val wells: {len(wells_val)}")
        print(f"  Test wells: {len(wells_test)}")
        
        # Create datasets
        train_df = metadata[metadata['well'].isin(wells_train)]
        val_df = metadata[metadata['well'].isin(wells_val)]
        test_df = metadata[metadata['well'].isin(wells_test)]
        
        print(f"\nSample splits:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        print(f"\nClass distribution:")
        print(f"Train:\n{train_df['class_label'].value_counts()}")
        print(f"Val:\n{val_df['class_label'].value_counts()}")
        print(f"Test:\n{test_df['class_label'].value_counts()}")
        
        self.train_dataset = NeuronDataset(train_df, self.root_dir, self.channel)
        self.val_dataset = NeuronDataset(val_df, self.root_dir, self.channel)
        self.test_dataset = NeuronDataset(test_df, self.root_dir, self.channel)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
