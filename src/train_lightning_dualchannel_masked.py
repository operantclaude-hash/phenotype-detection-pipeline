"""
Dual-channel training with MASKED images (RFP1_masked + Halo_masked)
"""
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

class DualChannelMaskedDataset(Dataset):
    """Load both RFP1_masked and Halo_masked as 2-channel input"""
    def __init__(self, metadata, root_dir, transform=None):
        self.metadata = metadata
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load MASKED channels
        rfp1_path = self.root_dir / row['rfp1_masked_path']
        halo_path = self.root_dir / row['halo_masked_path']
        
        rfp1 = Image.open(rfp1_path).convert('L')
        halo = Image.open(halo_path).convert('L')
        
        rfp1_t = transforms.ToTensor()(rfp1)
        halo_t = transforms.ToTensor()(halo)
        
        img = torch.cat([rfp1_t, halo_t], dim=0)
        img = transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])(img)
        
        label = row['label_idx']
        return img, label

class DualChannelClassifier(pl.LightningModule):
    """ResNet18 with 2-channel input"""
    def __init__(self, num_classes, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

class DualChannelMaskedDataModule(pl.LightningDataModule):
    """Data module with well-based splits for masked dual-channel"""
    def __init__(self, metadata_path, root_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.metadata_path = metadata_path
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        metadata = pd.read_csv(self.metadata_path)
        
        unique_labels = sorted(metadata['class_label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        metadata['label_idx'] = metadata['class_label'].map(label_to_idx)
        
        self.num_classes = len(unique_labels)
        self.class_names = unique_labels
        
        print(f"Classes: {self.num_classes} - {self.class_names}")
        
        # Well-based splits
        unique_wells = metadata['well'].unique()
        well_labels = metadata.groupby('well')['class_label'].first()
        
        wells_train_val, wells_test = train_test_split(
            unique_wells, test_size=0.15, random_state=42, stratify=well_labels
        )
        
        well_labels_train_val = well_labels[wells_train_val]
        wells_train, wells_val = train_test_split(
            wells_train_val, test_size=0.15/(1-0.15), random_state=42, stratify=well_labels_train_val
        )
        
        train_df = metadata[metadata['well'].isin(wells_train)]
        val_df = metadata[metadata['well'].isin(wells_val)]
        test_df = metadata[metadata['well'].isin(wells_test)]
        
        print(f"Samples: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        self.train_dataset = DualChannelMaskedDataset(train_df, self.root_dir)
        self.val_dataset = DualChannelMaskedDataset(val_df, self.root_dir)
        self.test_dataset = DualChannelMaskedDataset(test_df, self.root_dir)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                         num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, persistent_workers=True)
