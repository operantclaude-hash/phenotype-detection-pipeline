"""
Dual-channel training (RFP1 + Halo) with data augmentation
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

class DualChannelDataset(Dataset):
    """Load both RFP1 and Halo as 2-channel input"""
    def __init__(self, metadata, root_dir, transform=None):
        self.metadata = metadata
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Augmentation for training
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load both channels
        rfp1_path = self.root_dir / row['rfp1_path']
        halo_path = self.root_dir / row['halo_path']
        
        rfp1 = Image.open(rfp1_path).convert('L')
        halo = Image.open(halo_path).convert('L')
        
        # Stack as 2-channel tensor
        if self.transform:
            # Apply same transform to both (for spatial consistency)
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            rfp1_t = transforms.ToTensor()(rfp1)
            
            torch.manual_seed(seed)
            halo_t = transforms.ToTensor()(halo)
            
            # Concatenate channels
            img = torch.cat([rfp1_t, halo_t], dim=0)
            
            # Apply remaining transforms
            img = transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])(img)
            
            # Apply augmentation
            img = transforms.RandomHorizontalFlip(p=0.5)(img)
            img = transforms.RandomVerticalFlip(p=0.5)(img)
            img = transforms.RandomRotation(15)(img)
        else:
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
        
        # ResNet18 with 2 input channels
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
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class DualChannelDataModule(pl.LightningDataModule):
    """Data module with well-based splits for dual-channel"""
    def __init__(self, metadata_path, root_dir, batch_size=32, num_workers=4,
                 test_size=0.15, val_size=0.15):
        super().__init__()
        self.metadata_path = metadata_path
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        
    def setup(self, stage=None):
        metadata = pd.read_csv(self.metadata_path)
        
        print(f"Loaded metadata: {len(metadata)} samples")
        
        # Create label encoding
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
            unique_wells, test_size=self.test_size, random_state=42, stratify=well_labels
        )
        
        well_labels_train_val = well_labels[wells_train_val]
        val_size_adjusted = self.val_size / (1 - self.test_size)
        
        wells_train, wells_val = train_test_split(
            wells_train_val, test_size=val_size_adjusted, random_state=42,
            stratify=well_labels_train_val
        )
        
        print(f"\nWell splits: Train={len(wells_train)}, Val={len(wells_val)}, Test={len(wells_test)}")
        
        train_df = metadata[metadata['well'].isin(wells_train)]
        val_df = metadata[metadata['well'].isin(wells_val)]
        test_df = metadata[metadata['well'].isin(wells_test)]
        
        print(f"Sample splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Augmentation only for training
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.train_dataset = DualChannelDataset(train_df, self.root_dir, transform=train_transform)
        self.val_dataset = DualChannelDataset(val_df, self.root_dir, transform=test_transform)
        self.test_dataset = DualChannelDataset(test_df, self.root_dir, transform=test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                         num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)
