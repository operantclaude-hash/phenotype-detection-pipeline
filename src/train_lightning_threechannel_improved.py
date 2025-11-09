"""
Three-channel improved: [RFP1, Halo, Halo_masked]
- Class imbalance handling with weighted loss
- More aggressive augmentation
- Per-class metrics
- Easy architecture swapping
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, ConfusionMatrix
import numpy as np

class ThreeChannelDataset(Dataset):
    """Load RFP1, Halo, and Halo_masked as 3 channels"""
    def __init__(self, metadata, root_dir, is_train=False):
        self.metadata = metadata
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # More aggressive augmentation for training
        if is_train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30),  # Increased
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
                transforms.ColorJitter(brightness=0.3, contrast=0.3),  # More aggressive
            ])
        else:
            self.augment = None
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        rfp1_path = self.root_dir / row['rfp1_path']
        halo_path = self.root_dir / row['halo_path']
        halo_masked_path = self.root_dir / row['halo_masked_path']
        
        rfp1 = Image.open(rfp1_path).convert('L')
        halo = Image.open(halo_path).convert('L')
        halo_masked = Image.open(halo_masked_path).convert('L')
        
        rfp1_t = transforms.ToTensor()(rfp1)
        halo_t = transforms.ToTensor()(halo)
        halo_masked_t = transforms.ToTensor()(halo_masked)
        
        img = torch.cat([rfp1_t, halo_t, halo_masked_t], dim=0)
        
        if self.augment:
            img = self.augment(img)
        
        img = self.normalize(img)
        
        label = row['label_idx']
        return img, label

class ThreeChannelClassifier(pl.LightningModule):
    """ResNet with class weighting and per-class metrics"""
    def __init__(self, num_classes, class_names, class_weights=None, 
                 learning_rate=1e-4, architecture='resnet18'):
        super().__init__()
        self.save_hyperparameters()
        
        # Easy architecture swap: resnet18, resnet34, resnet50
        self.model = timm.create_model(
            architecture,
            pretrained=True,
            num_classes=num_classes
        )
        
        # Class-weighted loss
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.class_names = class_names
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        
        # Per-class accuracy
        self.test_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.confusion_matrix = ConfusionMatrix(
            task='multiclass', num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        self.train_acc(preds, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        self.val_acc(preds, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        self.test_acc(preds, labels)
        self.test_acc_per_class(preds, labels)
        self.confusion_matrix(preds, labels)
        
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        return {'test_loss': loss, 'test_acc': self.test_acc}
    
    def on_test_epoch_end(self):
        # Print per-class accuracy
        per_class_acc = self.test_acc_per_class.compute()
        
        print("\n" + "="*70)
        print("PER-CLASS ACCURACY")
        print("="*70)
        for i, (name, acc) in enumerate(zip(self.class_names, per_class_acc)):
            print(f"{name:20s}: {acc:.3f}")
        
        # Print confusion matrix
        cm = self.confusion_matrix.compute().cpu().numpy()
        print("\nConfusion Matrix:")
        print(f"{'':20s} {'Pred ' + self.class_names[0]:20s} {'Pred ' + self.class_names[1]:20s}")
        for i, name in enumerate(self.class_names):
            print(f"{'True ' + name:20s} {cm[i, 0]:20d} {cm[i, 1]:20d}")
        print("="*70)
    
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

class ThreeChannelDataModule(pl.LightningDataModule):
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
        
        # Calculate class weights for imbalanced dataset
        class_counts = metadata['class_label'].value_counts()
        total = len(metadata)
        self.class_weights = torch.tensor([
            total / class_counts[label] for label in unique_labels
        ], dtype=torch.float32)
        
        print(f"\nClasses: {unique_labels}")
        print(f"Class counts: {class_counts.to_dict()}")
        print(f"Class weights: {self.class_weights.numpy()}")
        
        # Well-based splits (no data leakage)
        unique_wells = metadata['well'].unique()
        well_labels = metadata.groupby('well')['class_label'].first()
        
        wells_train_val, wells_test = train_test_split(
            unique_wells, test_size=0.15, random_state=42, stratify=well_labels
        )
        
        well_labels_train_val = well_labels[wells_train_val]
        wells_train, wells_val = train_test_split(
            wells_train_val, test_size=0.15/(1-0.15), random_state=42,
            stratify=well_labels_train_val
        )
        
        train_df = metadata[metadata['well'].isin(wells_train)]
        val_df = metadata[metadata['well'].isin(wells_val)]
        test_df = metadata[metadata['well'].isin(wells_test)]
        
        print(f"Samples: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        self.train_dataset = ThreeChannelDataset(train_df, self.root_dir, is_train=True)
        self.val_dataset = ThreeChannelDataset(val_df, self.root_dir, is_train=False)
        self.test_dataset = ThreeChannelDataset(test_df, self.root_dir, is_train=False)
    
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
