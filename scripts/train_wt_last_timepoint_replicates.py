#!/usr/bin/env python3
"""
Train WT treatment replicate using only last timepoints (T14 for 0ms, T15 for 10000ms).
"""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
from pathlib import Path
import timm
import argparse
from torchmetrics import Accuracy

# Set seed from command line
parser = argparse.ArgumentParser()
parser.add_argument('replicate_id', type=int, help='Replicate ID (1-10)')
args = parser.parse_args()

replicate_id = args.replicate_id
seed = 42 + replicate_id

print(f"\n{'='*80}")
print(f"TRAINING WT TREATMENT REPLICATE {replicate_id:02d} (LAST TIMEPOINT ONLY)")
print(f"Seed: {seed}")
print(f"{'='*80}\n")

pl.seed_everything(seed)

class BinaryTemporalDataset(Dataset):
    """Dataset for temporal classification (T0 vs T15)"""
    def __init__(self, csv_path, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train

        print(f"Dataset: {len(self.df)} images")
        print(f"  Class 0 (pre): {(self.df['class_binary'] == 0).sum()}")
        print(f"  Class 1 (post): {(self.df['class_binary'] == 1).sum()}")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.2233, 0.2073, 0.0507],
            std=[0.1644, 0.1699, 0.1601]
        )

        if self.is_train:
            self.augment = transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(90)
            ], p=0.5)
        else:
            self.augment = None

    def center_crop_224(self, img):
        """Center crop from any size to 224x224"""
        w, h = img.size
        if w == 224 and h == 224:
            return img
        elif w == 300 and h == 300:
            return img.crop((38, 38, 262, 262))
        else:
            return img.resize((224, 224), Image.BILINEAR)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        root_dir = Path(row['root_dir'])

        # Load 3 channels
        imgs = []
        for key in ['rfp1_path', 'halo_path', 'halo_masked_path']:
            img_path = root_dir / row[key]
            img = Image.open(img_path).convert('L')
            img = self.center_crop_224(img)
            img_t = self.to_tensor(img)
            imgs.append(img_t)

        img = torch.cat(imgs, dim=0)

        if self.augment:
            img = self.augment(img)

        img = self.normalize(img)

        label = int(row['class_binary'])
        return img, label

class BinaryClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, class_weights=None, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
        self.learning_rate = learning_rate

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Load datasets
train_csv = 'A2screen_combined/data/wt_treatment_LAST_TIMEPOINT/train.csv'
val_csv = 'A2screen_combined/data/wt_treatment_LAST_TIMEPOINT/val.csv'
test_csv = 'A2screen_combined/data/wt_treatment_LAST_TIMEPOINT/test.csv'

train_ds = BinaryTemporalDataset(train_csv, is_train=True)
val_ds = BinaryTemporalDataset(val_csv, is_train=False)
test_ds = BinaryTemporalDataset(test_csv, is_train=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

# Compute class weights
train_df = pd.read_csv(train_csv)
class_counts = train_df['class_binary'].value_counts().sort_index()
total = len(train_df)
class_weights = [total / (2 * count) for count in class_counts]
print(f"Class weights: {class_weights}")

# Create model
model = BinaryClassifier(num_classes=2, class_weights=class_weights, learning_rate=1e-4)

# Callbacks
output_dir = f'A2screen_combined/models/wt_treatment_LAST_TIMEPOINT_replicates/replicate_{replicate_id:02d}'
Path(output_dir).mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=f'{output_dir}/checkpoints',
    filename='best-{epoch:02d}-{val_acc:.3f}',
    monitor='val_acc',
    mode='max',
    save_top_k=1
)

early_stop_callback = EarlyStopping(
    monitor='val_acc',
    patience=10,
    mode='max',
    verbose=True
)

# Trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir=f'{output_dir}/lightning_logs',
    enable_progress_bar=True,
    log_every_n_steps=10
)

# Train
trainer.fit(model, train_loader, val_loader)

print(f"\n{'='*80}")
print(f"REPLICATE {replicate_id:02d} TRAINING COMPLETE")
print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
print(f"Best val_acc: {checkpoint_callback.best_model_score:.4f}")
print(f"{'='*80}\n")
