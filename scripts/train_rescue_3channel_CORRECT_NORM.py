#!/usr/bin/env python3
"""
Train RESCUE 3-Channel Model - CORRECT NORMALIZATION
Uses EXACT same normalization as 6-channel models:
1. Percentile clipping [p1, p99]
2. Normalize to [0, 1]
3. ImageNet normalization

For APP vs WT classification at single timepoints.
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
import numpy as np
import json
import argparse
from torchmetrics import Accuracy

class RescueThreeChannelDataset(Dataset):
    """
    3-channel RESCUE dataset with CORRECT normalization (same as 6-channel)
    """
    def __init__(self, csv_path, percentiles_path='A2screen_combined/data/intensity_percentiles.json', is_train=False):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train

        # Load intensity percentiles
        with open(percentiles_path) as f:
            self.percentiles = json.load(f)

        print(f"\n✓ Loaded intensity percentiles:")
        for channel, bounds in self.percentiles.items():
            print(f"  {channel.upper()}: p1={bounds['p1']:.0f}, p99={bounds['p99']:.0f}")

        print(f"\nDataset: {len(self.df)} images")
        print(f"  Neurons: {self.df['neuron_id'].nunique()}")
        print(f"  WT: {(self.df['genotype'] == 'WT').sum()}")
        print(f"  APP: {(self.df['genotype'] == 'APP').sum()}")

        plate1_count = (self.df['plate'] == 'plate1').sum()
        plate2_count = (self.df['plate'] == 'plate2').sum()
        print(f"  Plate1: {plate1_count}, Plate2: {plate2_count}")

        # Resize transform
        self.resize = transforms.Resize((224, 224))

        # ImageNet normalization (applied AFTER percentile normalization) - SAME AS 6-CHANNEL
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self.is_train:
            self.augment = transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(90)
            ], p=0.5)
        else:
            self.augment = None

    def load_and_normalize_channel(self, path, channel_name):
        """
        CORRECT normalization - SAME AS 6-CHANNEL:
        1. Load 16-bit image
        2. Clip to percentile bounds [p1, p99]
        3. Normalize to [0, 1]: (img - p1) / (p99 - p1)
        4. Return as tensor
        """
        # Load 16-bit image as numpy array
        img = np.array(Image.open(path), dtype=np.float32)

        # Get percentile bounds for this channel
        p1 = self.percentiles[channel_name]['p1']
        p99 = self.percentiles[channel_name]['p99']

        # Step 1: Clip to percentile bounds
        img_clipped = np.clip(img, p1, p99)

        # Step 2: Normalize to [0, 1]
        if p99 > p1:
            img_normalized = (img_clipped - p1) / (p99 - p1)
        else:
            img_normalized = img_clipped

        # Convert to PIL for transforms, then to tensor
        img_pil = Image.fromarray((img_normalized * 255).astype(np.uint8))
        img_pil = self.resize(img_pil)

        # Convert to tensor (now in [0, 1])
        img_tensor = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).unsqueeze(0)

        return img_tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        root_dir = Path(row['root_dir'])

        # Load 3 channels with CORRECT normalization
        imgs = []
        for path_key, channel_name in [('rfp1_path', 'rfp1'),
                                        ('halo_path', 'halo'),
                                        ('halo_masked_path', 'halo')]:
            img_path = root_dir / row[path_key]
            img_tensor = self.load_and_normalize_channel(img_path, channel_name)
            imgs.append(img_tensor)

        # Concatenate to 3-channel image
        img = torch.cat(imgs, dim=0)

        # Apply augmentation if training
        if self.augment:
            img = self.augment(img)

        # Apply ImageNet normalization (SAME AS 6-CHANNEL)
        img = self.normalize(img)

        # Label: WT=0, APP=1
        label = 1 if row['genotype'] == 'APP' else 0

        return img, label


class RescueClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, class_weights=None, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes, in_chans=3)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed
    pl.seed_everything(args.seed)

    print(f"\n{'='*80}")
    print(f"TRAINING RESCUE 3-CHANNEL MODEL - CORRECT NORMALIZATION")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")

    # Load datasets
    train_csv = f'{args.data_dir}/train.csv'
    val_csv = f'{args.data_dir}/val.csv'
    test_csv = f'{args.data_dir}/test.csv'

    train_ds = RescueThreeChannelDataset(train_csv, is_train=True)
    val_ds = RescueThreeChannelDataset(val_csv, is_train=False)
    test_ds = RescueThreeChannelDataset(test_csv, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Compute class weights
    train_df = pd.read_csv(train_csv)
    wt_count = (train_df['genotype'] == 'WT').sum()
    app_count = (train_df['genotype'] == 'APP').sum()
    total = len(train_df)

    class_weights = [total / (2 * wt_count), total / (2 * app_count)]
    print(f"\nClass weights: WT={class_weights[0]:.3f}, APP={class_weights[1]:.3f}")

    # Create model
    model = RescueClassifier(num_classes=2, class_weights=class_weights, learning_rate=1e-4)

    # Callbacks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{args.output_dir}/checkpoints',
        filename='best-epoch={epoch:02d}-val_acc={val_acc:.3f}',
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
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=f'{args.output_dir}/lightning_logs',
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_acc: {checkpoint_callback.best_model_score:.4f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
