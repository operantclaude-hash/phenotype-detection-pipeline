#!/usr/bin/env python3
"""Evaluate binary classification model"""
import sys
sys.path.insert(0, 'src')
from train_lightning import AggregationClassifier, AggregationDataModule
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def evaluate_binary(checkpoint, metadata, root_dir, channel, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model with 2 classes
    model = AggregationClassifier.load_from_checkpoint(
        checkpoint,
        num_classes=2,
        class_names=['Control', 'APPV717I']
    )
    model.eval()
    
    # Load data
    data_module = AggregationDataModule(
        metadata_path=metadata,
        root_dir=root_dir,
        channel=channel,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            images, labels = batch
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n✅ {channel} Binary Accuracy: {accuracy:.3f}")
    
    # Save
    metrics = {'test_accuracy': float(accuracy), 'num_classes': 2, 'channel': channel}
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'APPV717I'],
                yticklabels=['Control', 'APPV717I'])
    plt.title(f'Binary: {channel}\nAcc: {accuracy:.3f}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    print(f"Saved: {output_dir}/confusion_matrix.png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--channel', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    evaluate_binary(args.checkpoint, args.metadata, args.root_dir, args.channel, args.output_dir)
