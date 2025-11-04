#!/usr/bin/env python3
"""
Evaluate trained 6-class phenotype detection models
Generates confusion matrices and performance metrics
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
import json

# Import training modules
sys.path.insert(0, 'src')
from train_lightning import AggregationClassifier, AggregationDataModule


# 6-class configuration
CLASS_CONFIG = {
    'label_column': 'label_6class',
    'num_classes': 6,
    'class_names': [
        'Control_T0',
        'APPV717I_T0',
        'Control_0ms_T16',
        'Control_10000ms_T16',
        'APPV717I_0ms_T16',
        'APPV717I_10000ms_T16'
    ],
    'description': '6-class: Collapse T0 pre-stim, separate T16 post-stim'
}


def evaluate_model(checkpoint_path, metadata_path, root_dir, channel, output_dir):
    """
    Evaluate a trained model and generate comprehensive reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"EVALUATING: {CLASS_CONFIG['description']}")
    print(f"Channel: {channel}")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*80)
    
    # Load model
    model = AggregationClassifier.load_from_checkpoint(
        checkpoint_path,
        num_classes=CLASS_CONFIG['num_classes'],
        class_names=CLASS_CONFIG['class_names']
    )
    model.eval()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create data module
    data_module = AggregationDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        channel=channel,
        batch_size=32,
        num_workers=4,
        image_size=224,
        stratify_by=CLASS_CONFIG['label_column']
    )
    data_module.setup()
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(data_module.train_dataset)} samples")
    print(f"  Val:   {len(data_module.val_dataset)} samples")
    print(f"  Test:  {len(data_module.test_dataset)} samples")
    
    # Get test loader
    test_loader = data_module.test_dataloader()
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest accuracy: {accuracy:.3f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(accuracy),
        'per_class_accuracy': {
            CLASS_CONFIG['class_names'][i]: float(class_acc[i])
            for i in range(CLASS_CONFIG['num_classes'])
        },
        'confusion_matrix_shape': cm.shape,
        'num_test_samples': len(all_labels)
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved: {output_dir / 'metrics.json'}")
    
    # Save numpy arrays
    np.save(output_dir / 'confusion_matrix.npy', cm)
    np.save(output_dir / 'confusion_matrix_normalized.npy', cm_normalized)
    print(f"Saved: {output_dir / 'confusion_matrix.npy'}")
    print(f"Saved: {output_dir / 'confusion_matrix_normalized.npy'}")
    
    # Plot confusion matrix (raw counts)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_CONFIG['class_names'],
                yticklabels=CLASS_CONFIG['class_names'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix: {CLASS_CONFIG["description"]}\n{channel} Channel (Acc: {accuracy:.3f})',
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    plt.close()
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_CONFIG['class_names'],
                yticklabels=CLASS_CONFIG['class_names'],
                vmin=0, vmax=1,
                cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Normalized Confusion Matrix: {CLASS_CONFIG["description"]}\n{channel} Channel',
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'confusion_matrix_normalized.png'}")
    plt.close()
    
    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_CONFIG['class_names'],
        digits=3
    )
    
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"Saved: {output_dir / 'classification_report.txt'}")
    print("\n" + "="*80)
    print("✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate 6-class phenotype detection model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
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
        '--channel',
        type=str,
        required=True,
        choices=['RFP1', 'Halo'],
        help='Image channel to evaluate'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for evaluation results'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        root_dir=args.root_dir,
        channel=args.channel,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
