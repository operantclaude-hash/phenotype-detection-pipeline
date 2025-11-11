#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import torch
import pandas as pd
from train_lightning_threechannel_improved import ThreeChannelClassifier, ThreeChannelDataModule
from pathlib import Path

def analyze_per_timepoint(checkpoint_path, metadata_path, root_dir):
    model = ThreeChannelClassifier.load_from_checkpoint(
        checkpoint_path, num_classes=2, class_names=['APPV717I', 'Control']
    )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    data_module = ThreeChannelDataModule(metadata_path, root_dir, batch_size=32, num_workers=4)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # Collect predictions with metadata
    all_preds = []
    all_labels = []
    all_timepoints = []
    
    test_df = data_module.test_dataset.metadata.reset_index(drop=True)
    
    with torch.no_grad():
        batch_idx = 0
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            batch_size = len(labels)
            start_idx = batch_idx * data_module.batch_size
            end_idx = start_idx + batch_size
            batch_df = test_df.iloc[start_idx:end_idx]
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_timepoints.extend(batch_df['timepoint'].values)
            
            batch_idx += 1
    
    # Calculate per-timepoint accuracy
    results_df = pd.DataFrame({
        'timepoint': all_timepoints,
        'true_label': all_labels,
        'pred_label': all_preds
    })
    
    print("\n" + "="*70)
    print("PER-TIMEPOINT ACCURACY")
    print("="*70)
    
    for tp in sorted(results_df['timepoint'].unique(), key=lambda x: int(x[1:])):
        tp_data = results_df[results_df['timepoint'] == tp]
        acc = (tp_data['true_label'] == tp_data['pred_label']).mean()
        
        # Per-class within timepoint
        app_data = tp_data[tp_data['true_label'] == 0]
        ctrl_data = tp_data[tp_data['true_label'] == 1]
        
        app_acc = (app_data['true_label'] == app_data['pred_label']).mean() if len(app_data) > 0 else 0
        ctrl_acc = (ctrl_data['true_label'] == ctrl_data['pred_label']).mean() if len(ctrl_data) > 0 else 0
        
        print(f"{tp:4s}: Overall={acc:.3f}  APP={app_acc:.3f}  Control={ctrl_acc:.3f}  (n={len(tp_data)})")
    
    print("="*70)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    
    args = parser.parse_args()
    analyze_per_timepoint(args.checkpoint, args.metadata, args.root_dir)
