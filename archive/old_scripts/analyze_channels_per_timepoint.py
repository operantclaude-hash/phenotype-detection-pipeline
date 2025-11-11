#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import torch
import pandas as pd
from train_lightning_wells import AggregationClassifier, AggregationDataModuleWells
from pathlib import Path

def analyze_channel_per_timepoint(checkpoint_path, metadata_path, root_dir, channel_name):
    model = AggregationClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Need to create data module with the right channel
    if 'RFP1' in channel_name:
        channel = 'RFP1'
    elif 'Halo_masked' in channel_name:
        channel = 'Halo_masked'
    else:
        channel = 'Halo'
    
    data_module = AggregationDataModuleWells(
        metadata_path=metadata_path,
        root_dir=root_dir,
        channel=channel,
        batch_size=32,
        num_workers=4
    )
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
    
    timepoint_results = {}
    for tp in sorted(results_df['timepoint'].unique(), key=lambda x: int(x[1:])):
        tp_data = results_df[results_df['timepoint'] == tp]
        
        app_data = tp_data[tp_data['true_label'] == 0]
        ctrl_data = tp_data[tp_data['true_label'] == 1]
        
        app_acc = (app_data['true_label'] == app_data['pred_label']).mean() if len(app_data) > 0 else 0
        ctrl_acc = (ctrl_data['true_label'] == ctrl_data['pred_label']).mean() if len(ctrl_data) > 0 else 0
        overall = (tp_data['true_label'] == tp_data['pred_label']).mean()
        
        timepoint_results[tp] = {
            'overall': overall,
            'app': app_acc,
            'control': ctrl_acc
        }
    
    return timepoint_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--rfp1_ckpt', required=True)
    parser.add_argument('--halo_ckpt', required=True)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PER-CHANNEL, PER-TIMEPOINT ACCURACY")
    print("="*70)
    
    # Analyze each channel
    rfp1_results = analyze_channel_per_timepoint(
        args.rfp1_ckpt, args.metadata, args.root_dir, 'RFP1'
    )
    
    halo_results = analyze_channel_per_timepoint(
        args.halo_ckpt, args.metadata, args.root_dir, 'Halo'
    )
    
    # Print comparison table
    timepoints = sorted(rfp1_results.keys(), key=lambda x: int(x[1:]))
    
    print(f"\n{'TP':4s}  {'RFP1':^30s}  {'Halo':^30s}")
    print(f"{'':4s}  {'Overall':>8s} {'APP':>8s} {'Control':>8s}  {'Overall':>8s} {'APP':>8s} {'Control':>8s}")
    print("-" * 70)
    
    for tp in timepoints:
        r = rfp1_results[tp]
        h = halo_results[tp]
        print(f"{tp:4s}  {r['overall']:8.3f} {r['app']:8.3f} {r['control']:8.3f}  "
              f"{h['overall']:8.3f} {h['app']:8.3f} {h['control']:8.3f}")
    
    print("="*70)
