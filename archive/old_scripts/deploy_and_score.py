#!/usr/bin/env python3
"""
Deploy trained models on new data
Outputs: Raw CSV files with confidence scores per neuron
NO interpretation - just numbers for downstream analysis
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from train_lightning import AggregationClassifier, AggregationDataModule
from tqdm import tqdm

def deploy_model(checkpoint_path, metadata_path, root_dir, channel, 
                 model_name, output_file):
    """
    Deploy one model and output scores
    """
    print(f"\n{'='*70}")
    print(f"DEPLOYING: {model_name} - {channel}")
    print(f"{'='*70}")
    
    # Load model
    model = AggregationClassifier.load_from_checkpoint(
        checkpoint_path,
        num_classes=2
    )
    model.eval()
    
    # Load data
    data_module = AggregationDataModule(
        metadata_path=metadata_path,
        root_dir=root_dir,
        channel=channel,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get metadata
    metadata = pd.read_csv(metadata_path)
    
    # Run inference
    all_confidences = []
    all_predictions = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_module.test_dataloader(), 
                                               desc=f"{channel}")):
            images, labels = batch
            images = images.to(device)
            
            # Get softmax probabilities
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_confidences.append(probs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            
            # Track indices
            batch_size = images.shape[0]
            start_idx = batch_idx * 32
            all_indices.extend(range(start_idx, start_idx + batch_size))
    
    # Combine results
    confidences = np.vstack(all_confidences)
    predictions = np.concatenate(all_predictions)
    
    # Create results dataframe
    results = metadata.iloc[all_indices].copy()
    results['confidence_class0'] = confidences[:, 0]
    results['confidence_class1'] = confidences[:, 1]
    results['predicted_class'] = predictions
    
    # Save
    results.to_csv(output_file, index=False)
    
    print(f"✅ Saved: {output_file}")
    print(f"   Samples: {len(results)}")
    
    return results

def aggregate_by_tile(scores_df, output_file):
    """Aggregate neuron scores to tile level"""
    agg = scores_df.groupby(['well', 'tile', 'timepoint', 
                             'condition', 'stimulation']).agg({
        'neuron_id': 'count',
        'confidence_class0': ['mean', 'std'],
        'confidence_class1': ['mean', 'std'],
        'predicted_class': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    agg.columns = [
        'well', 'tile', 'timepoint', 'condition', 'stimulation',
        'n_neurons',
        'mean_confidence_class0', 'std_confidence_class0',
        'mean_confidence_class1', 'std_confidence_class1',
        'mean_predicted_class', 'std_predicted_class'
    ]
    
    agg['fraction_predicted_class1'] = agg['mean_predicted_class']
    agg = agg.drop(['mean_predicted_class', 'std_predicted_class'], axis=1)
    
    agg.to_csv(output_file, index=False)
    print(f"✅ Aggregated: {output_file}")
    
    return agg

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--reversion_checkpoint_rfp1', required=True)
    parser.add_argument('--reversion_checkpoint_halo', required=True)
    parser.add_argument('--rescue_checkpoint_rfp1', required=True)
    parser.add_argument('--rescue_checkpoint_halo', required=True)
    parser.add_argument('--output_dir', required=True)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("MODEL DEPLOYMENT")
    print("="*70)
    
    # Deploy reversion model
    print("\n--- REVERSION MODEL (0=pre-stim, 1=post-stim) ---")
    rev_rfp1 = deploy_model(
        args.reversion_checkpoint_rfp1, args.metadata, args.root_dir, 'RFP1',
        'Reversion', output_dir / 'reversion_scores_rfp1_neurons.csv'
    )
    rev_halo = deploy_model(
        args.reversion_checkpoint_halo, args.metadata, args.root_dir, 'Halo',
        'Reversion', output_dir / 'reversion_scores_halo_neurons.csv'
    )
    
    # Deploy rescue model
    print("\n--- RESCUE MODEL (0=WT, 1=APP) ---")
    resc_rfp1 = deploy_model(
        args.rescue_checkpoint_rfp1, args.metadata, args.root_dir, 'RFP1',
        'Rescue', output_dir / 'rescue_scores_rfp1_neurons.csv'
    )
    resc_halo = deploy_model(
        args.rescue_checkpoint_halo, args.metadata, args.root_dir, 'Halo',
        'Rescue', output_dir / 'rescue_scores_halo_neurons.csv'
    )
    
    # Aggregate
    print("\n--- AGGREGATING TO TILE LEVEL ---")
    aggregate_by_tile(rev_rfp1, output_dir / 'reversion_scores_rfp1_tiles.csv')
    aggregate_by_tile(rev_halo, output_dir / 'reversion_scores_halo_tiles.csv')
    aggregate_by_tile(resc_rfp1, output_dir / 'rescue_scores_rfp1_tiles.csv')
    aggregate_by_tile(resc_halo, output_dir / 'rescue_scores_halo_tiles.csv')
    
    print("\n" + "="*70)
    print("COMPLETE! Ready for R analysis")
    print("="*70)

if __name__ == '__main__':
    main()
