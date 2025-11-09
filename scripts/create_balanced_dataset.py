#!/usr/bin/env python3
"""Balance dataset by downsampling majority class"""
import pandas as pd
import numpy as np

def balance_by_downsampling(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    print("="*70)
    print("ORIGINAL DATASET")
    print("="*70)
    print(df['class_label'].value_counts())
    
    # Get minority class size
    min_count = df['class_label'].value_counts().min()
    
    # Downsample each class to match minority
    balanced_dfs = []
    for class_label in df['class_label'].unique():
        class_df = df[df['class_label'] == class_label]
        
        # Downsample to min_count, stratified by neuron to keep temporal info
        if len(class_df) > min_count:
            # Get unique neurons
            neurons = class_df['neuron_id'].unique()
            # Calculate how many neurons we need
            avg_timepoints_per_neuron = len(class_df) / len(neurons)
            neurons_needed = int(min_count / avg_timepoints_per_neuron)
            
            # Sample neurons
            sampled_neurons = np.random.choice(neurons, size=neurons_needed, replace=False)
            class_df = class_df[class_df['neuron_id'].isin(sampled_neurons)]
        
        balanced_dfs.append(class_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    print("\n" + "="*70)
    print("BALANCED DATASET")
    print("="*70)
    print(df_balanced['class_label'].value_counts())
    print(f"\nTotal samples: {len(df_balanced)}")
    print(f"Neurons: {df_balanced['neuron_id'].nunique()}")
    
    df_balanced.to_csv(output_csv, index=False)
    print(f"\n✅ Saved to {output_csv}")

if __name__ == '__main__':
    balance_by_downsampling(
        'experiments/2025-11-06_rescue_masked/dataset/metadata_10000ms.csv',
        'experiments/2025-11-06_rescue_masked/dataset/metadata_balanced.csv'
    )
