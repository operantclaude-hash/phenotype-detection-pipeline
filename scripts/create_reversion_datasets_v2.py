#!/usr/bin/env python3
"""
Create properly balanced reversion datasets
"""
import pandas as pd
import numpy as np

df = pd.read_csv('experiments/2025-11-06_rescue_masked/dataset/metadata.csv')
df_subset = df[df['timepoint'].isin(['T0', 'T15'])].copy()

print("="*70)
print("REVERSION DATASETS V2 (PROPERLY BALANCED)")
print("="*70)

def create_balanced_dataset(df_subset, class_0_conditions, class_1_conditions, 
                            label_0, label_1, filename):
    """Helper to create properly balanced datasets"""
    
    # Create masks for each class
    mask_0 = pd.concat([pd.Series(cond) for cond in class_0_conditions]).groupby(level=0).any()
    mask_1 = pd.concat([pd.Series(cond) for cond in class_1_conditions]).groupby(level=0).any()
    
    df_0 = df_subset[mask_0].copy()
    df_0['class_label'] = label_0
    
    df_1 = df_subset[mask_1].copy()
    df_1['class_label'] = label_1
    
    print(f"\n### {filename} ###")
    print(f"Class 0 ({label_0}): {len(df_0)} samples, {df_0['neuron_id'].nunique()} neurons")
    print(f"Class 1 ({label_1}): {len(df_1)} samples, {df_1['neuron_id'].nunique()} neurons")
    
    # Balance by taking minimum
    min_samples = min(len(df_0), len(df_1))
    
    # Randomly sample from each class
    df_0_sampled = df_0.sample(n=min_samples, random_state=42)
    df_1_sampled = df_1.sample(n=min_samples, random_state=42)
    
    df_balanced = pd.concat([df_0_sampled, df_1_sampled], ignore_index=True)
    
    print(f"Balanced: {len(df_balanced)} total ({min_samples} each class)")
    print(df_balanced['class_label'].value_counts())
    
    df_balanced.to_csv(f'experiments/2025-11-06_rescue_masked/dataset/{filename}', index=False)
    print(f"✅ Saved: {filename}")
    
    return df_balanced

# =============================================================================
# Model 1: All non-aggregated vs Aggregated (APP+WT)
# =============================================================================
class_0_m1 = [
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '10000ms'),
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '10000ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '0ms'),
]

class_1_m1 = [
    (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '10000ms')
]

create_balanced_dataset(df_subset, class_0_m1, class_1_m1, 
                       'Non-aggregated', 'Aggregated', 
                       'metadata_reversion_model1_balanced.csv')

# =============================================================================
# Model 2: APP baseline vs APP aggregated
# =============================================================================
df_app = df_subset[df_subset['condition'] == 'APPV717I'].copy()

class_0_m2 = [
    (df_app['timepoint'] == 'T0') & (df_app['stimulation'] == '10000ms'),
    (df_app['timepoint'] == 'T0') & (df_app['stimulation'] == '0ms'),
    (df_app['timepoint'] == 'T15') & (df_app['stimulation'] == '0ms'),
]

class_1_m2 = [
    (df_app['timepoint'] == 'T15') & (df_app['stimulation'] == '10000ms')
]

create_balanced_dataset(df_app, class_0_m2, class_1_m2,
                       'APP_Non-aggregated', 'APP_Aggregated',
                       'metadata_reversion_model2_balanced.csv')

# =============================================================================
# Model 3: Everything except APP_T15_10000ms vs APP_T15_10000ms (MOST SPECIFIC)
# =============================================================================
class_0_m3 = [
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '10000ms'),
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '10000ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '10000ms'),  # Add WT T15 10000ms!
]

class_1_m3 = [
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '10000ms')
]

create_balanced_dataset(df_subset, class_0_m3, class_1_m3,
                       'Baseline', 'APP_Aggregated_Specific',
                       'metadata_reversion_model3_balanced.csv')

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Model 1: Aggregation detection (APP or WT)")
print("Model 2: APP-specific aggregation detection")
print("Model 3: APP aggregation vs everything else (most specific)")
print("="*70)
