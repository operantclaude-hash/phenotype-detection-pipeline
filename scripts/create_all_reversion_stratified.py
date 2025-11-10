#!/usr/bin/env python3
"""
Create all three reversion models with stratified sampling
"""
import pandas as pd
import numpy as np

df = pd.read_csv('experiments/2025-11-06_rescue_masked/dataset/metadata.csv')
df_subset = df[df['timepoint'].isin(['T0', 'T15'])].copy()

print("="*70)
print("STRATIFIED REVERSION DATASETS")
print("="*70)

def create_stratified_balanced_dataset(df_subset, class_0_specs, class_1_specs, 
                                       label_0, label_1, filename):
    """
    Stratified sampling: equal samples from each subcategory
    class_0_specs: list of (condition, timepoint, stimulation) tuples
    class_1_specs: list of (condition, timepoint, stimulation) tuples
    """
    
    print(f"\n### {filename} ###")
    
    # Get all subcategories for each class
    class_0_dfs = []
    for cond, tp, stim in class_0_specs:
        subcat = df_subset[(df_subset['condition'] == cond) & 
                          (df_subset['timepoint'] == tp) & 
                          (df_subset['stimulation'] == stim)].copy()
        if len(subcat) > 0:
            print(f"  Class 0 ({cond}_{tp}_{stim}): {len(subcat)} samples")
            class_0_dfs.append(subcat)
    
    class_1_dfs = []
    for cond, tp, stim in class_1_specs:
        subcat = df_subset[(df_subset['condition'] == cond) & 
                          (df_subset['timepoint'] == tp) & 
                          (df_subset['stimulation'] == stim)].copy()
        if len(subcat) > 0:
            print(f"  Class 1 ({cond}_{tp}_{stim}): {len(subcat)} samples")
            class_1_dfs.append(subcat)
    
    # Find minimum subcategory size
    all_sizes = [len(d) for d in class_0_dfs + class_1_dfs]
    min_per_subcat = min(all_sizes)
    
    print(f"\n→ Sampling {min_per_subcat} from each subcategory for fairness")
    
    # Sample equally from each subcategory
    balanced_0_dfs = []
    for subdf in class_0_dfs:
        sampled = subdf.sample(n=min(min_per_subcat, len(subdf)), random_state=42)
        sampled['class_label'] = label_0
        balanced_0_dfs.append(sampled)
    
    balanced_1_dfs = []
    for subdf in class_1_dfs:
        sampled = subdf.sample(n=min(min_per_subcat, len(subdf)), random_state=42)
        sampled['class_label'] = label_1
        balanced_1_dfs.append(sampled)
    
    df_balanced = pd.concat(balanced_0_dfs + balanced_1_dfs, ignore_index=True)
    
    print(f"\nFinal dataset:")
    print(df_balanced['class_label'].value_counts())
    print(f"Total: {len(df_balanced)} samples\n")
    
    df_balanced.to_csv(f'experiments/2025-11-06_rescue_masked/dataset/{filename}', index=False)
    print(f"✅ Saved: {filename}")

# =============================================================================
# Model 1: Non-aggregated vs Aggregated (APP+WT)
# =============================================================================
print("\n" + "="*70)
print("MODEL 1: Non-aggregated vs Aggregated (both genotypes)")
print("="*70)

class_0_m1 = [
    ('APPV717I', 'T0', '10000ms'),
    ('APPV717I', 'T0', '0ms'),
    ('APPV717I', 'T15', '0ms'),
    ('Control', 'T0', '10000ms'),
    ('Control', 'T0', '0ms'),
    ('Control', 'T15', '0ms'),
]

class_1_m1 = [
    ('APPV717I', 'T15', '10000ms'),
    ('Control', 'T15', '10000ms'),
]

create_stratified_balanced_dataset(df_subset, class_0_m1, class_1_m1,
                                   'Non-aggregated', 'Aggregated',
                                   'metadata_reversion_model1_stratified.csv')

# =============================================================================
# Model 2: APP Non-aggregated vs APP Aggregated
# =============================================================================
print("\n" + "="*70)
print("MODEL 2: APP Non-aggregated vs APP Aggregated")
print("="*70)

class_0_m2 = [
    ('APPV717I', 'T0', '10000ms'),
    ('APPV717I', 'T0', '0ms'),
    ('APPV717I', 'T15', '0ms'),
]

class_1_m2 = [
    ('APPV717I', 'T15', '10000ms'),
]

create_stratified_balanced_dataset(df_subset, class_0_m2, class_1_m2,
                                   'APP_Non-aggregated', 'APP_Aggregated',
                                   'metadata_reversion_model2_stratified.csv')

# =============================================================================
# Model 3: Everything vs APP_T15_10000ms (most specific)
# =============================================================================
print("\n" + "="*70)
print("MODEL 3: Everything else vs APP_T15_10000ms (most specific)")
print("="*70)

class_0_m3 = [
    ('APPV717I', 'T0', '10000ms'),
    ('APPV717I', 'T0', '0ms'),
    ('APPV717I', 'T15', '0ms'),
    ('Control', 'T0', '10000ms'),
    ('Control', 'T0', '0ms'),
    ('Control', 'T15', '0ms'),
    ('Control', 'T15', '10000ms'),
]

class_1_m3 = [
    ('APPV717I', 'T15', '10000ms'),
]

create_stratified_balanced_dataset(df_subset, class_0_m3, class_1_m3,
                                   'Baseline', 'APP_Aggregated_Specific',
                                   'metadata_reversion_model3_stratified.csv')

print("\n" + "="*70)
print("STRATIFIED SAMPLING COMPLETE")
print("="*70)
print("Each subcategory contributes equally to the final dataset")
print("="*70)
