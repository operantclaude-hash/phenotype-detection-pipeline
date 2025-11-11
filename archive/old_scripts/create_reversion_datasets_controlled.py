#!/usr/bin/env python3
"""
Create controlled reversion datasets that isolate stimulation effects
"""
import pandas as pd
import numpy as np

df = pd.read_csv('experiments/2025-11-06_rescue_masked/dataset/metadata.csv')

# Filter to relevant timepoints and conditions
df_subset = df[df['timepoint'].isin(['T0', 'T15'])].copy()

print("="*70)
print("CONTROLLED REVERSION DATASETS")
print("="*70)

# =============================================================================
# Model 1: All non-aggregated vs Aggregated (APP+WT)
# =============================================================================
print("\n### Model 1: Non-aggregated vs Aggregated (both genotypes) ###")

# Class 0: Everything without T15 10000ms stimulation
non_agg_conditions = [
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '10000ms'),
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'APPV717I') & (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '10000ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T0') & (df_subset['stimulation'] == '0ms'),
    (df_subset['condition'] == 'Control') & (df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '0ms'),
]

df_model1 = df_subset.copy()
df_model1['class_label'] = 'Non-aggregated'

# Class 1: T15 with 10000ms (aggregated)
agg_mask = ((df_subset['timepoint'] == 'T15') & (df_subset['stimulation'] == '10000ms'))
df_model1.loc[agg_mask, 'class_label'] = 'Aggregated'

# Only keep the conditions we want
keep_mask = pd.concat([pd.Series(cond) for cond in non_agg_conditions]).groupby(level=0).any() | agg_mask
df_model1 = df_model1[keep_mask].copy()

print(f"\nOriginal class counts:")
print(df_model1['class_label'].value_counts())

# Balance by downsampling
min_count = df_model1['class_label'].value_counts().min()
balanced_dfs = []
for label in ['Non-aggregated', 'Aggregated']:
    class_df = df_model1[df_model1['class_label'] == label]
    
    if len(class_df) > min_count:
        neurons = class_df['neuron_id'].unique()
        sampled_neurons = np.random.choice(neurons, size=min_count, replace=False)
        class_df = class_df[class_df['neuron_id'].isin(sampled_neurons)]
    
    balanced_dfs.append(class_df)

df_model1_balanced = pd.concat(balanced_dfs, ignore_index=True)

print(f"\nBalanced class counts:")
print(df_model1_balanced['class_label'].value_counts())
print(f"Neurons: {df_model1_balanced['neuron_id'].nunique()}")

df_model1_balanced.to_csv('experiments/2025-11-06_rescue_masked/dataset/metadata_reversion_model1.csv', index=False)
print("✅ Saved: metadata_reversion_model1.csv")

# =============================================================================
# Model 2: APP baseline vs APP aggregated
# =============================================================================
print("\n### Model 2: APP Non-aggregated vs APP Aggregated ###")

# Only APP neurons
df_app = df_subset[df_subset['condition'] == 'APPV717I'].copy()

df_model2 = df_app.copy()
df_model2['class_label'] = 'APP_Non-aggregated'

# Class 0: APP at T0 (both stim) or T15 unstimulated
non_agg_app = [
    (df_app['timepoint'] == 'T0') & (df_app['stimulation'] == '10000ms'),
    (df_app['timepoint'] == 'T0') & (df_app['stimulation'] == '0ms'),
    (df_app['timepoint'] == 'T15') & (df_app['stimulation'] == '0ms'),
]

# Class 1: APP T15 10000ms (aggregated)
agg_app_mask = (df_app['timepoint'] == 'T15') & (df_app['stimulation'] == '10000ms')
df_model2.loc[agg_app_mask, 'class_label'] = 'APP_Aggregated'

# Only keep conditions we want
keep_app_mask = pd.concat([pd.Series(cond) for cond in non_agg_app]).groupby(level=0).any() | agg_app_mask
df_model2 = df_model2[keep_app_mask].copy()

print(f"\nOriginal class counts:")
print(df_model2['class_label'].value_counts())

# Balance
min_count = df_model2['class_label'].value_counts().min()
balanced_dfs = []
for label in ['APP_Non-aggregated', 'APP_Aggregated']:
    class_df = df_model2[df_model2['class_label'] == label]
    
    if len(class_df) > min_count:
        neurons = class_df['neuron_id'].unique()
        sampled_neurons = np.random.choice(neurons, size=min_count, replace=False)
        class_df = class_df[class_df['neuron_id'].isin(sampled_neurons)]
    
    balanced_dfs.append(class_df)

df_model2_balanced = pd.concat(balanced_dfs, ignore_index=True)

print(f"\nBalanced class counts:")
print(df_model2_balanced['class_label'].value_counts())
print(f"Neurons: {df_model2_balanced['neuron_id'].nunique()}")

df_model2_balanced.to_csv('experiments/2025-11-06_rescue_masked/dataset/metadata_reversion_model2.csv', index=False)
print("✅ Saved: metadata_reversion_model2.csv")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Model 1: Detects aggregation in APP OR WT")
print("Model 2: Detects aggregation specifically in APP")
print("="*70)
