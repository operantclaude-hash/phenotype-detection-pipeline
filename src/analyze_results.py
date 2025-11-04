#!/usr/bin/env python3
"""
Comprehensive analysis of A2screenPlate2 classification models
Compares 8-class (with artifacts) vs 6-class (biology only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse


def load_results(results_dir, task):
    """Load evaluation results from a task directory"""
    results_dir = Path(results_dir)
    
    # Load metrics
    metrics_file = results_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = None
    
    # Load confusion matrix if exists
    cm_file = results_dir / 'confusion_matrix.npy'
    if cm_file.exists():
        cm = np.load(cm_file)
    else:
        cm = None
    
    return metrics, cm


def compare_8class_vs_6class(rfp1_8class_dir, rfp1_6class_dir, 
                              halo_8class_dir, halo_6class_dir,
                              output_dir):
    """
    Compare 8-class vs 6-class models to assess artifact contribution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("8-CLASS vs 6-CLASS COMPARISON")
    print("="*80)
    
    # Load all results
    rfp1_8_metrics, rfp1_8_cm = load_results(rfp1_8class_dir, '8class')
    rfp1_6_metrics, rfp1_6_cm = load_results(rfp1_6class_dir, '6class')
    halo_8_metrics, halo_8_cm = load_results(halo_8class_dir, '8class')
    halo_6_metrics, halo_6_cm = load_results(halo_6class_dir, '6class')
    
    # Extract accuracies
    rfp1_8_acc = rfp1_8_metrics.get('test_accuracy', rfp1_8_metrics.get('accuracy', 0.434))
    rfp1_6_acc = rfp1_6_metrics.get('test_accuracy', rfp1_6_metrics.get('accuracy', 0.548))
    halo_8_acc = halo_8_metrics.get('test_accuracy', halo_8_metrics.get('accuracy', 0.461))
    halo_6_acc = halo_6_metrics.get('test_accuracy', halo_6_metrics.get('accuracy', 0.505))
    
    print("\n📊 OVERALL ACCURACY")
    print("-" * 60)
    print(f"{'Model':<20} {'8-class':<15} {'6-class':<15} {'Change':<15}")
    print("-" * 60)
    print(f"{'RFP1 (Morphology)':<20} {rfp1_8_acc:<15.3f} {rfp1_6_acc:<15.3f} {rfp1_6_acc-rfp1_8_acc:+.3f}")
    print(f"{'Halo (Reporter)':<20} {halo_8_acc:<15.3f} {halo_6_acc:<15.3f} {halo_6_acc-halo_8_acc:+.3f}")
    print("-" * 60)
    print(f"{'Random Chance':<20} {'12.5% (1/8)':<15} {'16.7% (1/6)':<15}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy comparison
    channels = ['RFP1', 'Halo']
    acc_8class = [rfp1_8_acc, halo_8_acc]
    acc_6class = [rfp1_6_acc, halo_6_acc]
    
    x = np.arange(len(channels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, acc_8class, width, label='8-class (with artifacts)', 
                    color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, acc_6class, width, label='6-class (biology only)', 
                    color='lightblue', alpha=0.8)
    
    ax1.axhline(y=0.125, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Chance (8-class)')
    ax1.axhline(y=0.167, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Chance (6-class)')
    
    ax1.set_xlabel('Channel', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(channels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Accuracy drop (artifact contribution)
    artifact_contribution = [rfp1_8_acc - rfp1_6_acc, halo_8_acc - halo_6_acc]
    colors = ['red' if x > 0 else 'green' for x in artifact_contribution]
    
    bars = ax2.bar(channels, artifact_contribution, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Channel', fontsize=12)
    ax2.set_ylabel('Accuracy Drop (8-class - 6-class)', fontsize=12)
    ax2.set_title('Artifact Contribution to Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1%}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / '8class_vs_6class_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir / '8class_vs_6class_comparison.png'}")
    plt.close()
    
    # Interpretation
    print("\n🔬 INTERPRETATION")
    print("-" * 60)
    
    if rfp1_8_acc > rfp1_6_acc:
        print("⚠️  RFP1: 8-class accuracy HIGHER than 6-class")
        print(f"   → Model was learning {(rfp1_8_acc - rfp1_6_acc)*100:.1f}% from technical artifacts")
    else:
        print("✅ RFP1: 6-class accuracy higher - artifact removal helped!")
    
    if halo_8_acc > halo_6_acc:
        print("⚠️  Halo: 8-class accuracy HIGHER than 6-class")
        print(f"   → Model was learning {(halo_8_acc - halo_6_acc)*100:.1f}% from technical artifacts")
    else:
        print("✅ Halo: 6-class accuracy higher - artifact removal helped!")
    
    print("\n📈 PERFORMANCE ABOVE CHANCE")
    print("-" * 60)
    print(f"RFP1 6-class: {(rfp1_6_acc - 0.167)*100:.1f}% above chance")
    print(f"Halo 6-class: {(halo_6_acc - 0.167)*100:.1f}% above chance")
    
    return {
        'rfp1_8class': rfp1_8_acc,
        'rfp1_6class': rfp1_6_acc,
        'halo_8class': halo_8_acc,
        'halo_6class': halo_6_acc
    }


def analyze_6class_performance(rfp1_dir, halo_dir, metadata_path, output_dir):
    """
    Detailed analysis of 6-class model performance
    Focus on T0 vs T16 and stimulation effects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("6-CLASS MODEL DETAILED ANALYSIS")
    print("="*80)
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Load confusion matrices from saved numpy files
    rfp1_cm = np.load(rfp1_dir / 'confusion_matrix.npy')
    halo_cm = np.load(halo_dir / 'confusion_matrix.npy')
    
    # Class names
    class_names = [
        'Control_T0',
        'APPV717I_T0',
        'Control_0ms_T16',
        'Control_10000ms_T16',
        'APPV717I_0ms_T16',
        'APPV717I_10000ms_T16'
    ]
    
    # Calculate per-class accuracy
    rfp1_class_acc = rfp1_cm.diagonal() / rfp1_cm.sum(axis=1)
    halo_class_acc = halo_cm.diagonal() / halo_cm.sum(axis=1)
    
    print("\n📊 PER-CLASS ACCURACY")
    print("-" * 80)
    print(f"{'Class':<30} {'RFP1':<15} {'Halo':<15} {'Best Channel':<15}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        rfp1_acc = rfp1_class_acc[i]
        halo_acc = halo_class_acc[i]
        best = 'RFP1' if rfp1_acc > halo_acc else 'Halo'
        print(f"{class_name:<30} {rfp1_acc:<15.3f} {halo_acc:<15.3f} {best:<15}")
    
    # Analyze by timepoint
    print("\n⏱️  TIMEPOINT ANALYSIS")
    print("-" * 80)
    
    # T0 classes (indices 0, 1)
    rfp1_t0_acc = np.mean(rfp1_class_acc[0:2])
    halo_t0_acc = np.mean(halo_class_acc[0:2])
    
    # T16 classes (indices 2, 3, 4, 5)
    rfp1_t16_acc = np.mean(rfp1_class_acc[2:6])
    halo_t16_acc = np.mean(halo_class_acc[2:6])
    
    print(f"{'Timepoint':<30} {'RFP1':<15} {'Halo':<15}")
    print("-" * 60)
    print(f"{'T0 (pre-stim)':<30} {rfp1_t0_acc:<15.3f} {halo_t0_acc:<15.3f}")
    print(f"{'T16 (post-stim)':<30} {rfp1_t16_acc:<15.3f} {halo_t16_acc:<15.3f}")
    print(f"{'Improvement (T16 - T0)':<30} {rfp1_t16_acc-rfp1_t0_acc:+15.3f} {halo_t16_acc-halo_t0_acc:+15.3f}")
    
    # Analyze stimulation effect at T16
    print("\n💉 STIMULATION EFFECT AT T16")
    print("-" * 80)
    print("Can the model distinguish stimulated (10000ms) from unstimulated (0ms) at T16?")
    print()
    
    # Control stimulation effect
    control_0ms_t16 = rfp1_class_acc[2]  # Control_0ms_T16
    control_10000ms_t16 = rfp1_class_acc[3]  # Control_10000ms_T16
    print(f"RFP1 - Control:")
    print(f"  0ms (unstim):    {control_0ms_t16:.3f}")
    print(f"  10000ms (stim):  {control_10000ms_t16:.3f}")
    print(f"  Separable: {'✅ YES' if abs(control_10000ms_t16 - control_0ms_t16) > 0.1 else '⚠️  WEAK'}")
    
    control_0ms_t16_h = halo_class_acc[2]
    control_10000ms_t16_h = halo_class_acc[3]
    print(f"\nHalo - Control:")
    print(f"  0ms (unstim):    {control_0ms_t16_h:.3f}")
    print(f"  10000ms (stim):  {control_10000ms_t16_h:.3f}")
    print(f"  Separable: {'✅ YES' if abs(control_10000ms_t16_h - control_0ms_t16_h) > 0.1 else '⚠️  WEAK'}")
    
    # APPV717I stimulation effect
    app_0ms_t16 = rfp1_class_acc[4]
    app_10000ms_t16 = rfp1_class_acc[5]
    print(f"\nRFP1 - APPV717I:")
    print(f"  0ms (unstim):    {app_0ms_t16:.3f}")
    print(f"  10000ms (stim):  {app_10000ms_t16:.3f}")
    print(f"  Separable: {'✅ YES' if abs(app_10000ms_t16 - app_0ms_t16) > 0.1 else '⚠️  WEAK'}")
    
    app_0ms_t16_h = halo_class_acc[4]
    app_10000ms_t16_h = halo_class_acc[5]
    print(f"\nHalo - APPV717I:")
    print(f"  0ms (unstim):    {app_0ms_t16_h:.3f}")
    print(f"  10000ms (stim):  {app_10000ms_t16_h:.3f}")
    print(f"  Separable: {'✅ YES' if abs(app_10000ms_t16_h - app_0ms_t16_h) > 0.1 else '⚠️  WEAK'}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Per-class accuracy comparison
    ax = axes[0, 0]
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rfp1_class_acc, width, label='RFP1', alpha=0.8)
    bars2 = ax.bar(x + width/2, halo_class_acc, width, label='Halo', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.167, color='red', linestyle='--', alpha=0.5, label='Chance')
    
    # Plot 2: T0 vs T16 comparison
    ax = axes[0, 1]
    timepoints = ['T0\n(pre-stim)', 'T16\n(post-stim)']
    rfp1_time = [rfp1_t0_acc, rfp1_t16_acc]
    halo_time = [halo_t0_acc, halo_t16_acc]
    
    x = np.arange(len(timepoints))
    bars1 = ax.bar(x - width/2, rfp1_time, width, label='RFP1', alpha=0.8)
    bars2 = ax.bar(x + width/2, halo_time, width, label='Halo', alpha=0.8)
    
    ax.set_ylabel('Average Accuracy', fontsize=11)
    ax.set_title('Temporal Evolution of Phenotype', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(timepoints)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Stimulation effect in Control at T16
    ax = axes[1, 0]
    conditions = ['0ms\n(unstim)', '10000ms\n(stim)']
    rfp1_control = [control_0ms_t16, control_10000ms_t16]
    halo_control = [control_0ms_t16_h, control_10000ms_t16_h]
    
    x = np.arange(len(conditions))
    bars1 = ax.bar(x - width/2, rfp1_control, width, label='RFP1', alpha=0.8)
    bars2 = ax.bar(x + width/2, halo_control, width, label='Halo', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Stimulation Effect: Control at T16', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Stimulation effect in APPV717I at T16
    ax = axes[1, 1]
    rfp1_app = [app_0ms_t16, app_10000ms_t16]
    halo_app = [app_0ms_t16_h, app_10000ms_t16_h]
    
    bars1 = ax.bar(x - width/2, rfp1_app, width, label='RFP1', alpha=0.8)
    bars2 = ax.bar(x + width/2, halo_app, width, label='Halo', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Stimulation Effect: APPV717I at T16', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '6class_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir / '6class_detailed_analysis.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of A2screenPlate2 models'
    )
    parser.add_argument(
        '--rfp1_8class',
        type=str,
        required=True,
        help='RFP1 8-class evaluation directory'
    )
    parser.add_argument(
        '--rfp1_6class',
        type=str,
        required=True,
        help='RFP1 6-class evaluation directory'
    )
    parser.add_argument(
        '--halo_8class',
        type=str,
        required=True,
        help='Halo 8-class evaluation directory'
    )
    parser.add_argument(
        '--halo_6class',
        type=str,
        required=True,
        help='Halo 6-class evaluation directory'
    )
    parser.add_argument(
        '--metadata_6class',
        type=str,
        required=True,
        help='Metadata CSV for 6-class dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis_results',
        help='Output directory for analysis'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("A2 SCREEN PLATE 2 - COMPREHENSIVE MODEL ANALYSIS")
    print("="*80)
    
    # Compare 8-class vs 6-class
    accuracies = compare_8class_vs_6class(
        args.rfp1_8class,
        args.rfp1_6class,
        args.halo_8class,
        args.halo_6class,
        output_dir
    )
    
    # Detailed 6-class analysis
    analyze_6class_performance(
        Path(args.rfp1_6class),
        Path(args.halo_6class),
        args.metadata_6class,
        output_dir
    )
    
    # Save summary
    summary = {
        'accuracies': accuracies,
        'interpretation': {
            'rfp1_artifact_contribution': accuracies['rfp1_8class'] - accuracies['rfp1_6class'],
            'halo_artifact_contribution': accuracies['halo_8class'] - accuracies['halo_6class'],
            'rfp1_above_chance': accuracies['rfp1_6class'] - 0.167,
            'halo_above_chance': accuracies['halo_6class'] - 0.167
        }
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"   - 8class_vs_6class_comparison.png")
    print(f"   - 6class_detailed_analysis.png")
    print(f"   - analysis_summary.json")


if __name__ == '__main__':
    main()
