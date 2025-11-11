#!/usr/bin/env python3
"""
Compare results across multiple experiments
Usage: python compare_experiments.py exp1 exp2 exp3 ...
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_experiment_results(exp_dir):
    """Load results from an experiment directory"""
    exp_dir = Path(exp_dir)
    
    results = {
        'name': exp_dir.name,
        'path': str(exp_dir)
    }
    
    # Load training results if available
    for channel in ['RFP1', 'Halo']:
        metrics_file = exp_dir / 'evaluation' / channel / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                results[f'{channel.lower()}_accuracy'] = metrics.get('test_accuracy', 0)
                results[f'{channel.lower()}_per_class'] = metrics.get('per_class_accuracy', {})
    
    # Load analysis summary if available
    summary_file = exp_dir / 'analysis' / 'summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            results['analysis'] = json.load(f)
    
    return results

def compare_experiments(experiment_dirs, output_dir='comparison_reports'):
    """Compare multiple experiments and generate report"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("EXPERIMENT COMPARISON")
    print("="*70)
    
    # Load all experiments
    experiments = []
    for exp_dir in experiment_dirs:
        if Path(exp_dir).exists():
            results = load_experiment_results(exp_dir)
            experiments.append(results)
            print(f"\n✅ Loaded: {results['name']}")
        else:
            print(f"\n❌ Not found: {exp_dir}")
    
    if len(experiments) < 2:
        print("\n⚠️  Need at least 2 experiments to compare")
        return
    
    # Create comparison table
    print("\n" + "="*70)
    print("ACCURACY COMPARISON")
    print("="*70)
    
    df_data = []
    for exp in experiments:
        row = {
            'Experiment': exp['name'],
            'RFP1': exp.get('rfp1_accuracy', 'N/A'),
            'Halo': exp.get('halo_accuracy', 'N/A')
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    print("\n", df.to_string(index=False))
    
    # Save to CSV
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(output_dir / f'comparison_{timestamp}.csv', index=False)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(experiments))
    width = 0.35
    
    rfp1_accs = [exp.get('rfp1_accuracy', 0) for exp in experiments]
    halo_accs = [exp.get('halo_accuracy', 0) for exp in experiments]
    
    ax.bar(x - width/2, rfp1_accs, width, label='RFP1', alpha=0.8)
    ax.bar(x + width/2, halo_accs, width, label='Halo', alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels([exp['name'] for exp in experiments], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/comparison_{timestamp}.png")
    
    # Generate markdown report
    report = f"""# Experiment Comparison Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiments Compared

{chr(10).join(f'- {exp["name"]}' for exp in experiments)}

## Overall Accuracy

| Experiment | RFP1 | Halo |
|------------|------|------|
"""
    
    for exp in experiments:
        rfp1 = exp.get('rfp1_accuracy', 'N/A')
        halo = exp.get('halo_accuracy', 'N/A')
        rfp1_str = f'{rfp1:.3f}' if isinstance(rfp1, float) else rfp1
        halo_str = f'{halo:.3f}' if isinstance(halo, float) else halo
        report += f"| {exp['name']} | {rfp1_str} | {halo_str} |\n"
    
    report += f"""
## Visualization

![Accuracy Comparison](comparison_{timestamp}.png)

## Analysis

### Best Performing
- **RFP1**: {experiments[np.argmax(rfp1_accs)]['name']} ({max(rfp1_accs):.3f})
- **Halo**: {experiments[np.argmax(halo_accs)]['name']} ({max(halo_accs):.3f})

### Key Observations

[Add your observations here]

## Conclusions

[Add your conclusions here]
"""
    
    with open(output_dir / f'comparison_{timestamp}.md', 'w') as f:
        f.write(report)
    
    print(f"✅ Saved: {output_dir}/comparison_{timestamp}.md")
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Compare results across multiple experiments'
    )
    parser.add_argument(
        'experiments',
        nargs='+',
        help='Experiment directories to compare'
    )
    parser.add_argument(
        '--output_dir',
        default='comparison_reports',
        help='Output directory for comparison reports'
    )
    
    args = parser.parse_args()
    
    compare_experiments(args.experiments, args.output_dir)

if __name__ == '__main__':
    main()
