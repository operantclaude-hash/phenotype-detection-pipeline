# Simple comprehensive analysis
python << 'EOF'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Create output directory
Path('analysis_6class').mkdir(exist_ok=True)

# Load confusion matrices
rfp1_cm = np.load('evaluation_6class/RFP1/confusion_matrix.npy')
halo_cm = np.load('evaluation_6class/Halo/confusion_matrix.npy')

class_names = [
    'Control_T0', 'APPV717I_T0',
    'Control_0ms_T16', 'Control_10000ms_T16',
    'APPV717I_0ms_T16', 'APPV717I_10000ms_T16'
]

# Calculate accuracies
rfp1_acc = rfp1_cm.diagonal() / rfp1_cm.sum(axis=1)
halo_acc = halo_cm.diagonal() / halo_cm.sum(axis=1)
rfp1_overall = rfp1_acc.mean()
halo_overall = halo_acc.mean()

print("="*70)
print("6-CLASS MODEL RESULTS")
print("="*70)
print(f"\nOverall Accuracy: RFP1={rfp1_overall:.3f}, Halo={halo_overall:.3f}")
print(f"\n{'Class':<30} {'RFP1':>10} {'Halo':>10} {'Best':>10}")
print("-"*70)
for i, name in enumerate(class_names):
    best = 'RFP1' if rfp1_acc[i] > halo_acc[i] else 'Halo'
    print(f"{name:<30} {rfp1_acc[i]:>10.3f} {halo_acc[i]:>10.3f} {best:>10}")

# Temporal analysis
rfp1_t0 = rfp1_acc[:2].mean()
rfp1_t16 = rfp1_acc[2:].mean()
halo_t0 = halo_acc[:2].mean()
halo_t16 = halo_acc[2:].mean()

print("\n" + "="*70)
print("TEMPORAL ANALYSIS")
print("="*70)
print(f"T0 (pre-stim):   RFP1={rfp1_t0:.3f}, Halo={halo_t0:.3f}")
print(f"T16 (post-stim): RFP1={rfp1_t16:.3f}, Halo={halo_t16:.3f}")
print(f"Improvement:     RFP1={rfp1_t16-rfp1_t0:+.3f}, Halo={halo_t16-halo_t0:+.3f}")

# Stimulation effect at T16
print("\n" + "="*70)
print("STIMULATION EFFECT (T16 only)")
print("="*70)
print("Control:  0ms={:.3f}, 10000ms={:.3f}, gap={:.3f}".format(
    rfp1_acc[2], rfp1_acc[3], rfp1_acc[3]-rfp1_acc[2]))
print("APPV717I: 0ms={:.3f}, 10000ms={:.3f}, gap={:.3f}".format(
    rfp1_acc[4], rfp1_acc[5], rfp1_acc[5]-rfp1_acc[4]))

# Save summary
summary = {
    'rfp1_overall': float(rfp1_overall),
    'halo_overall': float(halo_overall),
    'rfp1_per_class': {name: float(acc) for name, acc in zip(class_names, rfp1_acc)},
    'halo_per_class': {name: float(acc) for name, acc in zip(class_names, halo_acc)},
    'temporal': {
        'rfp1_t0': float(rfp1_t0),
        'rfp1_t16': float(rfp1_t16),
        'halo_t0': float(halo_t0),
        'halo_t16': float(halo_t16)
    }
}

with open('analysis_6class/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ Analysis complete! Summary saved to analysis_6class/summary.json")
EOF
