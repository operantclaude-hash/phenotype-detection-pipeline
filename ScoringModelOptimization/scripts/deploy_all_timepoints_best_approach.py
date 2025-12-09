#!/usr/bin/env python3
"""
Deploy best approach (temperature_scaled_confidence_weighted) to ALL timepoints.

For each timepoint T1, T2, ..., T14/T15:
- Create pairs: (T0, Ti)
- Generate predictions using temperature scaling + confidence weighting
- Track temporal evolution of phenotype

This allows you to see how the disease phenotype evolves over time.
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'ScoringModelOptimization/scripts/phase0_fix_imbalance')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

from train_binary_6channel_REBALANCED import BinarySixChannelClassifier


class AllTimepointsDataset(Dataset):
    """Dataset for all timepoint pairs (T0 + Ti)"""

    def __init__(self, csv_path, target_timepoint,
                 percentiles_path='A2screen_combined/data/intensity_percentiles.json'):
        self.df = pd.read_csv(csv_path)
        self.target_timepoint = target_timepoint

        # Load intensity percentiles
        with open(percentiles_path) as f:
            self.percentiles = json.load(f)

        # Determine plate and final timepoint
        self.plate = 'plate1' if 'plate1' in str(csv_path) else 'plate2'

        print(f"\n✓ Processing {self.plate}, T0 + T{target_timepoint}")

        # Create T0 + Ti pairs
        self.pairs = []
        for neuron_id in self.df['neuron_id'].unique():
            neuron_data = self.df[self.df['neuron_id'] == neuron_id]

            t0 = neuron_data[neuron_data['timepoint'] == 0]
            ti = neuron_data[neuron_data['timepoint'] == target_timepoint]

            if len(t0) == 0 or len(ti) == 0:
                continue

            t0 = t0.iloc[0]
            ti = ti.iloc[0]

            # Determine root directory
            if self.plate == 'plate1':
                root_dir = 'A2screen_combined/data/plate1_aligned_extracted'
            else:
                root_dir = f'A2screen_combined/data/{self.plate}_extracted'

            self.pairs.append({
                't0_rfp1': f"{root_dir}/{t0['rfp1_path']}",
                't0_halo': f"{root_dir}/{t0['halo_path']}",
                't0_halo_masked': f"{root_dir}/{t0['halo_masked_path']}",
                'ti_rfp1': f"{root_dir}/{ti['rfp1_path']}",
                'ti_halo': f"{root_dir}/{ti['halo_path']}",
                'ti_halo_masked': f"{root_dir}/{ti['halo_masked_path']}",
                'neuron_id': neuron_id,
                'well': t0['well'],
                'column': t0['column'],
                'tile': t0['tile'],
                'genotype': t0['genotype'],
                'stimulation': t0['stimulation'],
                'timepoint': target_timepoint,
            })

        print(f"✓ Created {len(self.pairs)} neuron pairs")

        # Setup transforms
        from torchvision import transforms
        from PIL import Image

        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406] * 2,  # 6 channels
            std=[0.229, 0.224, 0.225] * 2
        )

    def _load_and_normalize(self, path, channel):
        """Load image and apply percentile normalization"""
        from PIL import Image

        # Load as grayscale
        img = np.array(Image.open(path), dtype=np.float32)

        # Get percentile bounds
        p1, p99 = self.percentiles[channel]['p1'], self.percentiles[channel]['p99']

        # Clip and normalize
        img_clipped = np.clip(img, p1, p99)
        if p99 > p1:
            img_normalized = (img_clipped - p1) / (p99 - p1)
        else:
            img_normalized = img_clipped

        # Resize
        img_pil = Image.fromarray((img_normalized * 255).astype(np.uint8))
        img_pil = self.resize(img_pil)

        # Convert to tensor (single channel)
        img_tensor = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).unsqueeze(0)

        return img_tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load all 6 channels
        t0_rfp1 = self._load_and_normalize(pair['t0_rfp1'], 'rfp1')
        t0_halo = self._load_and_normalize(pair['t0_halo'], 'halo')
        t0_halo_masked = self._load_and_normalize(pair['t0_halo_masked'], 'halo_masked')
        ti_rfp1 = self._load_and_normalize(pair['ti_rfp1'], 'rfp1')
        ti_halo = self._load_and_normalize(pair['ti_halo'], 'halo')
        ti_halo_masked = self._load_and_normalize(pair['ti_halo_masked'], 'halo_masked')

        # Concatenate: (6, H, W)
        img_6ch = torch.cat([
            t0_rfp1, t0_halo, t0_halo_masked,
            ti_rfp1, ti_halo, ti_halo_masked
        ], dim=0)

        # Apply ImageNet normalization
        img_6ch = self.normalize(img_6ch)

        return img_6ch


def apply_temperature_scaling(logits, temperature=1.545):
    """Apply temperature scaling"""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    return probs.numpy()


def confidence_weighted_aggregation(df_predictions):
    """Aggregate by (well, tile, timepoint) with confidence weighting"""
    results = []

    for (well, tile, timepoint), group in df_predictions.groupby(['well', 'tile', 'timepoint']):
        probs_class1 = group['prob_class1'].values
        weights = 2 * np.abs(probs_class1 - 0.5)

        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        weighted_prob_class1 = (probs_class1 * weights).sum()
        weighted_prob_class0 = 1 - weighted_prob_class1

        results.append({
            'well': well,
            'tile': tile,
            'timepoint': timepoint,
            'prob_class0': weighted_prob_class0,
            'prob_class1': weighted_prob_class1,
            'n_neurons': len(group),
            'column': group['column'].iloc[0],
            'genotype': group['genotype'].iloc[0],
            'stimulation': group['stimulation'].iloc[0],
        })

    return pd.DataFrame(results)


def deploy_timepoint(model, dataset, temperature=1.545, device='cuda'):
    """Deploy model for a specific timepoint"""
    model.eval()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing T0+T{dataset.target_timepoint}"):
            batch = batch.to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()

    # Apply temperature scaling
    probs = apply_temperature_scaling(torch.tensor(all_logits), temperature)

    # Create predictions dataframe
    predictions = []
    for i, pair in enumerate(dataset.pairs):
        predictions.append({
            'neuron_id': pair['neuron_id'],
            'well': pair['well'],
            'column': pair['column'],
            'tile': pair['tile'],
            'genotype': pair['genotype'],
            'stimulation': pair['stimulation'],
            'timepoint': pair['timepoint'],
            'prob_class0': probs[i, 0],
            'prob_class1': probs[i, 1],
            'predicted_class': int(np.argmax(probs[i])),
        })

    df_predictions = pd.DataFrame(predictions)

    # Aggregate by (well, tile) with confidence weighting
    well_tile_aggregated = confidence_weighted_aggregation(df_predictions)

    return df_predictions, well_tile_aggregated


def main():
    print("\n" + "="*80)
    print("DEPLOYING BEST APPROACH TO ALL TIMEPOINTS")
    print("="*80)
    print("\nApproach: Temperature Scaling (T=1.545) + Confidence Weighting")
    print("="*80)

    # Load model
    checkpoint_path = 'ScoringModelOptimization/models/binary_6channel_REBALANCED/checkpoints/best-epoch=19-val_acc=0.691.ckpt'
    model = BinarySixChannelClassifier.load_from_checkpoint(checkpoint_path)
    print(f"\n✓ Loaded model: {checkpoint_path}")

    OPTIMAL_TEMPERATURE = 1.545

    output_dir = Path('ScoringModelOptimization/results/temporal_predictions_best_approach')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each plate
    for plate_name in ['plate1', 'plate2']:
        print("\n" + "="*80)
        print(f"PROCESSING {plate_name.upper()}")
        print("="*80)

        if plate_name == 'plate1':
            csv_path = 'A2screen_combined/data/plate1_aligned_extracted/metadata_plate1_aligned.csv'
        else:
            csv_path = f'A2screen_combined/data/{plate_name}_extracted/metadata_{plate_name}.csv'

        # Determine final timepoint
        final_timepoint = 14 if plate_name == 'plate1' else 15

        # Collect all predictions across timepoints
        all_neuron_predictions = []
        all_well_tile_predictions = []

        # Deploy to each timepoint
        for timepoint in range(1, final_timepoint + 1):
            print(f"\n{'-'*80}")
            print(f"Timepoint {timepoint}/{final_timepoint}")
            print(f"{'-'*80}")

            # Create dataset for this timepoint
            dataset = AllTimepointsDataset(csv_path, timepoint)

            # Deploy
            neuron_preds, well_tile_preds = deploy_timepoint(
                model, dataset, temperature=OPTIMAL_TEMPERATURE
            )

            all_neuron_predictions.append(neuron_preds)
            all_well_tile_predictions.append(well_tile_preds)

            print(f"✓ T0+T{timepoint}: {len(neuron_preds)} neurons, {len(well_tile_preds)} (well,tile) combinations")

        # Combine all timepoints
        print(f"\n{'-'*80}")
        print("COMBINING ALL TIMEPOINTS")
        print(f"{'-'*80}")

        df_all_neurons = pd.concat(all_neuron_predictions, ignore_index=True)
        df_all_well_tiles = pd.concat(all_well_tile_predictions, ignore_index=True)

        # Save
        df_all_neurons.to_csv(
            output_dir / f'{plate_name}_neuron_predictions_all_timepoints.csv',
            index=False
        )
        df_all_well_tiles.to_csv(
            output_dir / f'{plate_name}_well_tile_predictions_all_timepoints.csv',
            index=False
        )

        print(f"\n✓ Saved predictions for {plate_name}:")
        print(f"  Neuron-level: {len(df_all_neurons)} predictions across {final_timepoint} timepoints")
        print(f"  (Well,Tile)-level: {len(df_all_well_tiles)} predictions")
        print(f"\n  Files:")
        print(f"  - {output_dir / f'{plate_name}_neuron_predictions_all_timepoints.csv'}")
        print(f"  - {output_dir / f'{plate_name}_well_tile_predictions_all_timepoints.csv'}")

    # Create README
    print("\n" + "="*80)
    print("CREATING README")
    print("="*80)

    readme_content = """# Temporal Predictions - Best Approach

## Approach

**Temperature Scaling (T=1.545) + Confidence-Weighted Aggregation**

This was determined to be the best approach from the comparison of:
1. Baseline + unweighted
2. Baseline + confidence-weighted
3. Temperature-scaled + unweighted
4. Temperature-scaled + confidence-weighted ✓ (BEST)

## Timepoints

For each plate, predictions are generated for ALL timepoint pairs:
- **Plate1**: T0+T1, T0+T2, ..., T0+T14 (14 timepoints)
- **Plate2**: T0+T2, T0+T2, ..., T0+T15 (15 timepoints)

This allows you to track the temporal evolution of the disease phenotype.

## Files Generated

### Plate 1
- `plate1_neuron_predictions_all_timepoints.csv` - All neuron predictions across all timepoints
- `plate1_well_tile_predictions_all_timepoints.csv` - Aggregated by (well, tile) across all timepoints

### Plate 2
- `plate2_neuron_predictions_all_timepoints.csv` - All neuron predictions across all timepoints
- `plate2_well_tile_predictions_all_timepoints.csv` - Aggregated by (well, tile) across all timepoints

## Columns in Neuron-Level Files

- `neuron_id`: Unique neuron identifier
- `well`: Well identifier (e.g., O19)
- `column`: Column number (1-24)
- `tile`: Tile number within well (1-4)
- `genotype`: Genotype (APP or WT)
- `stimulation`: Stimulation condition (0ms or 10000ms)
- `timepoint`: Target timepoint (1-14 or 1-15)
- `prob_class0`: Probability of Class 0 (APP_0ms phenotype)
- `prob_class1`: Probability of Class 1 (APP_10000ms phenotype)
- `predicted_class`: Predicted class (0 or 1)

## Columns in (Well, Tile)-Level Files

- `well`: Well identifier
- `tile`: Tile number within well (1-4)
- `timepoint`: Target timepoint
- `prob_class0`: Confidence-weighted mean probability
- `prob_class1`: Confidence-weighted mean probability
- `n_neurons`: Number of neurons in (well, tile, timepoint)
- `column`: Column number
- `genotype`: Genotype
- `stimulation`: Stimulation condition

## Temporal Analysis

You can now analyze:
1. **How phenotype evolves over time** for each (well, tile)
2. **When stimulation effects appear** by comparing 0ms vs 10000ms tiles
3. **Drug effects on temporal trajectory** by comparing treated vs control wells
4. **Rescue kinetics** - how quickly drugs reverse the phenotype

## Example Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv('plate2_well_tile_predictions_all_timepoints.csv')

# Plot temporal evolution for a specific well
well_data = df[df['well'] == 'O19']

for tile in well_data['tile'].unique():
    tile_data = well_data[well_data['tile'] == tile]
    plt.plot(tile_data['timepoint'], tile_data['prob_class1'],
             label=f"Tile {tile} ({tile_data['stimulation'].iloc[0]})")

plt.xlabel('Timepoint')
plt.ylabel('Prob(APP_10000ms phenotype)')
plt.title('Temporal Evolution - Well O19')
plt.legend()
plt.show()
```

## Model Information

- **Model**: binary_6channel_REBALANCED
- **Validation Accuracy**: 69.1%
- **Temperature**: 1.545 (optimized for calibration)
- **Aggregation**: Confidence-weighted (high-confidence predictions emphasized)
"""

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    print(f"\n✓ Saved README: {output_dir / 'README.md'}")

    print("\n" + "="*80)
    print("✓ DEPLOYMENT COMPLETE")
    print("="*80)
    print(f"\nAll predictions saved to: {output_dir}")
    print("\nYou can now analyze temporal evolution of the phenotype!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
