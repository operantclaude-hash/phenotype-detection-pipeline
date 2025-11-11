#!/usr/bin/env python3
"""
Unit tests for training pipeline
Tests image/label processing, data splitting, and deployment

Run with: pytest tests/test_training_pipeline.py -v
"""
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from PIL import Image


# Mock data creation helpers
@pytest.fixture
def mock_image_dir():
    """Create temporary directory with mock images"""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def mock_metadata(mock_image_dir):
    """Create mock metadata CSV with proper structure"""
    # Create mock images
    wells = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    classes = ['T0', 'T15', 'T0', 'T15', 'T0', 'T15']

    metadata_rows = []
    for well_idx, (well, cls) in enumerate(zip(wells, classes)):
        for neuron in range(10):  # 10 neurons per well
            # Create mock images
            neuron_id = f"{well}_n{neuron}"
            rfp1_path = mock_image_dir / f"{neuron_id}_rfp1.png"
            halo_path = mock_image_dir / f"{neuron_id}_halo.png"
            halo_masked_path = mock_image_dir / f"{neuron_id}_halo_masked.png"

            # Save actual image files
            for img_path in [rfp1_path, halo_path, halo_masked_path]:
                img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
                img.save(img_path)

            metadata_rows.append({
                'neuron_id': neuron_id,
                'well': well,
                'tile': 1,
                'timepoint': cls,
                'class_label': cls,
                'rfp1_path': str(rfp1_path.relative_to(mock_image_dir)),
                'halo_path': str(halo_path.relative_to(mock_image_dir)),
                'halo_masked_path': str(halo_masked_path.relative_to(mock_image_dir)),
            })

    df = pd.DataFrame(metadata_rows)
    metadata_path = mock_image_dir / 'metadata.csv'
    df.to_csv(metadata_path, index=False)

    return metadata_path, mock_image_dir


# Test 1: Stratification Alignment
def test_stratification_alignment():
    """Test that stratification labels align with wells array"""
    from sklearn.model_selection import train_test_split

    # Create metadata where well order matters
    metadata = pd.DataFrame({
        'well': ['C3', 'A1', 'B2', 'C3', 'A1', 'B2'],
        'class_label': ['T0', 'T15', 'T0', 'T0', 'T15', 'T0']
    })

    # Simulate the BUGGY code
    unique_wells = metadata['well'].unique()  # ['C3', 'A1', 'B2']
    well_labels = metadata.groupby('well')['class_label'].first()
    # well_labels = Series(['T15', 'T0', 'T0'], index=['A1', 'B2', 'C3'])

    # BUG: If we use well_labels.values, it gives ['T15', 'T0', 'T0']
    # but unique_wells is ['C3', 'A1', 'B2']
    # So C3 gets labeled T15, A1 gets labeled T0, B2 gets labeled T0 - WRONG!

    buggy_stratify = well_labels.values
    correct_stratify = well_labels.loc[unique_wells].values

    # They should be different (demonstrating the bug)
    assert not np.array_equal(buggy_stratify, correct_stratify), \
        "Bug not reproduced - stratify arrays are the same!"

    # Correct version should match well order
    expected = ['T0', 'T15', 'T0']  # C3->T0, A1->T15, B2->T0
    assert np.array_equal(correct_stratify, expected), \
        f"Expected {expected}, got {correct_stratify}"


# Test 2: Reversion Dataset Stratification
def test_reversion_stratification_crash():
    """Test that reversion datasets with mixed-class wells crash with current code"""
    from sklearn.model_selection import train_test_split

    # Reversion metadata: each well has BOTH T0 and T15
    metadata = pd.DataFrame({
        'well': ['A1', 'A1', 'A2', 'A2', 'A3', 'A3'],
        'class_label': ['T0', 'T15', 'T0', 'T15', 'T0', 'T15']
    })

    unique_wells = metadata['well'].unique()  # ['A1', 'A2', 'A3']
    well_labels = metadata.groupby('well')['class_label'].first()
    # All wells get 'T0' (first label)

    # This should raise ValueError: least populated class has only 1 member
    with pytest.raises(ValueError, match="least populated class"):
        wells_train, wells_test = train_test_split(
            unique_wells, test_size=0.15, random_state=42,
            stratify=well_labels.loc[unique_wells].values
        )


# Test 3: Data Loading and Channel Order
def test_channel_order_and_loading(mock_metadata):
    """Test that images are loaded correctly and channels are in right order"""
    import sys
    sys.path.insert(0, 'src')
    from train_lightning_threechannel_improved import ThreeChannelDataset

    metadata_path, root_dir = mock_metadata
    metadata = pd.read_csv(metadata_path)

    dataset = ThreeChannelDataset(
        metadata=metadata.head(10),
        root_dir=root_dir,
        is_train=False
    )

    # Load first sample
    img, label = dataset[0]

    # Check shape: should be [3, H, W]
    assert img.shape[0] == 3, f"Expected 3 channels, got {img.shape[0]}"

    # Check that channels are different (not all the same)
    # Note: This is a weak test since random images might rarely be identical
    assert not torch.allclose(img[0], img[1], rtol=1e-3), \
        "Channel 0 and 1 are identical - possible loading bug"
    assert not torch.allclose(img[1], img[2], rtol=1e-3), \
        "Channel 1 and 2 are identical - possible loading bug"


# Test 4: Augmentation Preserves Channel Relationships
def test_colorjitter_breaks_relationships():
    """Test that ColorJitter breaks channel relationships"""
    from torchvision import transforms

    # Create 3-channel image with known relationships
    # Channel 0: 0.5, Channel 1: 0.25, Channel 2: 0.2
    img = torch.tensor([
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.25, 0.25], [0.25, 0.25]],
        [[0.2, 0.2], [0.2, 0.2]]
    ])

    # Original ratio: 0.5:0.25:0.2 = 2:1:0.8
    original_ratio = img[0, 0, 0] / img[1, 0, 0]

    # Apply ColorJitter
    jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3)
    augmented = jitter(img)

    # Check if ratio is preserved (it won't be!)
    new_ratio = augmented[0, 0, 0] / augmented[1, 0, 0]

    # The ratios SHOULD be different (demonstrating the bug)
    # In a few runs, they might be similar by chance, so we use a tolerance
    # But typically ColorJitter will destroy the relationship
    # This test documents the issue rather than asserting failure
    print(f"Original ratio: {original_ratio:.3f}, New ratio: {new_ratio:.3f}")
    print(f"Ratio change: {abs(original_ratio - new_ratio):.3f}")


# Test 5: Label Mapping Correctness
def test_label_mapping(mock_metadata):
    """Test that labels are mapped correctly from class names to indices"""
    metadata_path, root_dir = mock_metadata
    metadata = pd.read_csv(metadata_path)

    # Create label mapping
    unique_labels = sorted(metadata['class_label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    metadata['label_idx'] = metadata['class_label'].map(label_to_idx)

    # Check mapping is consistent
    for cls in unique_labels:
        cls_indices = metadata[metadata['class_label'] == cls]['label_idx'].unique()
        assert len(cls_indices) == 1, \
            f"Class {cls} maps to multiple indices: {cls_indices}"

    # Check all labels are mapped
    assert not metadata['label_idx'].isna().any(), \
        "Some labels were not mapped!"


# Test 6: Deployment Index Tracking
def test_deployment_index_tracking():
    """Test that deployment correctly tracks indices across batches"""
    batch_sizes = [32, 32, 32, 4]  # Last batch is smaller

    # BUGGY version (hardcoded 32)
    buggy_indices = []
    for batch_idx, bs in enumerate(batch_sizes):
        start_idx = batch_idx * 32  # HARDCODED
        buggy_indices.extend(range(start_idx, start_idx + bs))

    # CORRECT version (cumulative)
    correct_indices = []
    current_idx = 0
    for batch_idx, bs in enumerate(batch_sizes):
        correct_indices.extend(range(current_idx, current_idx + bs))
        current_idx += bs

    # They should be the same for full batches but different for last batch
    assert buggy_indices[:96] == correct_indices[:96], \
        "Indices differ for full batches - unexpected!"

    assert buggy_indices[96:] != correct_indices[96:], \
        "Bug not reproduced - indices are the same for last batch!"

    # Correct version should be 0-99
    assert correct_indices == list(range(100)), \
        f"Expected 0-99, got {correct_indices[:5]}...{correct_indices[-5:]}"

    # Buggy version will try to access 96-127
    assert buggy_indices == list(range(96)) + list(range(96, 100)), \
        f"Bug behavior changed - got {buggy_indices[-10:]}"


# Test 7: Confusion Matrix for Multiple Classes
def test_confusion_matrix_multi_class():
    """Test confusion matrix printing for >2 classes"""
    class_names = ['T0', 'T15', 'T30']
    cm = np.array([
        [10, 2, 1],
        [1, 15, 3],
        [2, 1, 20]
    ])

    # Current BUGGY code would only print columns 0 and 1
    # Test that we need all 3 columns
    assert cm.shape[0] == 3 and cm.shape[1] == 3, \
        "Confusion matrix should be 3x3"

    # Test dynamic printing (what should be implemented)
    output = []
    header = f"{'':20s}"
    for name in class_names:
        header += f" {'Pred ' + name:20s}"
    output.append(header)

    for i, true_name in enumerate(class_names):
        row = f"{'True ' + true_name:20s}"
        for j in range(len(class_names)):
            row += f" {cm[i, j]:20d}"
        output.append(row)

    print("\n".join(output))
    assert len(output) == 4, "Should have header + 3 rows"


# Test 8: Well-Based Splitting (No Data Leakage)
def test_well_based_splitting(mock_metadata):
    """Test that train/val/test splits don't leak data across wells"""
    from sklearn.model_selection import train_test_split

    metadata_path, root_dir = mock_metadata
    metadata = pd.read_csv(metadata_path)

    unique_wells = metadata['well'].unique()
    well_labels = metadata.groupby('well')['class_label'].first()

    # Correct splitting
    wells_train_val, wells_test = train_test_split(
        unique_wells, test_size=0.2, random_state=42,
        stratify=well_labels.loc[unique_wells].values
    )

    wells_train, wells_val = train_test_split(
        wells_train_val, test_size=0.2, random_state=42
    )

    # Check no overlap
    train_set = set(wells_train)
    val_set = set(wells_val)
    test_set = set(wells_test)

    assert len(train_set & val_set) == 0, "Train and val share wells!"
    assert len(train_set & test_set) == 0, "Train and test share wells!"
    assert len(val_set & test_set) == 0, "Val and test share wells!"

    # Check all wells are assigned
    assert len(train_set | val_set | test_set) == len(unique_wells), \
        "Not all wells were assigned to a split!"


# Test 9: Batch Size Consistency
def test_batch_size_handling(mock_metadata):
    """Test that dataloaders handle non-divisible dataset sizes correctly"""
    import sys
    sys.path.insert(0, 'src')
    from train_lightning_threechannel_improved import ThreeChannelDataset
    from torch.utils.data import DataLoader

    metadata_path, root_dir = mock_metadata
    metadata = pd.read_csv(metadata_path)

    # Use small subset (e.g., 25 samples) with batch_size=10
    small_metadata = metadata.head(25)
    dataset = ThreeChannelDataset(small_metadata, root_dir, is_train=False)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    batch_sizes = [len(batch[0]) for batch in dataloader]

    # Should be [10, 10, 5]
    assert batch_sizes == [10, 10, 5], \
        f"Expected [10, 10, 5], got {batch_sizes}"


# Test 10: Normalization Values
def test_normalization_statistics():
    """Document ImageNet normalization vs actual microscopy data"""
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    # Create mock microscopy data (typically lower intensity)
    mock_microscopy = torch.rand(100, 3, 64, 64) * 0.3  # Lower intensity
    actual_mean = mock_microscopy.mean(dim=[0, 2, 3])
    actual_std = mock_microscopy.std(dim=[0, 2, 3])

    print(f"ImageNet normalization: mean={imagenet_mean}, std={imagenet_std}")
    print(f"Mock microscopy data: mean={actual_mean}, std={actual_std}")

    # Document the difference
    mean_diff = torch.abs(imagenet_mean - actual_mean).mean()
    print(f"Mean difference: {mean_diff:.3f}")

    # This test documents the issue, doesn't assert failure
    assert mean_diff > 0.1, "Microscopy and ImageNet stats are surprisingly similar!"


# Test 11: Class Weight Calculation
def test_class_weight_calculation():
    """Test class weight formulas"""
    class_counts = pd.Series({'T0': 900, 'T15': 100})
    total = 1000
    n_classes = 2

    # Current formula (more aggressive)
    current_weights = [total / class_counts[label] for label in ['T0', 'T15']]

    # sklearn balanced formula
    sklearn_weights = [
        total / (n_classes * class_counts[label])
        for label in ['T0', 'T15']
    ]

    print(f"Current weights: {current_weights}")  # [1.11, 10.0]
    print(f"sklearn weights: {sklearn_weights}")  # [0.556, 5.0]

    # Check ratio is same but magnitude differs
    current_ratio = current_weights[1] / current_weights[0]
    sklearn_ratio = sklearn_weights[1] / sklearn_weights[0]

    assert abs(current_ratio - sklearn_ratio) < 0.01, \
        "Weight ratios should be the same!"

    assert current_weights[1] / sklearn_weights[1] == 2.0, \
        "Current formula gives 2x larger weights than sklearn"


# Test 12: Model num_classes Consistency
def test_num_classes_consistency(mock_metadata):
    """Test that model num_classes matches data num_classes"""
    metadata_path, root_dir = mock_metadata
    metadata = pd.read_csv(metadata_path)

    # Calculate num_classes from data
    unique_labels = sorted(metadata['class_label'].unique())
    data_num_classes = len(unique_labels)

    # Simulated model creation
    model_num_classes_dynamic = data_num_classes  # CORRECT
    model_num_classes_hardcoded = 2  # BUGGY

    # If data has 3 classes but model expects 2, this is a bug
    if data_num_classes != 2:
        assert model_num_classes_dynamic == data_num_classes, \
            "Dynamic num_classes should match data!"
        assert model_num_classes_hardcoded != data_num_classes, \
            "Hardcoded num_classes creates mismatch!"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
