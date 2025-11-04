#!/bin/bash
# Automated setup script for phenotype-detection-pipeline GitHub repository
# Run this after creating your GitHub repo and cloning it locally

set -e  # Exit on error

echo "🚀 Setting up phenotype-detection-pipeline repository..."
echo ""

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository."
    echo "   Please run this script from your cloned repository directory."
    exit 1
fi

echo "✅ Git repository detected"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p src docs examples config tests

# Create __init__.py
echo "📝 Creating src/__init__.py..."
cat > src/__init__.py << 'EOF'
"""
Artifact-Free Phenotype Detection Pipeline

A production-ready deep learning pipeline for detecting biological 
phenotypes while avoiding technical artifacts through intelligent 
class design.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .prepare_dataset import *
from .train_models import *
from .evaluate_models import *
from .analyze_results import *
EOF

# Check if pipeline scripts exist and move them
echo "🔄 Organizing pipeline scripts..."

if [ -f "1_prepare_dataset.py" ]; then
    mv 1_prepare_dataset.py src/prepare_dataset.py
    echo "  ✓ Moved prepare_dataset.py"
elif [ ! -f "src/prepare_dataset.py" ]; then
    echo "  ⚠️  Warning: prepare_dataset.py not found"
fi

if [ -f "2_train_models.py" ]; then
    mv 2_train_models.py src/train_models.py
    echo "  ✓ Moved train_models.py"
elif [ ! -f "src/train_models.py" ]; then
    echo "  ⚠️  Warning: train_models.py not found"
fi

if [ -f "3_evaluate_models.py" ]; then
    mv 3_evaluate_models.py src/evaluate_models.py
    echo "  ✓ Moved evaluate_models.py"
elif [ ! -f "src/evaluate_models.py" ]; then
    echo "  ⚠️  Warning: evaluate_models.py not found"
fi

if [ -f "4_analyze_results.py" ]; then
    mv 4_analyze_results.py src/analyze_results.py
    echo "  ✓ Moved analyze_results.py"
elif [ ! -f "src/analyze_results.py" ]; then
    echo "  ⚠️  Warning: analyze_results.py not found"
fi

# Move documentation
echo "📚 Organizing documentation..."

if [ -f "QUICKSTART.md" ]; then
    mv QUICKSTART.md docs/
    echo "  ✓ Moved QUICKSTART.md"
fi

if [ -f "METHODOLOGY.md" ]; then
    mv METHODOLOGY.md docs/
    echo "  ✓ Moved METHODOLOGY.md"
fi

if [ -f "PACKAGE_MANIFEST.md" ]; then
    mv PACKAGE_MANIFEST.md docs/USAGE.md
    echo "  ✓ Moved PACKAGE_MANIFEST.md → USAGE.md"
fi

if [ -f "INDEX.md" ]; then
    mv INDEX.md docs/
    echo "  ✓ Moved INDEX.md"
fi

# Move examples
echo "📖 Organizing examples..."

if [ -f "example_commands.sh" ]; then
    mv example_commands.sh examples/
    echo "  ✓ Moved example_commands.sh"
fi

if [ -f "expected_outputs.md" ]; then
    mv expected_outputs.md examples/
    echo "  ✓ Moved expected_outputs.md"
fi

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "📦 Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0

# Data handling
pandas>=1.5.0
numpy>=1.24.0
h5py>=3.8.0
Pillow>=9.5.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine learning utilities
scikit-learn>=1.2.0
tqdm>=4.65.0

# Optional but recommended
tensorboard>=2.13.0
EOF
    echo "  ✓ Created requirements.txt"
else
    echo "  ℹ️  requirements.txt already exists"
fi

# Create/update .gitignore
if [ ! -f ".gitignore" ]; then
    echo "🚫 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
ENV/
*.egg-info/

# PyTorch
*.pth
*.ckpt

# Data - IMPORTANT: Don't commit large files
data/
dataset/
datasets/
*.h5
*.hdf5
raw_data/
processed_data/

# Results
results/
outputs/
models/
checkpoints/
logs/
evaluation/
analysis/
figures/

# Images and arrays (except documentation)
*.png
!docs/**/*.png
*.jpg
*.jpeg
*.npy
*.npz

# TensorBoard
runs/
lightning_logs/

# OS
.DS_Store
Thumbs.db

# IDEs
.vscode/
.idea/
*.swp

# Environment
.env
*.log
EOF
    echo "  ✓ Created .gitignore"
else
    echo "  ℹ️  .gitignore already exists"
fi

# Create example config
echo "⚙️  Creating example config..."
cat > config/default_config.yaml << 'EOF'
# Default configuration for phenotype detection pipeline

# Data paths
data:
  hdf5_dir: "/path/to/hdf5_neurons"
  output_dir: "./dataset"
  bad_wells: ["A1", "A2", "A3"]

# Training parameters
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  num_workers: 4
  architecture: "resnet18"

# Channels to train
channels:
  - "RFP1"
  - "Halo"

# Tasks
tasks:
  - "6class"

# Evaluation
evaluation:
  channels:
    - "RFP1"
    - "Halo"
EOF
echo "  ✓ Created default_config.yaml"

# Create simple test file
echo "🧪 Creating test template..."
cat > tests/test_prepare_dataset.py << 'EOF'
"""
Unit tests for prepare_dataset.py

Run with: pytest tests/
"""

import pytest
import pandas as pd
from pathlib import Path

def test_imports():
    """Test that all imports work"""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from prepare_dataset import extract_condition_info
        assert True
    except ImportError:
        pytest.fail("Failed to import prepare_dataset")

def test_metadata_structure():
    """Test metadata DataFrame structure"""
    # Add your tests here
    pass

# Add more tests as needed
EOF
echo "  ✓ Created test template"

# Create simple setup.py for installation
echo "📦 Creating setup.py..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phenotype-detection-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Artifact-free phenotype detection using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/phenotype-detection-pipeline",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "h5py>=3.8.0",
        "Pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
    ],
)
EOF
echo "  ✓ Created setup.py"

# Stage all changes
echo ""
echo "📋 Staging changes..."
git add .

# Show status
echo ""
echo "📊 Repository status:"
git status

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Directory structure created:"
echo "   src/       - Source code"
echo "   docs/      - Documentation"
echo "   examples/  - Usage examples"
echo "   config/    - Configuration files"
echo "   tests/     - Unit tests"
echo ""
echo "📝 Next steps:"
echo "   1. Review the staged changes: git status"
echo "   2. Update README.md with your info"
echo "   3. Update setup.py with your details"
echo "   4. Commit changes: git commit -m 'Initial repository setup'"
echo "   5. Push to GitHub: git push origin main"
echo "   6. Start using with Claude Code!"
echo ""
echo "🤖 To use with Claude Code:"
echo "   $ claude-code"
echo "   Then: 'Help me customize this pipeline for my experiment'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
