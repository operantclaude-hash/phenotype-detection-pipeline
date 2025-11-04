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
