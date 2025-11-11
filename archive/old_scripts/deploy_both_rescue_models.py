#!/usr/bin/env python3
"""Deploy both T15 and T0-T15 rescue models on full dataset"""
import sys
sys.path.insert(0, 'src')

# Use the existing deployment framework
# Create wrapper to deploy both models

# Model 1: T15 10000ms (72.5%)
# Model 2: T0-T15 balanced (68%)

# Output: CSV with scores from both models for comparison
