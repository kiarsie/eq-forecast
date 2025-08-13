#!/usr/bin/env python3
"""
Test script to verify the merging logic for bins 10 and 11.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from binning.quadtree import clean_duplicate_bins, merge_low_count_bins_nearest

# Test data based on the problematic bins from the notebook
test_bounds = [
    (2.09, 4.18, 2.00, 4.50),  # Bin 0
    (2.09, 4.18, 4.50, 7.00),  # Bin 1
    (2.09, 4.18, 7.00, 9.50),  # Bin 2
    (2.09, 4.18, 9.50, 12.00), # Bin 3
    (2.09, 4.18, 12.00, 14.50), # Bin 4
    (2.09, 4.18, 14.50, 17.00), # Bin 5
    (2.09, 4.18, 17.00, 19.50), # Bin 6
    (2.09, 4.18, 19.50, 22.00), # Bin 7
    (4.18, 8.35, 2.00, 7.00),   # Bin 8
    (8.35, 12.52, 2.00, 7.00),  # Bin 9
    (4.18, 6.26, 7.00, 9.50),   # Bin 10 - PROBLEMATIC
    (4.18, 6.26, 9.50, 12.00),  # Bin 11 - PROBLEMATIC
    (8.35, 12.52, 7.00, 12.00), # Bin 12
    (4.18, 6.26, 12.00, 14.50), # Bin 13
    (8.35, 12.52, 12.00, 17.00), # Bin 14
    (8.35, 12.52, 17.00, 22.00), # Bin 15
    (4.18, 6.26, 14.50, 17.00), # Bin 16
    (4.18, 6.26, 17.00, 19.50), # Bin 17
    (4.18, 6.26, 19.50, 22.00), # Bin 18
    (4.18, 6.26, 22.00, 24.50), # Bin 19
    (4.18, 8.35, 7.00, 12.00),  # Bin 20
    (8.35, 16.70, 2.00, 12.00), # Bin 21
]

print(f"Original number of bins: {len(test_bounds)}")
print("Original bounds:")
for i, b in enumerate(test_bounds):
    print(f"  Bin {i}: {b[1] - b[0]:.2f}° lon x {b[3] - b[2]:.2f}° lat")

# Test cleaning duplicates
print(f"\n{'='*50}")
cleaned_bounds = clean_duplicate_bins(test_bounds, tolerance=0.01)
print(f"After cleaning duplicates: {len(cleaned_bounds)} bins")

# Test merging
print(f"\n{'='*50}")
# Create a mock catalog for testing
import numpy as np

class MockCatalog:
    def get_longitudes(self):
        return np.array([120.0] * 100)  # Mock longitude values
    def get_latitudes(self):
        return np.array([10.0] * 100)   # Mock latitude values

mock_catalog = MockCatalog()
merged_bounds = merge_low_count_bins_nearest(mock_catalog, cleaned_bounds.copy(), threshold=50)
print(f"After merging: {len(merged_bounds)} bins")

print(f"\nFinal result: {len(merged_bounds)} bins (reduced from {len(test_bounds)})")
