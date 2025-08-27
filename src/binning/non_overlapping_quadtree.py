#!/usr/bin/env python3
"""
Non-Overlapping Quadtree Binner for Earthquake Forecasting

This implementation ensures that each spatial region is uniquely assigned to one bin,
creating a clean grid of non-overlapping squares for better earthquake forecasting.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging


class NonOverlappingQuadtreeBinner:
    """
    Quadtree binner that creates non-overlapping, distinct square regions.
    
    Each bin represents a unique spatial area with no overlap between bins.
    This ensures clean separation of earthquake data for forecasting.
    """
    
    def __init__(self, max_depth: int = 4, min_events: int = 50, merge_threshold: int = 50):
        """
        Initialize the non-overlapping quadtree binner.
        
        Args:
            max_depth: Maximum depth of the quadtree
            min_events: Minimum events per bin
            merge_threshold: Threshold for merging low-count bins
        """
        self.max_depth = max_depth
        self.min_events = min_events
        self.merge_threshold = merge_threshold
        self.bounds = None
        self.bin_centers = None
        self.logger = logging.getLogger(__name__)
        
    def assign_bins(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Assign bin IDs to earthquake locations using non-overlapping quadtree.
        
        Args:
            lats: Array of latitudes
            lons: Array of longitudes
            
        Returns:
            Array of bin IDs corresponding to each location
        """
        self.logger.info("Creating non-overlapping quadtree bins...")
        
        # Get spatial bounds
        min_lon, max_lon = np.min(lons), np.max(lons)
        min_lat, max_lat = np.min(lats), np.max(lats)
        
        # Create initial grid based on max_depth
        initial_bounds = self._create_initial_grid(min_lon, max_lon, min_lat, max_lat)
        
        # Refine grid based on earthquake density
        refined_bounds = self._refine_grid(initial_bounds, lons, lats)
        
        # Merge low-count bins while maintaining non-overlapping property
        final_bounds = self._merge_low_count_bins(refined_bounds, lons, lats)
        
        # Store bounds
        self.bounds = final_bounds
        
        # Assign bin IDs
        bin_ids = self._assign_bin_ids(lons, lats, final_bounds)
        
        return bin_ids
    
    def _create_initial_grid(self, min_lon: float, max_lon: float, 
                            min_lat: float, max_lat: float) -> List[Tuple]:
        """
        Create initial uniform grid based on max_depth.
        
        Args:
            min_lon, max_lon: Longitude bounds
            min_lat, max_lat: Latitude bounds
            
        Returns:
            List of (min_lon, max_lon, min_lat, max_lat) tuples
        """
        # Calculate number of divisions based on max_depth
        num_divisions = 2 ** self.max_depth
        
        # Calculate step sizes
        lon_step = (max_lon - min_lon) / num_divisions
        lat_step = (max_lat - min_lat) / num_divisions
        
        bounds = []
        
        for i in range(num_divisions):
            for j in range(num_divisions):
                bin_min_lon = min_lon + i * lon_step
                bin_max_lon = min_lon + (i + 1) * lon_step
                bin_min_lat = min_lat + j * lat_step
                bin_max_lat = min_lat + (j + 1) * lat_step
                
                bounds.append((bin_min_lon, bin_max_lon, bin_min_lat, bin_max_lat))
        
        self.logger.info(f"Created initial grid with {len(bounds)} bins")
        return bounds
    
    def _refine_grid(self, bounds: List[Tuple], lons: np.ndarray, lats: np.ndarray) -> List[Tuple]:
        """
        Refine grid by subdividing bins with too many events.
        
        Args:
            bounds: Current bin bounds
            lons, lats: Earthquake coordinates
            
        Returns:
            Refined list of bounds
        """
        refined_bounds = []
        
        for min_lon, max_lon, min_lat, max_lat in bounds:
            # Count events in this bin
            mask = (
                (lons >= min_lon) & (lons < max_lon) &
                (lats >= min_lat) & (lats < max_lat)
            )
            event_count = np.sum(mask)
            
            if event_count > self.min_events * 4:  # If significantly over threshold
                # Subdivide this bin
                mid_lon = (min_lon + max_lon) / 2
                mid_lat = (min_lat + max_lat) / 2
                
                # Create 4 sub-bins
                sub_bins = [
                    (min_lon, mid_lon, min_lat, mid_lat),
                    (mid_lon, max_lon, min_lat, mid_lat),
                    (min_lon, mid_lon, mid_lat, max_lat),
                    (mid_lon, max_lon, mid_lat, max_lat)
                ]
                refined_bounds.extend(sub_bins)
            else:
                refined_bounds.append((min_lon, max_lon, min_lat, max_lat))
        
        self.logger.info(f"Refined grid to {len(refined_bounds)} bins")
        return refined_bounds
    
    def _merge_low_count_bins(self, bounds: List[Tuple], lons: np.ndarray, lats: np.ndarray) -> List[Tuple]:
        """
        Merge low-count bins while maintaining non-overlapping property.
        
        Args:
            bounds: Current bin bounds
            lons, lats: Earthquake coordinates
            
        Returns:
            Merged list of bounds
        """
        if not bounds:
            return bounds
        
        # Count events in each bin
        bin_counts = {}
        for i, (min_lon, max_lon, min_lat, max_lat) in enumerate(bounds):
            mask = (
                (lons >= min_lon) & (lons < max_lon) &
                (lats >= min_lat) & (lats < max_lat)
            )
            bin_counts[i] = np.sum(mask)
        
        # Find bins below threshold
        low_count_bins = [i for i, count in bin_counts.items() if count < self.merge_threshold]
        
        if not low_count_bins:
            return bounds
        
        # Sort bins by count (lowest first)
        low_count_bins.sort(key=lambda i: bin_counts[i])
        
        merged_bounds = bounds.copy()
        bins_to_remove = set()
        
        for bin_idx in low_count_bins:
            if bin_idx in bins_to_remove:
                continue
                
            # Find best neighbor to merge with
            best_neighbor = self._find_best_merge_neighbor(
                bin_idx, merged_bounds, bin_counts, bins_to_remove
            )
            
            if best_neighbor is not None:
                # Merge bins
                bin1 = merged_bounds[bin_idx]
                bin2 = merged_bounds[best_neighbor]
                
                merged_bin = (
                    min(bin1[0], bin2[0]), max(bin1[1], bin2[1]),
                    min(bin1[2], bin2[2]), max(bin1[3], bin2[3])
                )
                
                # Replace both bins with merged bin
                merged_bounds[bin_idx] = merged_bin
                merged_bounds[best_neighbor] = merged_bin
                
                # Mark for removal (will be cleaned up)
                bins_to_remove.add(best_neighbor)
        
        # Remove duplicate merged bins and clean up
        final_bounds = []
        seen_bounds = set()
        
        for bound in merged_bounds:
            if bound not in seen_bounds:
                final_bounds.append(bound)
                seen_bounds.add(bound)
        
        self.logger.info(f"Merged low-count bins: {len(bounds)} -> {len(final_bounds)}")
        return final_bounds
    
    def _find_best_merge_neighbor(self, bin_idx: int, bounds: List[Tuple], 
                                 bin_counts: Dict, bins_to_remove: set) -> int:
        """
        Find the best neighbor bin to merge with.
        
        Args:
            bin_idx: Index of the bin to merge
            bounds: List of all bin bounds
            bin_counts: Dictionary of bin counts
            bins_to_remove: Set of bins marked for removal
            
        Returns:
            Index of best neighbor to merge with, or None if no good neighbor
        """
        current_bin = bounds[bin_idx]
        best_neighbor = None
        best_score = float('inf')
        
        for i, other_bin in enumerate(bounds):
            if i == bin_idx or i in bins_to_remove:
                continue
            
            # Check if bins are adjacent (share edges)
            if self._are_adjacent_bins(current_bin, other_bin):
                # Score based on count difference (prefer similar counts)
                count_diff = abs(bin_counts[bin_idx] - bin_counts[i])
                if count_diff < best_score:
                    best_score = count_diff
                    best_neighbor = i
        
        return best_neighbor
    
    def _are_adjacent_bins(self, bin1: Tuple, bin2: Tuple, tolerance: float = 1e-6) -> bool:
        """
        Check if two bins are adjacent (share edges).
        
        Args:
            bin1, bin2: Bin bounds as (min_lon, max_lon, min_lat, max_lat)
            tolerance: Numerical tolerance for floating point comparison
            
        Returns:
            True if bins are adjacent, False otherwise
        """
        # Check if bins share a longitude edge
        lon_adjacent = (
            np.isclose(bin1[1], bin2[0], rtol=tolerance) or  # bin1 right edge = bin2 left edge
            np.isclose(bin1[0], bin2[1], rtol=tolerance)     # bin1 left edge = bin2 right edge
        )
        
        # Check if bins share a latitude edge
        lat_adjacent = (
            np.isclose(bin1[3], bin2[2], rtol=tolerance) or  # bin1 top edge = bin2 bottom edge
            np.isclose(bin1[2], bin2[3], rtol=tolerance)     # bin1 bottom edge = bin2 top edge
        )
        
        # Bins are adjacent if they share an edge and overlap in the other dimension
        if lon_adjacent:
            # Check latitude overlap
            lat_overlap = (bin1[2] < bin2[3] + tolerance) and (bin2[2] < bin1[3] + tolerance)
            return lat_overlap
        
        if lat_adjacent:
            # Check longitude overlap
            lon_overlap = (bin1[0] < bin2[1] + tolerance) and (bin2[0] < bin1[1] + tolerance)
            return lon_overlap
        
        return False
    
    def _assign_bin_ids(self, lons: np.ndarray, lats: np.ndarray, 
                        bounds: List[Tuple]) -> np.ndarray:
        """
        Assign bin IDs to earthquake locations.
        
        Args:
            lons, lats: Earthquake coordinates
            bounds: Final bin bounds
            
        Returns:
            Array of bin IDs
        """
        bin_ids = np.full(len(lons), -1, dtype=int)
        
        # Print bin statistics
        print(f"\nFinal Non-Overlapping Bin Statistics:")
        for i, (min_lon, max_lon, min_lat, max_lat) in enumerate(bounds):
            mask = (
                (lons >= min_lon) & (lons < max_lon) &
                (lats >= min_lat) & (lats < max_lat)
            )
            count = np.sum(mask)
            width = max_lon - min_lon
            height = max_lat - min_lat
            print(f"Bin {i}: {width:.2f}° lon x {height:.2f}° lat, {count} events")
            print(f"      Coordinates: lon[{min_lon:.12f}, {max_lon:.12f}], lat[{min_lat:.12f}, {max_lat:.12f}]")
        
        # Assign bin IDs
        print(f"\nAssigning earthquakes to {len(bounds)} non-overlapping bins:")
        
        for bin_id, (min_lon, max_lon, min_lat, max_lat) in enumerate(bounds):
            mask = (
                (lons >= min_lon) & (lons < max_lon) &
                (lats >= min_lat) & (lats < max_lat)
            )
            count = np.sum(mask)
            bin_ids[mask] = bin_id
            print(f"  Bin {bin_id}: {count} earthquakes")
            print(f"      Coordinates: lon[{min_lon:.12f}, {max_lon:.12f}], lat[{min_lat:.12f}, {max_lat:.12f}]")
        
        # Check for unassigned earthquakes
        unassigned = np.sum(bin_ids == -1)
        if unassigned > 0:
            print(f"  ⚠️  {unassigned} earthquakes could not be assigned to any bin")
        
        return bin_ids
    
    def get_bin_bounds(self) -> List[Tuple]:
        """Get the bounds of all bins."""
        return self.bounds if self.bounds is not None else []
    
    def get_bin_count(self) -> int:
        """Get the total number of bins."""
        return len(self.bounds) if self.bounds is not None else 0
    
    def get_bin_statistics(self) -> pd.DataFrame:
        """Get detailed statistics for each bin."""
        if self.bounds is None:
            return pd.DataFrame()
        
        stats = []
        for i, (min_lon, max_lon, min_lat, max_lat) in enumerate(self.bounds):
            stats.append({
                'bin_id': i,
                'min_lon': min_lon,
                'max_lon': max_lon,
                'min_lat': min_lat,
                'max_lat': max_lat,
                'width_deg': max_lon - min_lon,
                'height_deg': max_lat - min_lat,
                'area_deg2': (max_lon - min_lon) * (max_lat - min_lat),
                'center_lon': (min_lon + max_lon) / 2,
                'center_lat': (min_lat + max_lat) / 2
            })
        
        return pd.DataFrame(stats)
