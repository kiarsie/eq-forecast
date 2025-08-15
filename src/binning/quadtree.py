import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.core.regions import CartesianGrid2D
from typing import List, Tuple

class QuadtreeNode:
    def __init__(self, bounds, depth=0):
        self.bounds = bounds  # (min_lon, max_lon, min_lat, max_lat)
        self.depth = depth
        self.children = []
        self.events_idx = []
        self.is_leaf = True

    def subdivide(self):
        """Subdivide node into 4 non-overlapping children."""
        min_lon, max_lon, min_lat, max_lat = self.bounds
        mid_lon = (min_lon + max_lon) / 2
        mid_lat = (min_lat + max_lat) / 2

        children = [
            QuadtreeNode((min_lon, mid_lon, min_lat, mid_lat), self.depth + 1),
            QuadtreeNode((mid_lon, max_lon, min_lat, mid_lat), self.depth + 1),
            QuadtreeNode((min_lon, mid_lon, mid_lat, max_lat), self.depth + 1),
            QuadtreeNode((mid_lon, max_lon, mid_lat, max_lat), self.depth + 1)
        ]

        self.children = children
        self.is_leaf = False
        return children

def build_quadtree_recursive(node, lons, lats, max_depth=4, min_events=10):
    """Build quadtree recursively until min_events or max_depth reached."""
    idx = np.where(
        (lons >= node.bounds[0]) & (lons < node.bounds[1]) &
        (lats >= node.bounds[2]) & (lats < node.bounds[3])
    )[0]
    node.events_idx = idx

    if len(idx) <= min_events or node.depth >= max_depth:
        return

    node.subdivide()
    for child in node.children:
        build_quadtree_recursive(child, lons, lats, max_depth, min_events)

def extract_leaf_bounds(node):
    """Extract bounds from all leaf nodes (non-overlapping)."""
    if node.is_leaf:
        return [node.bounds]
    
    bounds = []
    for child in node.children:
        bounds.extend(extract_leaf_bounds(child))
    return bounds

def count_events_in_bin(bounds, lons, lats):
    """Count events in a specific bin."""
    min_lon, max_lon, min_lat, max_lat = bounds
    mask = (
        (lons >= min_lon) & (lons < max_lon) &
        (lats >= min_lat) & (lats < max_lat)
    )
    return np.sum(mask)

def are_bins_adjacent(bin1, bin2, tolerance=1e-6):
    """Check if two bins are adjacent (share edges)."""
    lon_adjacent = (
        abs(bin1[1] - bin2[0]) < tolerance or
        abs(bin1[0] - bin2[1]) < tolerance
    )
    
    lat_adjacent = (
        abs(bin1[3] - bin2[2]) < tolerance or
        abs(bin1[2] - bin2[3]) < tolerance
    )
    
    if lon_adjacent:
        lat_overlap = (bin1[2] < bin2[3] + tolerance) and (bin2[2] < bin1[3] + tolerance)
        return lat_overlap
    
    if lat_adjacent:
        lon_overlap = (bin1[0] < bin2[1] + tolerance) and (bin2[0] < bin1[1] + tolerance)
        return lon_overlap
    
    return False

def merge_adjacent_bins(bounds, lons, lats, threshold=50, max_bin_size=10.0):
    """
    Merge low-count bins with adjacent neighbors while maintaining complete coverage.
    
    Args:
        bounds: List of bin bounds
        lons, lats: Earthquake coordinates
        threshold: Event count threshold below which bins are merged
        max_bin_size: Maximum size for any bin dimension
        
    Returns:
        List of merged bounds
    """
    if not bounds:
        return bounds
    
    print(f"🔄 Starting merge process with {len(bounds)} bins, threshold: {threshold}")
    
    # Count events in each bin
    bin_counts = {b: count_events_in_bin(b, lons, lats) for b in bounds}
    
    # Find bins below threshold
    low_count_bins = [b for b, count in bin_counts.items() if count < threshold]
    
    if not low_count_bins:
        print("✅ All bins meet threshold, no merging needed")
        return bounds
    
    print(f"📊 {len(low_count_bins)} bins below threshold {threshold}")
    
    # Sort by count (lowest first) to merge worst bins first
    low_count_bins.sort(key=lambda b: bin_counts[b])
    
    # Create a copy to work with
    working_bounds = bounds.copy()
    bins_to_remove = set()
    
    # Iterate until no more merges are possible
    iteration = 0
    max_iterations = 100
    
    while iteration < max_iterations and low_count_bins:
        iteration += 1
        print(f"  🔄 Iteration {iteration}: {len(low_count_bins)} bins below threshold")
        
        merged_this_iteration = False
        
        for low_bin in low_count_bins[:]:  # Copy list to avoid modification during iteration
            if low_bin not in working_bounds or low_bin in bins_to_remove:
                continue
            
            # Find best adjacent neighbor to merge with
            best_neighbor = None
            best_score = float('inf')
            
            for other_bin in working_bounds:
                if other_bin == low_bin or other_bin in bins_to_remove:
                    continue
                
                if are_bins_adjacent(low_bin, other_bin):
                    # Score based on count (prefer merging with lower count bins)
                    other_count = bin_counts[other_bin]
                    score = other_count
                    
                    if score < best_score:
                        best_score = score
                        best_neighbor = other_bin
            
            if best_neighbor is not None:
                # Create merged bin
                merged_bin = (
                    min(low_bin[0], best_neighbor[0]), max(low_bin[1], best_neighbor[1]),
                    min(low_bin[2], best_neighbor[2]), max(low_bin[3], best_neighbor[3])
                )
                
                # Check if merged bin would be too large
                merged_width = merged_bin[1] - merged_bin[0]
                merged_height = merged_bin[3] - merged_bin[2]
                
                if merged_width > max_bin_size or merged_height > max_bin_size:
                    print(f"    ⚠️  Skipping merge: would create bin too large ({merged_width:.2f}° × {merged_height:.2f}°)")
                    continue
                
                # CRITICAL FIX: Check if merged bin would overlap with any existing bins
                would_overlap = False
                for existing_bin in working_bounds:
                    if existing_bin == low_bin or existing_bin == best_neighbor or existing_bin in bins_to_remove:
                        continue
                    
                    if has_overlap(merged_bin, existing_bin):
                        print(f"    ⚠️  Skipping merge: would create overlapping bin with existing bin")
                        would_overlap = True
                        break
                
                if would_overlap:
                    continue
                
                # Remove both original bins and add merged bin
                working_bounds.remove(low_bin)
                working_bounds.remove(best_neighbor)
                working_bounds.append(merged_bin)
                
                # Mark for removal
                bins_to_remove.add(best_neighbor)
                
                # Update count for merged bin
                new_count = bin_counts[low_bin] + bin_counts[best_neighbor]
                bin_counts[merged_bin] = new_count
                
                print(f"    ✅ Merged bins: {low_bin} + {best_neighbor} → {merged_bin} ({new_count} events)")
                
                # Remove from low_count_bins if it now meets threshold
                if new_count >= threshold:
                    low_count_bins.remove(low_bin)
                
                merged_this_iteration = True
                break  # Restart iteration after each merge
        
        if not merged_this_iteration:
            print(f"    ⚠️  No more merges possible after {iteration} iterations")
            break
    
    # Remove any remaining duplicate bins
    final_bounds = []
    seen_bounds = set()
    
    for bound in working_bounds:
        if bound not in seen_bounds:
            final_bounds.append(bound)
            seen_bounds.add(bound)
    
    print(f"🎯 Final result: {len(bounds)} → {len(final_bounds)} bins")
    
    # Verify no overlaps in final result (should be clean now)
    print("🔍 Verifying no overlaps in final bins...")
    overlaps_found = 0
    for i, b1 in enumerate(final_bounds):
        for j, b2 in enumerate(final_bounds[i+1:], i+1):
            if has_overlap(b1, b2):
                overlaps_found += 1
                print(f"  ❌ OVERLAP FOUND: Bin {i} {b1} overlaps with Bin {j} {b2}")
    
    if overlaps_found == 0:
        print("✅ No overlaps detected in final bins")
    else:
        print(f"❌ {overlaps_found} overlaps detected!")
        # Remove overlapping bins to ensure clean result
        final_bounds = remove_overlapping_bins(final_bounds)
    
    # Verify coverage - ensure we haven't lost any regions
    print("🔍 Verifying complete coverage...")
    original_coverage = get_coverage_area(bounds)
    final_coverage = get_coverage_area(final_bounds)
    
    coverage_ratio = final_coverage / original_coverage if original_coverage > 0 else 0
    print(f"  📊 Coverage: {coverage_ratio:.2%} ({original_coverage:.2f} → {final_coverage:.2f} square degrees)")
    
    if coverage_ratio < 0.95:  # Allow 5% loss due to merging
        print("  ⚠️  WARNING: Significant coverage loss detected!")
        print("  🔧 Attempting to restore coverage...")
        final_bounds = restore_coverage(bounds, final_bounds, lons, lats)
    
    return final_bounds

def get_coverage_area(bounds):
    """Calculate total coverage area of all bins."""
    total_area = 0
    for min_lon, max_lon, min_lat, max_lat in bounds:
        width = max_lon - min_lon
        height = max_lat - min_lat
        total_area += width * height
    return total_area

def restore_coverage(original_bounds, merged_bounds, lons, lats):
    """Restore coverage by adding back bins that were lost during merging."""
    print("  🔧 Restoring coverage...")
    
    # Find bins that were lost
    lost_bins = []
    for orig_bin in original_bounds:
        if orig_bin not in merged_bounds:
            # Check if this bin's area is covered by merged bins
            if not is_bin_covered(orig_bin, merged_bounds):
                lost_bins.append(orig_bin)
    
    if lost_bins:
        print(f"    📍 Found {len(lost_bins)} lost bins, restoring...")
        restored_bounds = merged_bounds + lost_bins
        
        # Remove duplicates
        final_bounds = []
        seen_bounds = set()
        for bound in restored_bounds:
            if bound not in seen_bounds:
                final_bounds.append(bound)
                seen_bounds.add(bound)
        
        print(f"    ✅ Restored coverage: {len(merged_bounds)} → {len(final_bounds)} bins")
        return final_bounds
    else:
        print("    ✅ No lost bins detected")
        return merged_bounds

def is_bin_covered(bin_to_check, covering_bins):
    """Check if a bin is completely covered by a set of covering bins."""
    min_lon, max_lon, min_lat, max_lat = bin_to_check
    
    # Check if any covering bin completely contains this bin
    for cover_bin in covering_bins:
        c_min_lon, c_max_lon, c_min_lat, c_max_lat = cover_bin
        
        if (c_min_lon <= min_lon and c_max_lon >= max_lon and 
            c_min_lat <= min_lat and c_max_lat >= max_lat):
            return True
    
    return False

def has_overlap(bin1, bin2):
    """Check if two bins overlap."""
    return not (bin1[1] <= bin2[0] or bin2[1] <= bin1[0] or 
               bin1[3] <= bin2[2] or bin2[3] <= bin1[2])

def remove_overlapping_bins(bounds):
    """Remove overlapping bins to ensure clean, non-overlapping result."""
    if not bounds:
        return bounds
    
    print("🧹 Cleaning overlapping bins...")
    clean_bounds = []
    
    for bound in bounds:
        is_overlapping = False
        
        for existing_bound in clean_bounds:
            if has_overlap(bound, existing_bound):
                is_overlapping = True
                print(f"  🗑️  Removing overlapping bin: {bound}")
                break
        
        if not is_overlapping:
            clean_bounds.append(bound)
    
    print(f"  📊 Cleaned: {len(bounds)} → {len(clean_bounds)} bins")
    return clean_bounds

def apply_quadtree_binning(catalog, max_depth=4, min_events=10, merge_threshold=None, max_bin_size=10.0):
    """Apply quadtree binning to catalog and return filtered catalog, region, and bounds."""
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    min_lon, max_lon = np.min(lons), np.max(lons)
    min_lat, max_lat = np.min(lats), np.max(lats)

    print(f"🌍 Catalog bounds: {min_lon:.2f}° to {max_lon:.2f}° lon, {min_lat:.2f}° to {max_lat:.2f}° lat")
    print(f"📊 Total events: {len(lons)}")
    print(f"🌳 Building quadtree with max_depth={max_depth}, min_events={min_events}")

    root = QuadtreeNode((min_lon, max_lon, min_lat, max_lat))
    build_quadtree_recursive(root, lons, lats, max_depth, min_events)

    unmerged_bounds = extract_leaf_bounds(root)
    print(f"🍃 Extracted {len(unmerged_bounds)} leaf nodes from quadtree")
    
    original_bounds = unmerged_bounds.copy()
    
    if merge_threshold is not None:
        print(f"\n🔄 Applying merging with threshold: {merge_threshold}")
        print(f"Before merging: {len(unmerged_bounds)} bins")
        
        merged_bounds = merge_adjacent_bins(
            unmerged_bounds, lons, lats, threshold=merge_threshold, max_bin_size=max_bin_size
        )
        
        print(f"After merging: {len(merged_bounds)} bins")
        final_bounds = merged_bounds
    else:
        final_bounds = unmerged_bounds

    # Create mask for events in final bins
    mask = np.full(len(lons), False)
    for (min_lon, max_lon, min_lat, max_lat) in final_bounds:
        in_lon = (lons >= min_lon) & (lons < max_lon)
        in_lat = (lats >= min_lat) & (lats < max_lat)
        mask |= (in_lon & in_lat)

    # Create filtered DataFrame
    filtered_mags = catalog.get_magnitudes()[mask]
    filtered_depths = catalog.get_depths()[mask] if catalog.get_depths() is not None else np.full(np.sum(mask), np.nan)
    filtered_times = [datetime_to_utc_epoch(dt) for dt in np.array(catalog.get_datetimes())[mask]]

    df = pd.DataFrame({
        'id': np.arange(np.sum(mask)),
        'magnitude': filtered_mags,
        'latitude': lats[mask],
        'longitude': lons[mask],
        'depth': filtered_depths,
        'origin_time': filtered_times
    })

    # Create region from bin centers
    bin_centers = []
    for (min_lon, max_lon, min_lat, max_lat) in final_bounds:
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        bin_centers.append([center_lon, center_lat])
    
    bin_centers = np.array(bin_centers)
    region = CartesianGrid2D.from_origins(bin_centers, dh=1.0)
    filtered_catalog = CSEPCatalog.from_dataframe(df, region=region)

    return filtered_catalog, region, final_bounds, original_bounds

def plot_quadtree_comparison(original_catalog, unmerged_bounds, merged_bounds, 
                            merge_threshold=None, save_path=None):
    """Plot both unmerged and merged quadtree grids for comparison."""
    print(f"🎨 Creating quadtree comparison plot...")
    print(f"📊 Unmerged bins: {len(unmerged_bounds)}")
    print(f"📊 Merged bins: {len(merged_bounds)}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), 
                                   subplot_kw={'projection': ccrs.PlateCarree()})
    
    lons = original_catalog.get_longitudes()
    lats = original_catalog.get_latitudes()
    min_lon, max_lon = np.min(lons), np.max(lons)
    min_lat, max_lat = np.min(lats), np.max(lats)
    
    lon_pad = (max_lon - min_lon) * 0.05
    lat_pad = (max_lat - min_lat) * 0.05
    
    extent = [min_lon - lon_pad, max_lon + lon_pad, 
              min_lat - lat_pad, max_lat + lat_pad]
    
    for ax in [ax1, ax2]:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)

    # Plot 1: Unmerged quadtree
    ax1.scatter(lons, lats, s=10, color='blue', alpha=0.3, 
                label='All Events', transform=ccrs.PlateCarree())
    
    for i, (min_lon, max_lon, min_lat, max_lat) in enumerate(unmerged_bounds):
        rect = plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                           edgecolor='red', facecolor='none', linewidth=1.5,
                             transform=ccrs.PlateCarree())
        ax1.add_patch(rect)
        
        count = count_events_in_bin((min_lon, max_lon, min_lat, max_lat), lons, lats)
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        ax1.text(center_lon, center_lat, str(count), 
                ha='center', va='center', fontsize=8, fontweight='bold',
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax1.set_title(f"Unmerged Quadtree ({len(unmerged_bounds)} bins)", fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Plot 2: Merged quadtree
    ax2.scatter(lons, lats, s=10, color='blue', alpha=0.3, 
                label='All Events', transform=ccrs.PlateCarree())
    
    for i, (min_lon, max_lon, min_lat, max_lat) in enumerate(merged_bounds):
        rect = plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                           edgecolor='green', facecolor='none', linewidth=2,
                           transform=ccrs.PlateCarree())
        ax2.add_patch(rect)
        
        count = count_events_in_bin((min_lon, max_lon, min_lat, max_lat), lons, lats)
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        ax2.text(center_lon, center_lat, str(count), 
                ha='center', va='center', fontsize=8, fontweight='bold',
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    merge_title = f"Merged Quadtree ({len(merged_bounds)} bins)"
    if merge_threshold:
        merge_title += f"\nThreshold: <{merge_threshold} events"
    ax2.set_title(merge_title, fontsize=14, fontweight='bold')
    ax2.legend()
    
    fig.suptitle("Quadtree Binning Comparison: Unmerged vs Merged", 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison plot saved to: {save_path}")
    
    plt.show()
    print("✅ Comparison plot completed!")

class QuadtreeBinner:
    """Wrapper class for quadtree binning functionality."""
    
    def __init__(self, max_depth: int = 4, min_events: int = 10, merge_threshold: int = 50, max_bin_size: float = 10.0, 
                 custom_bounds: Tuple[float, float, float, float] = None):
        self.max_depth = max_depth
        self.min_events = min_events
        self.merge_threshold = merge_threshold
        self.max_bin_size = max_bin_size
        self.custom_bounds = custom_bounds  # (min_lon, max_lon, min_lat, max_lat)
        self.bounds = None
        self.unmerged_bounds = None
        
    def assign_bins(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Assign bin IDs to earthquake locations using non-overlapping quadtree."""
        print("🔧 Creating non-overlapping quadtree bins...")
        
        if self.custom_bounds:
            print(f"🔧 Using custom bounds: {self.custom_bounds}")
            min_lon, max_lon, min_lat, max_lat = self.custom_bounds
            
            # Filter data to custom bounds
            mask = (
                (lons >= min_lon) & (lons < max_lon) &
                (lats >= min_lat) & (lats < max_lat)
            )
            filtered_lons = lons[mask]
            filtered_lats = lats[mask]
            
            print(f"🌍 Custom bounds: {min_lon:.2f}° to {max_lon:.2f}° lon, {min_lat:.2f}° to {max_lat:.2f}° lat")
            print(f"📊 Filtered events: {len(filtered_lons)} (from {len(lons)} total)")
            
            root = QuadtreeNode((min_lon, max_lon, min_lat, max_lat))
            build_quadtree_recursive(root, filtered_lons, filtered_lats, self.max_depth, self.min_events)
            unmerged_bounds = extract_leaf_bounds(root)
            self.unmerged_bounds = unmerged_bounds
            print(f"🍃 Extracted {len(unmerged_bounds)} leaf nodes from quadtree with custom bounds")
        else:
            min_lon, max_lon = np.min(lons), np.max(lons)
            min_lat, max_lat = np.min(lats), np.max(lats)
            
            print(f"🌍 Spatial bounds: {min_lon:.2f}° to {max_lon:.2f}° lon, {min_lat:.2f}° to {max_lat:.2f}° lat")
            print(f"📊 Total events: {len(lons)}")
            
            root = QuadtreeNode((min_lon, max_lon, min_lat, max_lat))
            build_quadtree_recursive(root, lons, lats, self.max_depth, self.min_events)
        
            unmerged_bounds = extract_leaf_bounds(root)
            self.unmerged_bounds = unmerged_bounds
        
            print(f"🍃 Extracted {len(unmerged_bounds)} leaf nodes from quadtree")
            filtered_lons, filtered_lats = lons, lats
            mask = np.ones(len(lons), dtype=bool)  # All events included
        
        if self.merge_threshold is not None:
            print(f"\n🔄 Applying merging with threshold: ≥{self.merge_threshold} events")
            print(f"Before merging: {len(unmerged_bounds)} bins")
            print(f"⚠️  Note: Will preserve complete coverage of catalog area")
            
            # Use enhanced merging that allows L-shaped and irregular merges
            merged_bounds = enhanced_merge_adjacent_bins(
                unmerged_bounds, filtered_lons, filtered_lats, threshold=self.merge_threshold, max_bin_size=self.max_bin_size
            )
            
            print(f"After merging: {len(merged_bounds)} bins")
            self.bounds = merged_bounds
            
            # Verify all bins meet threshold
            print("\n🔍 Verifying all bins meet threshold...")
            all_meet_threshold = True
            for i, b in enumerate(merged_bounds):
                count = count_events_in_bin(b, filtered_lons, filtered_lats)
                width = b[1] - b[0]
                height = b[3] - b[2]
                
                if count < self.merge_threshold:
                    all_meet_threshold = False
                    print(f"  ❌ Bin {i}: {width:.2f}° lon × {height:.2f}° lat, {count} events (below threshold)")
                else:
                    print(f"  ✅ Bin {i}: {width:.2f}° lon × {height:.2f}° lat, {count} events")
            
            if all_meet_threshold:
                print("🎉 All bins meet threshold!")
            else:
                print("⚠️  Some bins still below threshold (this is expected for isolated regions)")
        else:
            print("⚠️  No merging applied")
            self.bounds = unmerged_bounds
        
        # Assign bin IDs to original data using the same mask
        bin_ids = np.full(len(lons), -1, dtype=int)
        
        print(f"\nAssigning earthquakes to {len(self.bounds)} non-overlapping bins:")
        
        # Only assign to events within the custom bounds
        for bin_id, (min_lon, max_lon, min_lat, max_lat) in enumerate(self.bounds):
            bin_mask = (
                (lons >= min_lon) & (lons < max_lon) &
                (lats >= min_lat) & (lats < max_lat)
            )
            # Only assign to events that are also in the filtered mask
            if self.custom_bounds:
                bin_mask = bin_mask & mask
            count = np.sum(bin_mask)
            bin_ids[bin_mask] = bin_id
            print(f"  Bin {bin_id}: {count} earthquakes")
        
        unassigned = np.sum(bin_ids == -1)
        if unassigned > 0:
            print(f"  ⚠️  {unassigned} earthquakes could not be assigned to any bin")
            if self.custom_bounds:
                print(f"     (This includes {np.sum(~mask)} events outside custom bounds)")
            
        return bin_ids
    
    def get_bin_bounds(self) -> List[Tuple]:
        return self.bounds if self.bounds is not None else []
    
    def get_unmerged_bounds(self) -> List[Tuple]:
        return self.unmerged_bounds if self.unmerged_bounds is not None else []
    
    def get_bin_count(self) -> int:
        return len(self.bounds) if self.bounds is not None else 0
    
    def plot_comparison(self, lats: np.ndarray, lons: np.ndarray, save_path: str = None):
        """Plot both unmerged and merged quadtree grids for comparison."""
        if self.unmerged_bounds is None or self.bounds is None:
            print("❌ No bounds available. Run assign_bins() first.")
            return
        
        class MockCatalog:
            def __init__(self, lons, lats):
                self._lons = lons
                self._lats = lats
            
            def get_longitudes(self):
                return self._lons
            
            def get_latitudes(self):
                return self._lats
        
        mock_catalog = MockCatalog(lons, lats)
        
        plot_quadtree_comparison(
            mock_catalog, 
            self.unmerged_bounds, 
            self.bounds, 
            merge_threshold=self.merge_threshold,
            save_path=save_path
        )

def can_merge_multiple_bins(bins, tolerance=1e-6):
    """
    Check if multiple bins can be merged (they form a connected region).
        
        Args:
        bins: List of bin bounds
        tolerance: Tolerance for adjacency
            
        Returns:
        True if bins can be merged, False otherwise
    """
    if len(bins) < 2:
        return False
    
    # Check if all bins are connected (adjacent to at least one other bin)
    connected = set()
    connected.add(0)  # Start with first bin
    
    while True:
        initial_size = len(connected)
        for i in range(len(bins)):
            if i in connected:
                continue
            # Check if bin i is adjacent to any connected bin
            for j in connected:
                if are_bins_adjacent(bins[i], bins[j], tolerance):
                    connected.add(i)
                break
            
        if len(connected) == initial_size:
            break
    
    return len(connected) == len(bins)

def merge_multiple_bins(bins):
    """
    Merge multiple bins into a single bounding box.
        
        Args:
        bins: List of bin bounds
            
        Returns:
        Merged bin bounds
    """
    if not bins:
        return None
    
    min_lon = min(b[0] for b in bins)
    max_lon = max(b[1] for b in bins)
    min_lat = min(b[2] for b in bins)
    max_lat = max(b[3] for b in bins)
    
    return (min_lon, max_lon, min_lat, max_lat)

def find_mergeable_groups(bounds, bin_counts, threshold, max_bin_size):
    """
    Find groups of bins that can be merged to meet the threshold.
        
        Args:
        bounds: List of all bin bounds
        bin_counts: Dictionary mapping bounds to event counts
        threshold: Target event count threshold
        max_bin_size: Maximum bin size constraint
            
        Returns:
        List of merge groups (each group is a list of bin bounds)
    """
    # Find bins below threshold
    low_count_bins = [b for b, count in bin_counts.items() if count < threshold]
    
    if not low_count_bins:
        return []
    
    merge_groups = []
    used_bins = set()
    
    # Try to find groups of 2-8 bins that can be merged (much more aggressive)
    for group_size in range(2, 9):  # Try groups of 2, 3, 4, 5, 6, 7, 8 bins
        for i, bin1 in enumerate(low_count_bins):
            if bin1 in used_bins:
                continue
                
            # Find all possible groups starting with bin1
            potential_groups = []
            
            def find_groups_recursive(current_group, remaining_bins):
                if len(current_group) == group_size:
                    # Check if this group can be merged
                    if can_merge_multiple_bins(current_group):
                        total_count = sum(bin_counts[b] for b in current_group)
                        merged_bin = merge_multiple_bins(current_group)
                        merged_width = merged_bin[1] - merged_bin[0]
                        merged_height = merged_bin[3] - merged_bin[2]
                        
                        # Check size constraints
                        if merged_width <= max_bin_size and merged_height <= max_bin_size:
                            # CRITICAL: Check if merged bin would overlap with any existing bins
                            would_overlap = False
                            for existing_bin in bounds:
                                if existing_bin in current_group or existing_bin in used_bins:
                                    continue
                                if has_overlap(merged_bin, existing_bin):
                                    would_overlap = True
                                    break
                            
                            if not would_overlap:
                                potential_groups.append((current_group.copy(), total_count))
                    return
                
                for j, bin2 in enumerate(remaining_bins):
                    if bin2 in used_bins:
                        continue
                    
                    # Check if bin2 is adjacent to any bin in current_group
                    can_add = False
                    for existing_bin in current_group:
                        if are_bins_adjacent(bin2, existing_bin):
                            can_add = True
                            break
                    
                    if can_add:
                        new_group = current_group + [bin2]
                        new_remaining = remaining_bins[j+1:]
                        find_groups_recursive(new_group, new_remaining)
            
            find_groups_recursive([bin1], low_count_bins[i+1:])
            
            # Sort by total count (prefer groups that get closer to threshold)
            potential_groups.sort(key=lambda x: abs(x[1] - threshold))
            
            # Take the best group that doesn't conflict with already used bins
            for group, total_count in potential_groups:
                if not any(b in used_bins for b in group):
                    merge_groups.append(group)
                    used_bins.update(group)
                    break
    
    # If we still have low-count bins, try merging with ANY available bins (not just low-count ones)
    remaining_low_bins = [b for b in low_count_bins if b not in used_bins]
    if remaining_low_bins:
        print(f"  🔧 Trying aggressive merges for {len(remaining_low_bins)} remaining low-count bins...")
        
        for low_bin in remaining_low_bins:
            if low_bin in used_bins:
                continue
                
            # Find ANY adjacent bin to merge with (prefer lower count bins)
            best_neighbor = None
            best_score = float('inf')
            
            for other_bin in bounds:
                if other_bin == low_bin or other_bin in used_bins:
                    continue
            
                if are_bins_adjacent(low_bin, other_bin):
                    # Score based on count (prefer merging with lower count bins)
                    other_count = bin_counts[other_bin]
                    score = other_count
                    
                    if score < best_score:
                        best_score = score
                        best_neighbor = other_bin
            
            if best_neighbor is not None:
                # Create merged bin
                merged_bin = (
                    min(low_bin[0], best_neighbor[0]), max(low_bin[1], best_neighbor[1]),
                    min(low_bin[2], best_neighbor[2]), max(low_bin[3], best_neighbor[3])
                )
                
                # Check if merged bin would be too large
                merged_width = merged_bin[1] - merged_bin[0]
                merged_height = merged_bin[3] - merged_bin[2]
                
                # CRITICAL: Check if merged bin would overlap with any existing bins
                would_overlap = False
                for existing_bin in bounds:
                    if existing_bin == low_bin or existing_bin == best_neighbor or existing_bin in used_bins:
                        continue
                    if has_overlap(merged_bin, existing_bin):
                        would_overlap = True
                        break
                
                if not would_overlap and merged_width <= max_bin_size and merged_height <= max_bin_size:
                    total_count = bin_counts[low_bin] + bin_counts[best_neighbor]
                    merge_groups.append([low_bin, best_neighbor])
                    used_bins.add(low_bin)
                    used_bins.add(best_neighbor)
                    print(f"    🔄 Aggressive merge: {bin_counts[low_bin]} + {bin_counts[best_neighbor]} = {total_count} events")
    
    return merge_groups

def enhanced_merge_adjacent_bins(bounds, lons, lats, threshold=50, max_bin_size=10.0):
    """
    Enhanced merging that allows L-shaped and irregular merges.
        
        Args:
        bounds: List of bin bounds
        lons, lats: Earthquake coordinates
        threshold: Event count threshold below which bins are merged
        max_bin_size: Maximum size for any bin dimension
            
        Returns:
        List of merged bounds
    """
    if not bounds:
        return bounds
    
    print(f"🔄 Starting enhanced merge process with {len(bounds)} bins, threshold: {threshold}")
    
    # Count events in each bin
    bin_counts = {b: count_events_in_bin(b, lons, lats) for b in bounds}
    
    # Find mergeable groups
    merge_groups = find_mergeable_groups(bounds, bin_counts, threshold, max_bin_size)
    
    if not merge_groups:
        print("✅ No mergeable groups found")
    else:
        print(f"🔧 Found {len(merge_groups)} mergeable groups")
        
        # Create a copy to work with
        working_bounds = bounds.copy()
        
        # Apply merges
        for i, group in enumerate(merge_groups):
            total_count = sum(bin_counts[b] for b in group)
            merged_bin = merge_multiple_bins(group)
            
            print(f"  🔄 Merging group {i+1}: {len(group)} bins → {total_count} events")
            print(f"     Bins: {[f'{bin_counts[b]}' for b in group]}")
            print(f"     Result: {merged_bin}")
            
            # Remove original bins and add merged bin
            for b in group:
                if b in working_bounds:
                    working_bounds.remove(b)
            
            working_bounds.append(merged_bin)
        
        print(f"🎯 Result after group merging: {len(bounds)} → {len(working_bounds)} bins")
        bounds = working_bounds
        # Update bin counts for the new bounds
        bin_counts = {b: count_events_in_bin(b, lons, lats) for b in bounds}
    
    # PHASE 2: Specifically target bins below threshold
    bounds = merge_below_threshold_bins(bounds, bin_counts, threshold, max_bin_size)
    
    print(f"🎯 Final result: {len(bounds)} bins")
    print("✅ Enhanced merging completed successfully")
    
    return bounds

def merge_below_threshold_bins(bounds, bin_counts, threshold, max_bin_size):
    """
    Merge bins below threshold with their adjacent neighbors.
    
    Args:
        bounds: List of bin bounds
        bin_counts: Dictionary mapping bounds to event counts
        threshold: Event count threshold
        max_bin_size: Maximum bin size constraint
            
    Returns:
        List of merged bounds
    """
    if not bounds:
        return bounds
        
    print(f"🎯 Targeting bins below threshold {threshold} for adjacent merging...")
            
    # Find bins below threshold
    below_threshold = [b for b, count in bin_counts.items() if count < threshold]
    
    if not below_threshold:
        print("✅ All bins meet threshold, no additional merging needed")
        return bounds
    
    print(f"📊 Found {len(below_threshold)} bins below threshold")
    
    # Sort by count (lowest first) to merge worst bins first
    below_threshold.sort(key=lambda b: bin_counts[b])
    
    working_bounds = bounds.copy()
    used_bins = set()
    merge_pairs = []
    
    for low_bin in below_threshold:
        if low_bin in used_bins:
            continue
            
        # Find ONLY ADJACENT neighbors
        adjacent_neighbors = []
        for other_bin in working_bounds:
            if other_bin == low_bin or other_bin in used_bins:
                continue
                
            # CRITICAL: Only consider ADJACENT bins
            if are_bins_adjacent(low_bin, other_bin):
                adjacent_neighbors.append((other_bin, bin_counts[other_bin]))
        
        if not adjacent_neighbors:
            print(f"  ⚠️  Bin with {bin_counts[low_bin]} events has no adjacent neighbors to merge with")
            continue
                    
        # Sort adjacent neighbors by count (lowest first) to prefer merging with low-count neighbors
        adjacent_neighbors.sort(key=lambda x: x[1])
        
        # Merge with the lowest-count adjacent neighbor
        best_neighbor, neighbor_count = adjacent_neighbors[0]
        
        # Create merged bin
        merged_bin = (
            min(low_bin[0], best_neighbor[0]), max(low_bin[1], best_neighbor[1]),
            min(low_bin[2], best_neighbor[2]), max(low_bin[3], best_neighbor[3])
        )
                    
        # Check if merged bin would be too large
        merged_width = merged_bin[1] - merged_bin[0]
        merged_height = merged_bin[3] - merged_bin[2]
                    
        # Check if merged bin would overlap with any existing bins
        would_overlap = False
        for existing_bin in working_bounds:
            if existing_bin == low_bin or existing_bin == best_neighbor or existing_bin in used_bins:
                continue
            if has_overlap(merged_bin, existing_bin):
                would_overlap = True
                break
        
        if not would_overlap and merged_width <= max_bin_size and merged_height <= max_bin_size:
            total_count = bin_counts[low_bin] + bin_counts[best_neighbor]
            merge_pairs.append((low_bin, best_neighbor, merged_bin, total_count))
            used_bins.add(low_bin)
            used_bins.add(best_neighbor)
            print(f"  🔄 Adjacent merge: {bin_counts[low_bin]} + {bin_counts[best_neighbor]} = {total_count} events")
        else:
            print(f"  ⚠️  Cannot merge bin {bin_counts[low_bin]} with adjacent neighbor {bin_counts[best_neighbor]} (size or overlap constraint)")
    
    # Apply all merges
    if merge_pairs:
        print(f"🔧 Applying {len(merge_pairs)} adjacent merges...")
        
        for low_bin, best_neighbor, merged_bin, total_count in merge_pairs:
            # Remove original bins
            if low_bin in working_bounds:
                working_bounds.remove(low_bin)
            if best_neighbor in working_bounds:
                working_bounds.remove(best_neighbor)
            
            # Add merged bin
            working_bounds.append(merged_bin)
        
        print(f"🎯 Result: {len(bounds)} → {len(working_bounds)} bins")
    else:
        print("⚠️  No adjacent merges possible")
    
    return working_bounds
