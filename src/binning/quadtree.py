import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.core.regions import CartesianGrid2D

#  QUADTREE NODE CLASS
class QuadtreeNode:
    def __init__(self, bounds, depth=0):
        self.bounds = bounds  # (min_lon, max_lon, min_lat, max_lat)
        self.depth = depth
        self.children = []
        self.events_idx = []

    def subdivide(self):
        min_lon, max_lon, min_lat, max_lat = self.bounds
        mid_lon = (min_lon + max_lon) / 2
        mid_lat = (min_lat + max_lat) / 2

        return [
            QuadtreeNode((min_lon, mid_lon, min_lat, mid_lat), self.depth + 1),
            QuadtreeNode((mid_lon, max_lon, min_lat, mid_lat), self.depth + 1),
            QuadtreeNode((min_lon, mid_lon, mid_lat, max_lat), self.depth + 1),
            QuadtreeNode((mid_lon, max_lon, mid_lat, max_lat), self.depth + 1)
        ]

#  BUILD QUADTREE RECURSIVELY
def build_quadtree(node, lons, lats, max_depth=4, min_events=10):
    idx = np.where(
        (lons >= node.bounds[0]) & (lons < node.bounds[1]) &
        (lats >= node.bounds[2]) & (lats < node.bounds[3])
    )[0]
    node.events_idx = idx

    if len(idx) <= min_events or node.depth >= max_depth:
        return

    node.children = node.subdivide()
    for child in node.children:
        build_quadtree(child, lons, lats, max_depth, min_events)

#  EXTRACT GRID CENTERS FROM LEAF NODES
def extract_leaf_origins(node):
    if not node.children:
        min_lon, max_lon, min_lat, max_lat = node.bounds
        return [((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)]
    origins = []
    for child in node.children:
        origins.extend(extract_leaf_origins(child))
    return origins

#  EXTRACT CELL BOUNDARIES FROM LEAVES
def extract_leaf_bounds(node):
    if not node.children:
        return [node.bounds]
    bounds = []
    for child in node.children:
        bounds.extend(extract_leaf_bounds(child))
    return bounds


def find_richest_neighbor(bounds_list, target_bounds, bin_counts):
    """
    Find the adjacent bin (sharing edge or corner) with most events.
    
    Args:
        bounds_list (list): List of all bin bounds
        target_bounds (tuple): Target bin bounds
        bin_counts (dict): Dictionary mapping bounds to event counts
        
    Returns:
        tuple: Bounds of the richest neighbor, or None if no neighbors found
    """
    max_count = -1
    best_neighbor = None
    
    for bounds in bounds_list:
        if bounds == target_bounds:
            continue
        
        # Check adjacency: share at least a point along one edge
        horizontal_overlap = (bounds[2] <= target_bounds[3]) and (bounds[3] >= target_bounds[2])
        vertical_overlap = (bounds[0] <= target_bounds[1]) and (bounds[1] >= target_bounds[0])
        
        adjacent = False
        if horizontal_overlap and (np.isclose(bounds[1], target_bounds[0]) or np.isclose(bounds[0], target_bounds[1])):
            adjacent = True
        if vertical_overlap and (np.isclose(bounds[3], target_bounds[2]) or np.isclose(bounds[2], target_bounds[3])):
            adjacent = True

        if adjacent and bin_counts.get(bounds, 0) > max_count:
            max_count = bin_counts[bounds]
            best_neighbor = bounds
            
    return best_neighbor


def clean_duplicate_bins(bounds, tolerance=0.01):
    """
    Remove duplicate or nearly identical bins from the bounds list.
    
    Args:
        bounds (list): List of bin bounds
        tolerance (float): Tolerance for considering bins as duplicates
        
    Returns:
        list: Cleaned list of bounds
    """
    if not bounds:
        return bounds
    
    cleaned_bounds = []
    for b in bounds:
        is_duplicate = False
        for existing_b in cleaned_bounds:
            if (abs(b[0] - existing_b[0]) < tolerance and
                abs(b[1] - existing_b[1]) < tolerance and
                abs(b[2] - existing_b[2]) < tolerance and
                abs(b[3] - existing_b[3]) < tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            cleaned_bounds.append(b)
    
    return cleaned_bounds

def merge_low_count_bins_nearest(catalog, bounds, threshold=50):
    """
    Merge bins with low event counts by combining them with the nearest neighbor.
    Also merge bins with very similar coordinates to reduce redundancy.
    
    Args:
        catalog: Earthquake catalog
        bounds (list): List of bin bounds
        threshold (int): Event count threshold below which bins are merged
        
    Returns:
        list: Updated list of bounds after merging
    """
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    def count_events(b):
        return np.sum((lons >= b[0]) & (lons < b[1]) &
                      (lats >= b[2]) & (lats < b[3]))

    def bin_center(b):
        return ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2)

    def are_similar_bins(b1, b2, tolerance=0.01):
        """Check if two bins have very similar coordinates (within tolerance)."""
        return (abs(b1[0] - b2[0]) < tolerance and 
                abs(b1[1] - b2[1]) < tolerance and
                abs(b1[2] - b2[2]) < tolerance and 
                abs(b1[3] - b2[3]) < tolerance)

    def are_adjacent_bins(b1, b2, tolerance=0.01):
        """Check if two bins are adjacent (share edges or corners)."""
        # Check if bins share edges
        horizontal_adjacent = (abs(b1[1] - b2[0]) < tolerance or abs(b1[0] - b2[1]) < tolerance)
        vertical_adjacent = (abs(b1[3] - b2[2]) < tolerance or abs(b1[2] - b2[3]) < tolerance)
        
        # Check if they overlap in the other dimension
        lon_overlap = (b1[2] <= b2[3] + tolerance) and (b2[2] <= b1[3] + tolerance)
        lat_overlap = (b1[0] <= b2[1] + tolerance) and (b2[0] <= b1[1] + tolerance)
        
        return (horizontal_adjacent and lat_overlap) or (vertical_adjacent and lon_overlap)

    changed = True
    iteration = 0
    max_iterations = 100  # Prevent infinite loops
    
    while changed and iteration < max_iterations:
        iteration += 1
        changed = False
        bin_counts = {b: count_events(b) for b in bounds}
        
        # First, merge bins with very similar coordinates
        for i, b1 in enumerate(bounds):
            if b1 not in bounds:  # Skip if already removed
                continue
            for j, b2 in enumerate(bounds[i+1:], i+1):
                if b2 not in bounds:  # Skip if already removed
                    continue
                if are_similar_bins(b1, b2):
                    # Merge similar bins
                    merged_bin = (
                        min(b1[0], b2[0]),
                        max(b1[1], b2[1]),
                        min(b1[2], b2[2]),
                        max(b1[3], b2[3])
                    )
                    bounds.remove(b1)
                    bounds.remove(b2)
                    bounds.append(merged_bin)
                    changed = True
                    break
            if changed:
                break
        
        if changed:
            continue
            
        # Then, merge low-count bins with nearest neighbors
        low_bins = [b for b, cnt in bin_counts.items() if cnt < threshold]
        if not low_bins:
            break

        for b in low_bins:
            if b not in bounds:  # Skip if already removed
                continue
            c_b = bin_center(b)
            
            # First try to find adjacent bins
            adjacent_bins = [ob for ob in bounds if ob != b and are_adjacent_bins(b, ob)]
            if adjacent_bins:
                # Merge with adjacent bin
                closest = adjacent_bins[0]
            else:
                # Find closest bin by Euclidean distance
                other_bins = [ob for ob in bounds if ob != b]
                if not other_bins:
                    continue
                closest = min(
                    other_bins,
                    key=lambda ob: np.hypot(c_b[0] - bin_center(ob)[0],
                                            c_b[1] - bin_center(ob)[1])
                )
            
            # Merge bins
            merged_bin = (
                min(b[0], closest[0]),
                max(b[1], closest[1]),
                min(b[2], closest[2]),
                max(b[3], closest[3])
            )
            bounds.remove(b)
            bounds.remove(closest)
            bounds.append(merged_bin)
            changed = True
            break  # restart after each merge

    if iteration >= max_iterations:
        print(f"Warning: Merging stopped after {max_iterations} iterations to prevent infinite loop")
    
    return bounds


def apply_quadtree(catalog, min_events=20, max_depth=4, min_bin_width=0.2):
    
    lons = catalog["longitude"]
    lats = catalog["latitude"]

    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()

    root = QuadtreeNode((min_lon, max_lon, min_lat, max_lat))
    build_quadtree(root, lons, lats, max_depth=max_depth, min_events=min_events)

    bounds = extract_leaf_bounds(root)
    filtered_bounds = []

    for b in bounds:
        width_lon = b[1] - b[0]
        width_lat = b[3] - b[2]
        if width_lon >= min_bin_width and width_lat >= min_bin_width:
            filtered_bounds.append(b)

    # Filter catalog events that fall into any valid bin
    mask = np.zeros(len(lons), dtype=bool)
    for b in filtered_bounds:
        in_bin = (
            (lons >= b[0]) & (lons < b[1]) &
            (lats >= b[2]) & (lats < b[3])
        )
        mask |= in_bin

    filtered_catalog = catalog.filter(mask)
    return filtered_catalog, root, filtered_bounds
    
#  APPLY QUADTREE BINNING TO CATALOG

def apply_quadtree_binning(catalog, max_depth=4, min_events=10, merge_threshold=None):
    """
    Apply quadtree binning to catalog and return filtered catalog, region, and bounds.
    
    Args:
        catalog: Earthquake catalog (CSEP catalog)
        max_depth (int): Maximum tree depth
        min_events (int): Minimum events per bin
        merge_threshold (int, optional): Threshold for merging low-count bins
        
    Returns:
        tuple: (filtered_catalog, region, bounds)
    """
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    min_lon, max_lon = np.min(lons), np.max(lons)
    min_lat, max_lat = np.min(lats), np.max(lats)

    root = QuadtreeNode((min_lon, max_lon, min_lat, max_lat))
    build_quadtree(root, lons, lats, max_depth, min_events)

    origins = np.array(extract_leaf_origins(root))
    bounds = extract_leaf_bounds(root)
    
    # Apply bin merging if threshold is specified
    if merge_threshold is not None:
        print(f"\nBefore merging: {len(bounds)} bins")
        
        # First clean any duplicate bins
        bounds = clean_duplicate_bins(bounds, tolerance=0.01)
        print(f"After cleaning duplicates: {len(bounds)} bins")
        
        # Then apply merging
        bounds = merge_low_count_bins_nearest(catalog, bounds, threshold=merge_threshold)
        print(f"After merging: {len(bounds)} bins")
        
        # Debug output
        print("\nFinal Bin Statistics:")
        for i, b in enumerate(bounds):
            count = np.sum((lons >= b[0]) & (lons < b[1]) & (lats >= b[2]) & (lats < b[3]))
            print(f"Bin {i}: {b[1] - b[0]:.2f}° lon x {b[3] - b[2]:.2f}° lat, {count} events")

    # Create mask for events in leaf cells
    mask = np.full(len(lons), False)
    for (min_lon, max_lon, min_lat, max_lat) in bounds:
        in_lon = (lons >= min_lon) & (lons < max_lon)
        in_lat = (lats >= min_lat) & (lats < max_lat)
        mask |= (in_lon & in_lat)

    #  CREATE FILTERED DF
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

    region = CartesianGrid2D.from_origins(origins, dh=1.0)
    filtered_catalog = CSEPCatalog.from_dataframe(df, region=region)

    return filtered_catalog, region, bounds


def plot_bin_counts(ax, catalog, bounds):
    """Add event count labels to each bin on the plot."""
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    for b in bounds:
        count = np.sum((lons >= b[0]) & (lons < b[1]) &
                       (lats >= b[2]) & (lats < b[3]))
        center_x = (b[0] + b[1]) / 2
        center_y = (b[2] + b[3]) / 2
        ax.text(center_x, center_y, str(count),
                ha='center', va='center',
                fontsize=8, color='black', weight='bold')


def plot_quadtree_grid(filtered_catalog, bounds, original_catalog, merge_threshold=None, show_bin_counts=False):
    """
    Plot quadtree grid overlaid on original and filtered events.
    
    Args:
        filtered_catalog: Filtered earthquake catalog
        bounds (list): List of bin bounds
        original_catalog: Original earthquake catalog
        merge_threshold (int, optional): Threshold used for merging (for title)
        show_bin_counts (bool): Whether to show event counts in each bin
    """
    print(f"Plotting {len(bounds)} bins")
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set map extent
    ax.set_extent([116.3, 133.0, 2.0, 22.0], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)

    # Plot events
    ax.scatter(filtered_catalog.get_longitudes(), filtered_catalog.get_latitudes(),
               s=filtered_catalog.get_magnitudes()**2, color='red', alpha=0.6,
               label='Filtered Earthquakes')

    ax.scatter(original_catalog.get_longitudes(), original_catalog.get_latitudes(),
               s=10, color='blue', alpha=0.3, label='All Events')

    # Draw cell outlines
    rect_count = 0
    for (min_lon, max_lon, min_lat, max_lat) in bounds:
        rect = plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                             edgecolor='black', facecolor='none', linewidth=1,
                             transform=ccrs.PlateCarree())
        ax.add_patch(rect)
        rect_count += 1
    
    print(f"Drew {rect_count} rectangles")

    # Add bin counts if requested
    if show_bin_counts:
        plot_bin_counts(ax, original_catalog, bounds)

    # Set title
    if merge_threshold is not None:
        title = f"Quadtree Grid (Merged bins < {merge_threshold} events)"
    else:
        title = "Quadtree Grid with Filtered Earthquakes"
    
    ax.set_title(title)
    ax.legend()
    plt.show()


# =============================================================================
# VALIDATION FUNCTIONS FOR PARAMETER OPTIMIZATION
# =============================================================================

def generate_uniform_bins(df, step=0.5):
    """Generate uniform grid bins for baseline comparison."""
    min_lon, max_lon = df["longitude"].min(), df["longitude"].max()
    min_lat, max_lat = df["latitude"].min(), df["latitude"].max()

    lon_bins = np.arange(min_lon, max_lon + step, step)
    lat_bins = np.arange(min_lat, max_lat + step, step)

    bins = []
    for i in range(len(lon_bins) - 1):
        for j in range(len(lat_bins) - 1):
            sub = df[
                (df["longitude"] >= lon_bins[i]) & (df["longitude"] < lon_bins[i + 1]) &
                (df["latitude"] >= lat_bins[j]) & (df["latitude"] < lat_bins[j + 1])
            ]
            if len(sub) > 0:
                bins.append(sub["magnitude"])
    return bins


def magnitude_entropy(mags, bin_size=0.1):
    """Calculate magnitude entropy for a set of magnitudes."""
    if len(mags) < 2:
        return 0
    hist, _ = np.histogram(mags, bins=np.arange(mags.min(), mags.max() + bin_size, bin_size))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def validate_quadtree_parameters(catalog, param_grid):
    """
    Validate different quadtree parameter combinations to find optimal settings.
    
    Args:
        catalog: Earthquake catalog (pandas DataFrame)
        param_grid: List of parameter dictionaries with 'min_events' and 'max_depth'
        
    Returns:
        DataFrame: Results with variance reduction and information gain metrics
    """
    # Calculate baseline metrics
    total_var = catalog["magnitude"].var()
    uniform_entropy = sum(magnitude_entropy(mags) for mags in generate_uniform_bins(catalog))
    
    print(f"Baseline metrics:")
    print(f"  Total magnitude variance: {total_var:.4f}")
    print(f"  Uniform grid entropy: {uniform_entropy:.4f}")
    print()
    
    results = []
    
    for params in param_grid:
        label = params["label"]
        min_events = params["min_events"]
        max_depth = params["max_depth"]
        
        print(f"Testing {label} (min_events={min_events}, max_depth={max_depth})...")
        
        # Apply quadtree
        filtered, root, bounds = apply_quadtree(
            catalog, min_events=min_events, max_depth=max_depth, min_bin_width=0.2
        )
        
        # Calculate statistics for each bin
        entropy_list = []
        var_list = []
        mmax_list = []
        count_list = []

        for (min_lon, max_lon, min_lat, max_lat) in bounds:
            sub = catalog[
                (catalog["longitude"] >= min_lon) & (catalog["longitude"] < max_lon) &
                (catalog["latitude"] >= min_lat) & (catalog["latitude"] < max_lat)
            ]
            mags = sub["magnitude"]
            count = len(mags)
            
            if count > 0:
                mmax = mags.max()
                var = mags.var() if count > 1 else 0
                ent = magnitude_entropy(mags)
                
                mmax_list.append(mmax)
                var_list.append(var)
                entropy_list.append(ent)
                count_list.append(count)

        # Calculate aggregate metrics
        avg_var = np.mean(var_list) if var_list else 0
        variance_reduction = (total_var - avg_var) / total_var if total_var > 0 else 0
        qt_entropy = sum(entropy_list)
        info_gain = uniform_entropy - qt_entropy

        # Store results
        result = {
            "config": label,
            "min_events": min_events,
            "max_depth": max_depth,
            "bins": len(bounds),
            "events": sum(count_list),
            "avg_mmax": np.mean(mmax_list) if mmax_list else 0,
            "variance_reduction": variance_reduction,
            "info_gain": info_gain,
            "avg_bin_variance": avg_var
        }
        
        results.append(result)
        
        print(f"  ✅ {len(bounds)} bins, VR: {variance_reduction:.4f}, IG: {info_gain:.4f}")
    
    print("\n✅ Validation completed!")
    return pd.DataFrame(results)
