import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.core.regions import CartesianGrid2D

class QuadtreeNode:
    """Quadtree node holding bounding box, event indices, and children."""
    def __init__(self, bounds, depth=0):
        self.bounds = bounds  # (min_lon, max_lon, min_lat, max_lat)
        self.depth = depth
        self.children = []
        self.events_idx = []

    def subdivide(self):
        """Split the node into four quadrants."""
        min_lon, max_lon, min_lat, max_lat = self.bounds
        mid_lon = (min_lon + max_lon) / 2
        mid_lat = (min_lat + max_lat) / 2

        return [
            QuadtreeNode((min_lon, mid_lon, min_lat, mid_lat), self.depth + 1),
            QuadtreeNode((mid_lon, max_lon, min_lat, mid_lat), self.depth + 1),
            QuadtreeNode((min_lon, mid_lon, mid_lat, max_lat), self.depth + 1),
            QuadtreeNode((mid_lon, max_lon, mid_lat, max_lat), self.depth + 1)
        ]

def build_quadtree(node, lons, lats, max_depth=3, min_events=100):
    """Recursively build quadtree until min_events or max_depth reached."""
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

def extract_leaf_origins(node):
    """Return cell centers for all leaf nodes."""
    if not node.children:
        min_lon, max_lon, min_lat, max_lat = node.bounds
        return [((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)]
    origins = []
    for child in node.children:
        origins.extend(extract_leaf_origins(child))
    return origins

def extract_leaf_bounds(node):
    """Return bounds of all leaf nodes."""
    if not node.children:
        return [node.bounds]
    bounds = []
    for child in node.children:
        bounds.extend(extract_leaf_bounds(child))
    return bounds

def find_richest_neighbor(bounds_list, target_bounds, bin_counts):
    """Find the adjacent bin (sharing edge or corner) with most events."""
    max_count = -1
    best_neighbor = None
    
    for bounds in bounds_list:
        if bounds == target_bounds:
            continue
        
        # adjacency: share at least a point along one edge
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


def merge_low_count_bins_nearest(catalog, bounds, threshold=50):
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    def count_events(b):
        return np.sum((lons >= b[0]) & (lons < b[1]) &
                      (lats >= b[2]) & (lats < b[3]))

    def bin_center(b):
        return ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2)

    changed = True
    while changed:
        changed = False
        bin_counts = {b: count_events(b) for b in bounds}
        low_bins = [b for b, cnt in bin_counts.items() if cnt < threshold]

        if not low_bins:
            break

        for b in low_bins:
            c_b = bin_center(b)
            # Find closest *other* bin by Euclidean distance
            other_bins = [ob for ob in bounds if ob != b]
            if not other_bins:
                continue
            closest = min(
                other_bins,
                key=lambda ob: np.hypot(c_b[0] - bin_center(ob)[0],
                                        c_b[1] - bin_center(ob)[1])
            )
            # Merge
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

    return bounds

def plot_bin_counts(ax, catalog, bounds):
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

def apply_quadtree_binning(catalog, max_depth=3, min_events=100, merge_threshold=50):
    """Apply quadtree binning and return filtered catalog, region, and bounds."""
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    root = QuadtreeNode((np.min(lons), np.max(lons), np.min(lats), np.max(lats)))
    build_quadtree(root, lons, lats, max_depth, min_events)

    origins = np.array(extract_leaf_origins(root))
    bounds = extract_leaf_bounds(root)
    
    bounds = merge_low_count_bins_nearest(catalog, bounds, threshold=merge_threshold)

    # Debug output
    print("\nFinal Bin Statistics:")
    for i, b in enumerate(bounds):
        count = np.sum((lons >= b[0]) & (lons < b[1]) & (lats >= b[2]) & (lats < b[3]))
        print(f"Bin {i}: {b[1] - b[0]:.2f}° lon x {b[3] - b[2]:.2f}° lat, {count} events")

    mask = np.full(len(lons), False)
    for (min_lon, max_lon, min_lat, max_lat) in bounds:
        mask |= ((lons >= min_lon) & (lons < max_lon) &
                 (lats >= min_lat) & (lats < max_lat))

    df = pd.DataFrame({
        'id': np.arange(np.sum(mask)),
        'magnitude': catalog.get_magnitudes()[mask],
        'latitude': lats[mask],
        'longitude': lons[mask],
        'depth': (catalog.get_depths()[mask] if catalog.get_depths() is not None 
                  else np.full(np.sum(mask), np.nan)),
        'origin_time': [datetime_to_utc_epoch(dt) for dt in np.array(catalog.get_datetimes())[mask]]
    })

    region = CartesianGrid2D.from_origins(origins, dh=1.0)
    filtered_catalog = CSEPCatalog.from_dataframe(df, region=region)

    return filtered_catalog, region, bounds

def plot_quadtree_grid(filtered_catalog, bounds, original_catalog, merge_threshold=50):
    """Plot quadtree grid overlaid on original and filtered events."""
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([116.3, 133.0, 2.0, 22.0], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)

    ax.scatter(filtered_catalog.get_longitudes(), filtered_catalog.get_latitudes(),
               s=filtered_catalog.get_magnitudes()**2, color='red', alpha=0.6,
               label='Filtered Earthquakes')

    ax.scatter(original_catalog.get_longitudes(), original_catalog.get_latitudes(),
               s=10, color='blue', alpha=0.3, label='All Events')

    for (min_lon, max_lon, min_lat, max_lat) in bounds:
        rect = plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                             edgecolor='black', facecolor='none', linewidth=1,
                             transform=ccrs.PlateCarree())
        ax.add_patch(rect)

    # ➕ Add this line to display event count per bin
    plot_bin_counts(ax, original_catalog, bounds)

    ax.set_title(f"Quadtree Grid (Merged bins < {merge_threshold} events)")
    ax.legend()
    plt.show()

