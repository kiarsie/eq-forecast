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
#NEW FUNCTION
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
def apply_quadtree_binning(catalog, max_depth=4, min_events=10):
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    min_lon, max_lon = np.min(lons), np.max(lons)
    min_lat, max_lat = np.min(lats), np.max(lats)

    root = QuadtreeNode((min_lon, max_lon, min_lat, max_lat))
    build_quadtree(root, lons, lats, max_depth, min_events)

    origins = np.array(extract_leaf_origins(root))
    bounds = extract_leaf_bounds(root)

    #  MASK EVENTS IN LEAF CELLS
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

#  PLOTTING FUNCTION
def plot_quadtree_grid(filtered_catalog, bounds, original_catalog):
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    min_lon, max_lon = 116.3, 133.0
    min_lat, max_lat = 2.0, 22.0
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    #  MAP FEATURES
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)

    #  PLOT EVENTS
    ax.scatter(
        filtered_catalog.get_longitudes(),
        filtered_catalog.get_latitudes(),
        s=filtered_catalog.get_magnitudes() ** 2,
        color='red',
        alpha=0.6,
        label='Filtered Earthquakes'
    )

    ax.scatter(
        original_catalog.get_longitudes(),
        original_catalog.get_latitudes(),
        s=10,
        color='blue',
        alpha=0.3,
        label='All Events'
    )

    #  DRAW CELL OUTLINES
    for (min_lon, max_lon, min_lat, max_lat) in bounds:
        rect = plt.Rectangle(
            (min_lon, min_lat),
            width=max_lon - min_lon,
            height=max_lat - min_lat,
            edgecolor='black',
            facecolor='none',
            linewidth=1,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)

    ax.set_title("Quadtree Grid with Filtered Earthquakes")
    ax.legend()
    plt.show()
