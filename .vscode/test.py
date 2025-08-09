import sys
sys.path.append('./')

from src.preprocessing.load_catalog import load_catalog
from src.binning.quadtree_practice import apply_quadtree_binning
from src.binning.quadtree_practice import plot_quadtree_grid
# --- Load Catalog ---
_, original_catalog = load_catalog("data/eq_catalog.csv")

# Apply with aggressive merging (higher threshold)
filtered_catalog, region, bounds = apply_quadtree_binning(
    original_catalog,
    max_depth=3,
    min_events=100,
    merge_threshold=50  # Will merge all bins with <50 events
)



plot_quadtree_grid(filtered_catalog, bounds, original_catalog)