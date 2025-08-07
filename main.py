import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
sys.path.append('./')
from src.preprocessing.load_catalog import load_catalog
from src.models.LSTM_algo import train_and_evaluate_lstm
from src.models.attention import run_attention_pipeline
from src.preprocessing.load_catalog import load_catalog
from src.binning.quadtree import apply_quadtree_binning, plot_quadtree_grid

# --- Load Catalog ---
_, original_catalog = load_catalog("data/eq_catalog.csv")

# --- Apply Quadtree Binning ---
filtered_catalog, region, bounds = apply_quadtree_binning(
    original_catalog,
    max_depth=3,
    min_events=15
)

# --- Plot Result ---
plot_quadtree_grid(filtered_catalog, bounds, original_catalog)

# --- LSTM Model ---
train_and_evaluate_lstm()

# --- Attention ---
run_attention_pipeline()