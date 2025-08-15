# quadtree_validation.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import numpy as np
import pandas as pd
from pathlib import Path
from src.binning.quadtree import apply_quadtree

# --- Load Earthquake Catalog ---
catalog_path = Path("data/eq_catalog.csv")
catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
catalog = catalog.rename(columns={"N_Lat": "latitude", "E_Long": "longitude", "Mag": "magnitude"})
catalog = catalog.dropna(subset=["latitude", "longitude", "magnitude"])

# --- Uniform Grid for Entropy Baseline ---
def generate_uniform_bins(df, step=0.5):
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

# --- Magnitude Entropy ---
def magnitude_entropy(mags, bin_size=0.1):
    if len(mags) < 2:
        return 0
    hist, _ = np.histogram(mags, bins=np.arange(mags.min(), mags.max() + bin_size, bin_size))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# --- Configurations to Test ---
param_grid = [
    {"label": "Min20_D3", "min_events": 20, "max_depth": 3},
    {"label": "Min20_D4", "min_events": 20, "max_depth": 4},
    {"label": "Min20_D5", "min_events": 20, "max_depth": 5},
    {"label": "Min50_D3", "min_events": 50, "max_depth": 3},
    {"label": "Min50_D4", "min_events": 50, "max_depth": 4},
    {"label": "Min50_D5", "min_events": 50, "max_depth": 5},
    {"label": "Min100_D3", "min_events": 100, "max_depth": 3},
    {"label": "Min100_D4", "min_events": 100, "max_depth": 4},
    {"label": "Min100_D5", "min_events": 100, "max_depth": 5},
]

# --- Total Magnitude Variance ---
total_var = catalog["magnitude"].var()
uniform_entropy = sum(magnitude_entropy(mags) for mags in generate_uniform_bins(catalog))

# --- Run Validations ---
results = []
for p in param_grid:
    label = p["label"]
    filtered, root, bounds = apply_quadtree(
        catalog,
        min_events=p["min_events"],
        max_depth=p["max_depth"],
        min_bin_width=0.2
    )

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

    avg_var = np.mean(var_list)
    vr = (total_var - avg_var) / total_var
    qt_entropy = sum(entropy_list)
    ig = uniform_entropy - qt_entropy

    results.append({
        "config": label,
        "bins": len(bounds),
        "events": sum(count_list),
        "avg_mmax": np.mean(mmax_list),
        "variance_reduction": vr,
        "info_gain": ig
    })

# --- Save Results ---
results_df = pd.DataFrame(results)
results_df.to_csv("quadtree_validation_summary.csv", index=False)
print(results_df)
