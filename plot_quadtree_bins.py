#!/usr/bin/env python3
"""
Standalone script to plot quadtree bins with better navigation.
Run this after preprocessing to see your 8 bins clearly.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import numpy as np
import seaborn as sns

def plot_quadtree_bins_standalone():
    """Plot quadtree bins with better navigation and display."""
    
    try:
        # Load the processed data
        data_path = "data/processed_earthquake_catalog.csv"
        processed_catalog = pd.read_csv(data_path)
        
        print(f"Loaded data with {len(processed_catalog)} earthquakes")
        print(f"Unique bin IDs: {sorted(processed_catalog['bin_id'].unique())}")
        
        # Set up the plot
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with better size
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Philippines bounding box (approximate)
        ph_bounds = {
            'lon_min': 116.0, 'lon_max': 127.0,
            'lat_min': 4.0, 'lat_max': 21.5
        }
        
        # Set map bounds
        ax.set_xlim(ph_bounds['lon_min'], ph_bounds['lon_max'])
        ax.set_ylim(ph_bounds['lat_min'], ph_bounds['lat_max'])
        
        # Add Philippines outline (simplified rectangle)
        ph_outline = patches.Rectangle(
            (ph_bounds['lon_min'], ph_bounds['lat_min']),
            ph_bounds['lon_max'] - ph_bounds['lon_min'],
            ph_bounds['lat_max'] - ph_bounds['lat_min'],
            linewidth=2, edgecolor='darkblue', facecolor='lightblue', alpha=0.2
        )
        ax.add_patch(ph_outline)
        
        # Get unique bins and their statistics
        bin_stats = processed_catalog.groupby('bin_id').agg({
            'latitude': ['min', 'max'],
            'longitude': ['min', 'max'],
            'magnitude': 'count'
        }).round(3)
        
        # Flatten column names
        bin_stats.columns = ['_'.join(col).strip() for col in bin_stats.columns]
        bin_stats = bin_stats.reset_index()
        
        print(f"\nBin statistics:")
        print(bin_stats)
        
        # Create color map for earthquake counts
        earthquake_counts = bin_stats['magnitude_count'].values
        norm = Normalize(vmin=earthquake_counts.min(), vmax=earthquake_counts.max())
        cmap = plt.cm.viridis
        
        # Plot each bin with better visibility
        for _, bin_row in bin_stats.iterrows():
            bin_id = bin_row['bin_id']
            
            # Get bin boundaries
            lat_min, lat_max = bin_row['latitude_min'], bin_row['latitude_max']
            lon_min, lon_max = bin_row['longitude_min'], bin_row['longitude_max']
            
            # Create rectangle for this bin
            width = lon_max - lon_min
            height = lat_max - lat_min
            
            # Color based on earthquake count
            color = cmap(norm(bin_row['magnitude_count']))
            
            # Create bin rectangle with better visibility
            bin_rect = patches.Rectangle(
                (lon_min, lat_min), width, height,
                linewidth=2, edgecolor='black', facecolor=color, alpha=0.8
            )
            ax.add_patch(bin_rect)
            
            # Add bin ID label with better visibility
            center_lon = lon_min + width/2
            center_lat = lat_min + height/2
            ax.text(center_lon, center_lat, f'Bin {bin_id}', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color='white', bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))
            
            # Add earthquake count with better visibility
            ax.text(center_lon, center_lat - height/4, f'{int(bin_row["magnitude_count"])} eq',
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white', bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Earthquake Count', fontsize=14, fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Longitude (°E)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold')
        ax.set_title('Quadtree Bins for Earthquake Forecasting - Philippines\n(8 Bins)', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.4, linewidth=0.8)
        
        # Add statistics box
        total_bins = len(bin_stats)
        total_eq = earthquake_counts.sum()
        ax.text(0.02, 0.98, f'Total Bins: {total_bins}\nTotal Earthquakes: {total_eq:,}', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        
        # Add Philippines label
        ax.text(121.5, 20.5, 'PHILIPPINES', fontsize=16, fontweight='bold',
               ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
        
        # Tight layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
        # Also save the plot
        save_path = "data/quadtree_bins_clear_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nPlot saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_quadtree_bins_standalone()
