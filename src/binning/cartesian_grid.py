import numpy as np
import pandas as pd
from csep.core.catalogs import CSEPCatalog
from csep.core.regions import CartesianGrid2D
from csep.utils.time_utils import datetime_to_utc_epoch

def apply_cartesian(catalog, n_lat=3, n_lon=3):
    # --- Region bounds ---
    min_lon, max_lon = 116.3, 133.0
    min_lat, max_lat = 2.0, 22.0

    # --- Create evenly spaced edges ---
    lon_edges = np.linspace(min_lon, max_lon, n_lon + 1)
    lat_edges = np.linspace(min_lat, max_lat, n_lat + 1)
    dh_lon = lon_edges[1] - lon_edges[0]
    dh_lat = lat_edges[1] - lat_edges[0]

    # --- Compute bin centers ---
    lon_centers = lon_edges[:-1] + dh_lon / 2
    lat_centers = lat_edges[:-1] + dh_lat / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    origins = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel()))

    # --- Filter original catalog ---
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    mask = np.full(len(lons), False)
    for lon_c, lat_c in origins:
        in_lon = (lons >= lon_c - dh_lon / 2) & (lons < lon_c + dh_lon / 2)
        in_lat = (lats >= lat_c - dh_lat / 2) & (lats < lat_c + dh_lat / 2)
        mask |= (in_lon & in_lat)

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

    # Dummy 1-cell region for CSEP compatibility
    dummy_region = CartesianGrid2D.from_origins(origins=np.array([[min_lon, min_lat]]), dh=1.0)
    filtered_catalog = CSEPCatalog.from_dataframe(df, region=dummy_region)

    # Bin boundaries for visualization
    bounds = []
    for lon_min in lon_edges[:-1]:
        for lat_min in lat_edges[:-1]:
            bounds.append((
                lon_min, lon_min + dh_lon,
                lat_min, lat_min + dh_lat
            ))

    return filtered_catalog, dummy_region, bounds
