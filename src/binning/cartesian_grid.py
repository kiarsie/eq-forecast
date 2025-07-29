import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.core.regions import CartesianGrid2D

def bin_cartesian(catalog):
    #  SET BOUNDS AND RESOLUTION
    min_lon, max_lon = 116.3, 133.0
    min_lat, max_lat = 2.0, 22.0
    dh_lon = (max_lon - min_lon) / 3  # ~5.567°
    dh_lat = (max_lat - min_lat) / 3  # ~6.667°

    #  CREATING GRID CELL CENTERS
    lon_edges = np.arange(min_lon, max_lon, dh_lon)
    lat_edges = np.arange(min_lat, max_lat, dh_lat)
    lon_centers = lon_edges + dh_lon / 2
    lat_centers = lat_edges + dh_lat / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    origins = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel()))

    #  LOADING CATALOG COORDINATES
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()

    #  MANUAL MASKING PER CELL
    mask = np.full(len(lons), False)
    for origin in origins:
        in_lon = (lons >= origin[0] - dh_lon / 2) & (lons < origin[0] + dh_lon / 2)
        in_lat = (lats >= origin[1] - dh_lat / 2) & (lats < origin[1] + dh_lat / 2)
        mask |= (in_lon & in_lat)

    #  FILTERING AND CREATING NEW DF
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

    #  DUMMY REGION TO SATISFY CSEP REQUIREMENT
    dummy_region = CartesianGrid2D.from_origins(
        origins=np.array([[min_lon, min_lat]]),
        dh=1.0
    )

    #  WRAP INTO NEW CSEP CATALOG
    filtered_catalog = CSEPCatalog.from_dataframe(df, region=dummy_region)

    '''
    if filtered_catalog.get_number_of_events() == 0:
        print("Warning: Filtered catalog is empty. Check your grid bounds or masking logic.")  ---> FOR CHECKING
    '''

    #  PLOTTING
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    #  MAP FEATURES
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)

    #  EQ EVENTS
    ax.scatter(
        filtered_catalog.get_longitudes(),
        filtered_catalog.get_latitudes(),
        s=filtered_catalog.get_magnitudes()**2,
        color='red',
        alpha=0.6,
        label='Earthquakes',
        transform=ccrs.PlateCarree()
    )

    #  GRID CELLS
    for origin in origins:
        rect = plt.Rectangle(
            (origin[0] - dh_lon / 2, origin[1] - dh_lat / 2),
            width=dh_lon,
            height=dh_lat,
            edgecolor='black',
            facecolor='none',
            linewidth=1,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)

    ax.set_title(f"3x3 Grid ({dh_lon:.3f}° × {dh_lat:.3f}°)")
    ax.legend()
    plt.show()

    return filtered_catalog
