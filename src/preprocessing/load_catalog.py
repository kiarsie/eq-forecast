import pandas as pd
from csep.core.catalogs import CSEPCatalog
from csep.utils.time_utils import datetime_to_utc_epoch

def load_catalog(filepath):
  
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # COMBINE TO DATE_TIME & DROP ORIGINAL COLS
    df['Date_Time'] = pd.to_datetime(
        df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']], 
        errors='coerce'
    )
    df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'])

    #  RENAME FOR CSEP COMPATIBILITY
    df = df.rename(columns={
        'N_Lat': 'latitude',
        'E_Long': 'longitude',
        'Mag': 'magnitude',
        'Depth': 'depth'
    })

    #  DROP ROW WITH MISSING DATE_TIME
    df = df[df['Date_Time'].notnull()]

    #  LOCALIZE TO MANILA AND CONVERT TO UTF
    df['Date_Time'] = df['Date_Time'].dt.tz_localize(
        'Asia/Manila', ambiguous='NaT', nonexistent='NaT'
    ).dt.tz_convert('UTC')

    #  CONVERT UTC TO EPOCH TIME
    df['origin_time'] = df['Date_Time'].apply(datetime_to_utc_epoch)

    #  ADD ID COL
    df = df.reset_index().rename(columns={'index': 'id'})

    #  CREATE CATALOG 
    catalog = CSEPCatalog.from_dataframe(df)

    return df, catalog