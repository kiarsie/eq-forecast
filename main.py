from src.preprocessing.load_catalog import load_catalog
from src.binning.cartesian_grid import bin_cartesian

#  LOAD CATALOG CSV
df, catalog = load_catalog("data/eq_catalog.csv")

#  PRINT BASIC INFO
print(df.head())
print(catalog)

#  APPLY 3×3 CARTESIAN BINNING
filtered_catalog = bin_cartesian(catalog)
