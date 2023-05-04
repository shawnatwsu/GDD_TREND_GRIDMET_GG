import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

tmin_path = "TMIN DATASET"
tmax_path = "TMAX DATASET"

tmin_ds = xr.open_dataset(tmin_path)
tmax_ds = xr.open_dataset(tmax_path)

lon_min, lon_max = -124.736342, -116.945392
lat_min, lat_max = 45.521208, 49.382808

tmax_subset = tmax_ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))
tmin_subset = tmin_ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))

tmax_subset = tmax_subset['tmax'] - 273.15
tmin_subset = tmin_subset['tmin'] - 273.15

def calc_gdd(tmin, tmax, base_temp=5):
    avg_temp = (tmin + tmax) / 2
    gdd = avg_temp - base_temp
    return np.maximum(gdd, 0)

gdd_subset = calc_gdd(tmin_subset, tmax_subset)

gdd_filtered = gdd_subset.sel(day=slice('1991-01-01', '2020-09-30'))
gdd_filtered = gdd_filtered.where(gdd_filtered['day.month'] >= 1).where(gdd_filtered['day.month'] <= 9)

seasonal_gdd = gdd_filtered.groupby('day.year').sum(dim='day')

p = seasonal_gdd.polyfit(dim='year', deg=1)
trend_values = p.polyfit_coefficients.sel(degree=1)

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.OCEAN)

lons = trend_values['lon']
lats = trend_values['lat']
gdd_data = trend_values.data
cmap = plt.get_cmap('bwr')

vmin, vmax = np.nanmin(gdd_data), np.nanmax(gdd_data)
#levels = np.linspace(vmin, vmax, 20)

gdd_data_masked = np.ma.masked_where(gdd_data == 0, gdd_data)

contour = plt.pcolormesh(lons, lats, gdd_data_masked,cmap=cmap, vmin=-10, vmax=10, transform=ccrs.PlateCarree())
ax.annotate('Data Source: Gridmet', xy=(0.01, 0.03), xycoords='axes fraction', fontsize=10, color='k', backgroundcolor='w')

colorbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=0.5)
colorbar.set_label('Growing Degree Days (GDD) Trend', fontsize=12)

gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = ticker.MaxNLocator(8)
gl.ylocator = ticker.MaxNLocator(6)
gl.xlabel_style = {'size': 10, 'color': 'gray'}
gl.ylabel_style = {'size': 10, 'color': 'gray'}

plt.title('Growing Degree Days (GDD) Trend January-September \n(1991-2020) Washington State', fontsize=14, fontweight='bold', pad=20)
ax.text(0.5, 1.01, 'General Growth', horizontalalignment='center', fontsize=10, transform=ax.transAxes)

plt.savefig('Washington_GDD_Trend_Map_GG.png', dpi=300, bbox_inches='tight')

plt.show()
