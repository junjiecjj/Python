

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
import pandas as pd
import pingouin as pg
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
def createFigure(figsize=(12, 8), dpi=300, subplotAdj=None, **kwargs):
    figsize = figsize
    figure = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
    if subplotAdj is not None:
        plt.subplots_adjust(**subplotAdj)
    return figure
apo = xr.open_dataset("apo.nc")['apo0'].values

month = [6, 7, 8]
nino = xr.open_dataset("NINO3.4_index.nc").where(
    lambda m: m.time.dt.month.isin(month), drop=True).groupby(
        "time.year").mean(
            "time")['NINO3.4'].sel(year = slice(1980, 2014)).values

sst = xr.open_dataset("sst.mnmean.nc").where(
    lambda m: m.time.dt.month.isin(month)).groupby(
        "time.year").mean(
        "time")["sst"].sel(year = slice(1980, 2014))

lat, lon = sst.lat.values, sst.lon.values
sst = sst.values
r_matrix, p_matrix = [np.zeros_like(sst[0]) for _ in range(2)]
for i in tqdm(range(len(lat)), desc = 'Calc apo and sst partial corr...'):
    for j in range(len(lon)):
        if np.isnan(sst[:, i, j]).any():
            continue
        else:
            df = pd.DataFrame({
                'apo': apo,
                'Nino3_4': nino,
                'sst': sst[:, i, j]
            })
            pc = pg.partial_corr(data=df, x='sst', y='apo', covar='Nino3_4', method='pearson')
            r_matrix[i, j] = pc['r'][0]  # Partial correlation between IOB and precip
            p_matrix[i, j] = pc['p-val'][0]  # p_value for partial correlation

p_matrix[p_matrix == 0] = np.nan

fig = createFigure(figsize=(12, 8), dpi=300,
             subplotAdj=dict(left=0.04, right=0.98,
                             top=0.9, bottom=0.05,
                             wspace=0.05, hspace=0.1))
ax = plt.subplot(111, projection=ccrs.Robinson(central_longitude=180))
ax.set_global()
ax.coastlines()

CF = plt.contourf(lon, lat,
                  r_matrix,
                  cmap= "RdBu_r",
                  levels=np.linspace(-0.5, 0.5, 21),
                  extend='both',
                  transform=ccrs.PlateCarree())

c1b = ax.contourf(lon, lat,
                  p_matrix,
                  levels=[0,0.05,0.5],
                  zorder=1,
                  hatches=['..', None],
                  colors="none", transform=ccrs.PlateCarree())
plt.title('Partial Correlation between APO and SST',
           fontweight='bold',
           fontsize=20)
position= fig.add_axes([0.31, 0.05,  0.4, 0.025])
fig.colorbar(CF,cax=position,orientation='horizontal',format='%.1f',)
plt.show()
