import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import nclcmaps
import smoother_leah
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from wrf import *
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap, maskoceans
import scipy.signal
import h5py
import pygrib

#%%
###############################################################################
#################### Read in random model data ################################
###############################################################################


#NCAR ENS
grbs = pygrib.open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_raw_output/2016_17_coolseason/ncarens/fill_in_days/mem_5/ncar_3km_2016121800_mem5_f047.grb2")
grb = grbs.select(name='Total Precipitation')[0]
lats_ncar, lons_ncar = grb.latlons()
row = len(lats_ncar[:,0])
col = len(lats_ncar[0,:])
ncar_domain = np.zeros((row, col))
ncar_domain[1:row-1,1:col-1] = ncar_domain[1:row-1,1:col-1] + 10


#HRRR
grbs = pygrib.open("/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/model_data/hrrr/hrrr.t00z.wrfprsf04.grib2")
grb = grbs.select(name='Total Precipitation')[0]
lats_hrrr, lons_hrrr = grb.latlons()
row = len(lats_hrrr[:,0])
col = len(lats_hrrr[0,:])
hrrr_domain = np.zeros((row, col))
hrrr_domain[1:row-1,1:col-1] = hrrr_domain[1:row-1,1:col-1] + 10



#NAM3km
grbs = pygrib.open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_raw_output/2016_17_coolseason/nam3km/dec2016/conusnestx_2016122400_015_018")
grb = grbs.select(name='Total Precipitation')[0]
lats_nam3km, lons_nam3km = grb.latlons()
row = len(lats_nam3km[:,0])
col = len(lats_nam3km[0,:])
nam3km_domain = np.zeros((row, col))
nam3km_domain[1:row-1,1:col-1] = nam3km_domain[1:row-1,1:col-1] + 10


#SREF
grbs = pygrib.open("/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/model_data/sref/sref_arw.t09z.pgrb132.ctl.f12.grib2")
grb = grbs.select(name='Total Precipitation')[0]
lats_sref, lons_sref = grb.latlons()
row = len(lats_sref[:,0])
col = len(lats_sref[0,:])
sref_domain = np.zeros((row, col))
sref_domain[1:row-1,1:col-1] = sref_domain[1:row-1,1:col-1] + 10



###### Elevation Data ######
el_file = '/uufs/chpc.utah.edu/common/home/horel-group/archive/terrain/WesternUS_terrain.nc'
fh = Dataset(el_file, mode='r')

elevation = fh.variables['elevation'][:]
lat = fh.variables['latitude'][:]
lon = fh.variables['longitude'][:]

lat_netcdf = zeros((3600,3600))
long_netcdf = zeros((3600,3600))
for i in range(3600):
    lat_netcdf[:,i] = lat
    long_netcdf[i,:] = lon




#%%

#############################   PLOT  #########################################




#latlon = [-145.3, 0, -90.5, 85]
latlon = [-137.3, 17.0, -57.5, 57.2]

fig1=plt.figure(num=None, figsize=(28,12.25),  facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.04, bottom=0.15, right=0.9, top=0.85, wspace=0, hspace=0)
levels_el = np.arange(0,5000,100)
ax = fig1.add_subplot(121)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],area_thresh=2000., resolution='i')

#Lats and Lons
x_ncar, y_ncar = map(lons_ncar, lats_ncar)
x_hrrr, y_hrrr = map(lons_hrrr, lats_hrrr)
x_nam3km, y_nam3km = map(lons_nam3km, lats_nam3km)
x_sref, y_sref = map(lons_sref, lats_sref)






#Plot domians
levels = [0]
#csAVG2 = map.contour(x_sref,y_sref,sref_domain, levels, origin='lower', linewidths = 5, colors = 'gold')
csAVG2 = map.contour(x_ncar,y_ncar,ncar_domain, levels, origin='lower', linewidths = 5, colors = 'blue')
csAVG2 = map.contour(x_hrrr,y_hrrr,hrrr_domain, levels, origin='lower', linewidths = 5, colors = 'red')
csAVG2 = map.contour(x_nam3km,y_nam3km,nam3km_domain, levels, origin='lower', linewidths = 5, colors = 'c')
csAVG2 = map.contour(x_hrrr,y_hrrr,hrrr_domain, levels, origin='lower', linewidths = 5, colors = 'red')



#ax.set_cscale('log')
map.drawcoastlines() 
map.drawstates()
map.drawcountries()
map.fillcontinents(color='mediumseagreen',lake_color='lightskyblue')
map.drawmapboundary(fill_color='lightskyblue')       
#cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks = levels_ticks)
#cbar.ax.set_xlabel('Mean-Daily Precipitation (mm)', fontsize = 24, labelpad = 15)
#cbar.ax.set_xticklabels(levels_ticks, fontsize = 20) 
props = dict(boxstyle='square', facecolor='white', alpha=1)
ax.text(250000, 5300000, '(a) Model Forecast Domains', fontsize = 35, bbox = props)



ax = fig1.add_subplot(122)
#Colormap
#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['topo_15lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap = nclcmaps.make_cmap(colors, bit=True)
latlon = [-125.3, 31.0, -102.5, 49.2]
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],area_thresh=2000.,resolution='i')
map.drawcoastlines() 
map.drawstates()
map.drawcountries()


#Plot elevaiton data
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
levels_el = np.arange(-200,4600.1,200)
map.drawlsmask(ocean_color='lightskyblue', zorder = 0)
elevation_mask = maskoceans(long_netcdf, lat_netcdf, elevation, inlands = False, resolution = 'f', grid = 10)
csAVG2 = map.contourf(x2,y2,elevation_mask[:,:], levels_el, vmin = -500, cmap = cmap, zorder = 1)#, extend = "min")
map.drawcoastlines() 
map.drawstates()
map.drawcountries()

props = dict(boxstyle='square', facecolor='white', alpha=1)
ax.text(1270000, 2500000, '(b) Mountain Ranges', fontsize = 25, bbox = props)

#Colorbar
levels_ticks = np.arange(-200,4600.1,600)
cbaxes = fig1.add_axes([0.54, 0.061, 0.29, 0.057])             
cbar = plt.colorbar(csAVG2, cax = cbaxes, ticks = levels_ticks, orientation='horizontal')
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_xlabel('Elevation (m)', fontsize = 20)
#ax.text(2600000, 64000, 'Elevation (m)', fontsize = 20)
#map.drawlsmask(ocean_color='lightskyblue', zorder = 1)
#map.fillcontinents(color='mediumseagreen',lake_color='lightskyblue')
#map.drawmapboundary(fill_color='lightskyblue') 


#plt.tight_layout()
plt.savefig("../../../public_html/ms_thesis_plots/model_domains.png")
plt.close(fig1)




