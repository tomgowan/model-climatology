import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import pygrib, os, sys, glob
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta




wmont = [-117.0, 43.0, -108.5, 49.0]
utah = [-114.7, 36.7, -108.9, 42.5]
colorado = [-110.0, 36.0, -104.0, 41.9]
wasatch = [-113.4, 39.5, -110.7, 41.9]
cascades = [-125.3, 42.0, -116.5, 49.1]
west = [-125.3, 31.0, -102.5, 49.2]


region = sys.argv[1]



if region == 'wmont':
    latlon = wmont
    
if region == 'utah':
    latlon = utah
    
if region == 'colorado':
    latlon = colorado
    
if region == 'wasatch':
    latlon = wasatch
    
if region == 'cascades':
    latlon = cascades

if region == 'west':
    latlon = west
    



num_days = 1
preciptotal = zeros((785,650))

for j in range(12,37):
    for NCARens_file in glob.glob('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/ncarens/ncar_3km_westus_20[0-9]5[0-9][0-9][0-9][0-9]00_mem1_f0%02d.nc' %  j):
        num_days = num_days + 1
        
        print(NCARens_file) 

        fh = Dataset(NCARens_file, mode='r')
        
        lon = fh.variables['gridlon_0'][:]
        lat = fh.variables['gridlat_0'][:]
            
            
        if j == 1:
            precipI = fh.variables['APCP_P8_L1_GLC0_acc'][:]
                
        else: 
            precipI = fh.variables['APCP_P8_L1_GLC0_acc1h'][:]
                    
        precip = precipI*0.0393689*25.4
        preciptotal = preciptotal+precip
        

fh.close()


num_days = num_days/24

precip_per_day = preciptotal/num_days

################################################11111111111111111111

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(lon, lat)
levels = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 14, 18, 24, 1000]
csAVG = map.contourf(xi,yi,precip_per_day,levels, colors=('w', 'lightgrey', 'dimgray','palegreen',
                                             'limegreen', 'g','blue', 
                                            'royalblue', 'lightskyblue', 'cyan', 'navajowhite', 
                                            'darkorange', 'orangered', 'sienna', 'maroon'))
map.drawcoastlines()
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks=[0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 14, 18, 24])
cbar.ax.set_xticklabels(['0', '0.025', '0.5', '0.75', '1', '1.5', '2', '3', '4', '6', '8', '10', '14', '18', '24+'])
cbar.ax.set_xlabel('mm', fontsize = 14)
ax.set_title('Mean Daily Precip from NCAR Ensemble Control Run', fontsize = 18)


plt.savefig("ncar_ensemble_control_daily_precip.pdf")
plt.show()








