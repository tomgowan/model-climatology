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
    

nearest_hrrr = zeros((1033,842))
totalprecip = zeros((785,650))
row_hrrr = zeros((4))
col_hrrr = zeros((4))





#######

# Look at peters emails for regridding the HRRR.  dont forget to moudule load wgrib2 before regridding




###########


####### Read in HRRR file ###################################################



#os.system('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr')
num_days = 1



  


grlatlon = pygrib.open('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/regridded.2015112312F01hrrr.grib2')

glatlon = grlatlon.select(name='Total Precipitation')[1]
hrrr_precip_test = glatlon.values
hrrr_lat,hrrr_lon = glatlon.latlons()


for j in range(0,1):  #13
    for HRRR_file in glob.glob('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/regridded.201511[0-9][0-9]12F01hrrr.grib2'):

        num_days = num_days + 1
        grbs = pygrib.open(HRRR_file)    
        
        grb = grbs.select(name='Total Precipitation')[1]
        hrrr_precip = grb.values*0.0393689*25.4
        totalprecip = totalprecip + hrrr_precip
        print totalprecip


'''
for j in range(0,2):  #13
    for HRRR_file in glob.glob('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/[0-9][0-9][0-9]5[0-9][0-9][0-9][0-9]00F%02d' % j + 'hrrr.grib2'):
        num_days = num_days + 1
        grbs = pygrib.open(HRRR_file)    
        grb = grbs.select(name='Total Precipitation')[1]
        hrrr_precip = grb.values*0.0393689*25.4
        totalprecip = totalprecip + hrrr_precip
        print HRRR_file

'''





num_days = num_days/24

precip_per_day_hrrr = totalprecip/num_days







##########  NCAR







num_days = 1
preciptotal = zeros((785,650))

for j in range(12,25):
    for NCARens_file in glob.glob('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/ncarens/ncar_3km_westus_201511[0-9][0-9]00_mem1_f0%02d.nc' %  j):
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

precip_per_day_ncar = preciptotal/num_days





fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(hrrr_lon, hrrr_lat)
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
cbar.ax.set_xlabel('mm', fontsize  = 14)
ax.set_title('Mean Daily Precip from HRRR', fontsize = 18)


plt.savefig("hrrr_daily_precip.pdf")
plt.show()
