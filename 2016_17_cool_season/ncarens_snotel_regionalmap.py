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
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from matplotlib.mlab import bivariate_normal
from matplotlib import colors, ticker, cm


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
    
    
    
    
nearestNCAR = zeros((785,650))
totalprecip = zeros((152,2))
Date = zeros((152))
frequency = zeros((80,2))
frequency_snotel = zeros((80,2))
daily_snotel_precip = zeros((798,20))
snotel_lat2 = zeros((798))
snotel_lon2 = zeros((798))
row = zeros((798))
col = zeros((798))





############ Read in SNOTEL lat and lon  data   ####################################

i = -1
j = 0
k = 0 


with open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/all_nov2015_march2016_english.txt", "rt") as f:
    for line in f:

        #print i
        snotel_lat1 = float(line.split(',')[5])
        snotel_lat1 = round(snotel_lat1, 3)
        
        snotel_lon1 = float(line.split(',')[6])
        snotel_lat1 = round(snotel_lat1,3)
        
        date =  line.split(',')[0]
        
        if date != '2015-12-17' and date != '2016-01-06' and date != '2016-02-03' and date != '2016-03-11': 
        
            if snotel_lat1 != snotel_lat2[j-1]:
                snotel_lat2[j] = snotel_lat1
                j = j + 1
                i = i + 1
            
        
            snotel_precip = float(line.split(',')[8])*25.4
            daily_snotel_precip[i,0] = daily_snotel_precip[i,0]+snotel_precip
            daily_snotel_precip[i,1] = daily_snotel_precip[i,1]+1
            
            
            if snotel_lon1 != snotel_lon2[k-1]:
                snotel_lon2[k] = snotel_lon1
                k = k + 1
                
                #print i

    
    
daily_snotel_precip[:,2] = (daily_snotel_precip[:,0])/daily_snotel_precip[:,1]









### Get NCAR data
i = 0
with open('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/snotel_ncarens_precip.txt', 'rt') as f:
    for line in f:
        daily_snotel_precip[i,3] = daily_snotel_precip[i,3] + float(line.split(',')[5])
        i = i + 1
        if i == 798:
            i = 0
    
    
daily_snotel_precip[:,4] = daily_snotel_precip[:,3]/148

#### Bias is in daily_snotel_precip[i,5]

     
daily_snotel_precip[:,5] = daily_snotel_precip[:,4]/daily_snotel_precip[:,2]
    
        
        
        
        
        
### Get hrrr data
i = 0
with open('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/snotel_hrrr_precip.txt', 'rt') as f:
    for line in f:
        daily_snotel_precip[i,8] = daily_snotel_precip[i,8] + float(line.split(',')[3])
        i = i + 1
        if i == 798:
            i = 0
    
    
daily_snotel_precip[:,9] = daily_snotel_precip[:,8]/152  #number of days






        
### Get nam4k data
i = 0
with open('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/snotel_nam4k_precip.txt', 'rt') as f:
    for line in f:
        daily_snotel_precip[i,11] = daily_snotel_precip[i,11] + float(line.split(',')[2])
        i = i + 1
        if i == 798:
            i = 0
    
    
daily_snotel_precip[:,12] = daily_snotel_precip[:,11]/152  #number of days












#### Bias for each model

daily_snotel_precip[:,10] = daily_snotel_precip[:,9]/daily_snotel_precip[:,2]  #### HRRR Bias
daily_snotel_precip[:,5] = daily_snotel_precip[:,4]/daily_snotel_precip[:,2]   #### NCAR Bias
daily_snotel_precip[:,13] = daily_snotel_precip[:,12]/daily_snotel_precip[:,2] #### NAM4km Bias

    
daily_snotel_precip[:,6] = snotel_lat2    
daily_snotel_precip[:,7] = snotel_lon2


for q in range(793):
    if daily_snotel_precip[q,1] < 145:
        daily_snotel_precip = np.delete(daily_snotel_precip, (q), axis=0)

'''
def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='brown', midcol='white', maxcol='green'):
    """ Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap 
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap

cmap= custom_div_cmap(400, mincol='#5D3809', midcol='w' ,maxcol='Bg')
'''


cmap=plt.cm.BrBG
#cmap.set_over('.4')
#cmap.set_under()
    
    
fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(daily_snotel_precip[:,7], daily_snotel_precip[:,6])
levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 2.2]
csAVG = map.scatter(xi,yi, c = daily_snotel_precip[:,5], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 125, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )
#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

ax.set_title('NCAR Ensemble Daily Precipitation Bias at SNOTEL Sites', fontsize = 18)

cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,2.2])
cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])

cbar.ax.set_xlabel('Daily Precipitation Bias (NCAR Ens/SNOTEL)', fontsize  = 14)


plt.savefig("ncarens_snotel_regionalmap.pdf")
plt.show()




##### HRRR PLOT

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(daily_snotel_precip[:,7], daily_snotel_precip[:,6])
levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 2.2]
csAVG = map.scatter(xi,yi, c = daily_snotel_precip[:,10], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 125, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )
#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

ax.set_title('HRRR Daily Precipitation Bias at SNOTEL Sites', fontsize = 18)

cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,2.2])
cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])

cbar.ax.set_xlabel('Daily Precipitation Bias (HRRR/SNOTEL)', fontsize  = 14)


plt.savefig("hrrr_snotel_regionalmap.pdf")
plt.show()



##### nam-4km PLOT

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(daily_snotel_precip[:,7], daily_snotel_precip[:,6])
levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 2.2]
csAVG = map.scatter(xi,yi, c = daily_snotel_precip[:,13], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 125, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )
#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

ax.set_title('HRRR Daily Precipitation Bias at SNOTEL Sites', fontsize = 18)

cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,2.2])
cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])

cbar.ax.set_xlabel('Daily Precipitation Bias (HRRR/SNOTEL)', fontsize  = 14)


plt.savefig("nam4km_snotel_regionalmap.pdf")
plt.show()





