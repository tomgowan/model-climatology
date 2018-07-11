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
from datetime import date, timedelta
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
    
    
    
    


inhouse_data = zeros((625,187))
ncar_data = zeros((798,187))
bias = zeros((800,10))





###############################################################################
############ Read in SNOTEL (inhouse (12Z to 12Z)) data   #####################
###############################################################################
             
x = 0
q = 0
v = 0
i = 0              
with open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/snotel/Tom_in_house.csv", "rt") as f:
    for line in f:
        commas = line.count(',') + 1
        year = line.split(',')[0]
        month = line.split(',')[1]
        day = line.split(',')[2]
        date = year + month + day
        y = 0
      

        if i == 0:     
            for t in range(3,commas):
                station_id_inhouse = line.split(',')[t]
                inhouse_data[x,0] = station_id_inhouse
                x = x + 1
                    
        if i == 1:     
            for t in range(3,commas):
                lat_inhouse = line.split(',')[t]
                inhouse_data[q,1] = lat_inhouse
                q = q + 1
            
        if i == 2:     
            for t in range(3,commas):
                lon_inhouse = line.split(',')[t]
                inhouse_data[v,2] = lon_inhouse
                v = v + 1

        if i != 0 and i != 1 and i != 2:
            for t in range(3,commas):   
                inhouse_precip = line.split(',')[t]
                if inhouse_precip != "NaN":
                    inhouse_data[y,186] = inhouse_data[y,186] + 1
                if inhouse_precip == "NaN":
                    inhouse_precip = 9999/25.4
                inhouse_precip = float(inhouse_precip)
                inhouse_data[y,i] = inhouse_precip*25.4

                y = y + 1
            
        i = i + 1
        
        
        
###############################################################################
############ Read in NCARENS (12Z to 12Z) data   ##############################
###############################################################################

             
x = 0
q = 0
v = 0
i = 0              
with open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/ncarens_precip_12Zto12Z.txt", "rt") as f:
    for line in f:
        commas = line.count(',') + 1
        year = line.split(',')[0]
        month = line.split(',')[1]
        day = line.split(',')[2]
        date = year + month + day
        y = 0
      

        if i == 0:     
            for t in range(3,commas):
                station_id_ncar = line.split(',')[t]
                ncar_data[x,0] = station_id_ncar
                x = x + 1
                    
        if i == 1:     
            for t in range(3,commas):
                lat_ncar = line.split(',')[t]
                ncar_data[q,1] = lat_ncar
                q = q + 1
            
        if i == 2:     
            for t in range(3,commas):
                lon_ncar = line.split(',')[t]
                ncar_data[v,2] = lon_ncar
                v = v + 1

        if i != 0 and i != 1 and i != 2:
            for t in range(3,commas):   
                ncar_precip = line.split(',')[t]
                if ncar_precip != "NaN":
                    ncar_data[y,186] = ncar_data[y,186] + 1
                if ncar_precip == "NaN":
                    ncar_precip = 9999
                ncar_precip = float(ncar_precip)
                ncar_data[y,i] = ncar_precip

                y = y + 1
            
        i = i + 1




###############################################################################
##### Create bias array (only use days with good data (<1000) for both ########
###############################################################################

        
w = -1
for x in range(len(ncar_data[:,0])):
    for y in range(len(inhouse_data[:,0])):
            if ncar_data[x,0] == inhouse_data[y,0]:
                w = w + 1
                bias[w,0] = ncar_data[x,0]
                bias[w,1] = inhouse_data[y,0]
                

                for z in range(3,185):
                    if ncar_data[x,z] < 1000 and inhouse_data[y,z] < 1000:
                        
                        #lat/lon data
                        bias[w,2] = ncar_data[x,1]
                        bias[w,3] = ncar_data[x,2]
                        bias[w,4] = inhouse_data[y,1]
                        bias[w,5] = inhouse_data[y,2]   
                        
                        #precip data
                        bias[w,6] = bias[w,6] + ncar_data[x,z]
                        bias[w,7] = bias[w,7] + inhouse_data[y,z]
                        bias[w,8] = bias[w,6]/bias[w,7]

                       
                    else:
                        print ncar_data[x,z]
                        print inhouse_data[y,z]
                        
                        
###############################################################################
################# Regional Plot (NCAR/inhouse snotel) #########################
###############################################################################                        
        

                
fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(bias[0:625,3], bias[0:625,2])
levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 2.2]
cmap=plt.cm.BrBG
csAVG = map.scatter(xi,yi, c = bias[0:625,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 125, vmin = 0.4, vmax = 2.2)
map.drawcoastlines() 
map.drawstates()
map.drawcountries()
ax.set_title('NCAR Ensemble Daily Precipitation Bias at SNOTEL Sites (OCT 2015 - MAR 2016)', fontsize = 18)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,2.2])
cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
cbar.ax.set_xlabel('Daily (12Z to 12Z) Precipitation Bias (NCAR Ens/SNOTEL)', fontsize  = 14)
plt.savefig("./plots/ncarens_inhouse_regionalmap_12Z.pdf")
plt.show()
  

                    

        
        