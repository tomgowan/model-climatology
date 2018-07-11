
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
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from matplotlib.mlab import bivariate_normal
from matplotlib import colors, ticker, cm
from datetime import date, timedelta
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colors, ticker, cm
import matplotlib.lines as mlines
import matplotlib.ticker as mtick



'''

inhouse_data = zeros((798,186))
ncar_data = zeros((798,186))
nam4k_data = zeros((798,186))
nam12k_data = zeros((798,186))
hrrr_data = zeros((798,186))


###############################################################################
############ Read in  12Z to 12Z data   #######################################
###############################################################################
             
x = 0
q = 0
v = 0
i = 0   

links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/snotel/Tom_in_house.csv", 
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/ncarens_precip_12Zto12Z_upperquart_prob.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/ncarens_precip_12Zto12Z_upperdec_prob.txt"]

#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        
data = zeros((3,798,186))

         
for c in range(3):
    x = 0
    q = 0
    v = 0
    i = 0     
    with open(links[c], "rt") as f:
        for line in f:
            commas = line.count(',') + 1
            year = line.split(',')[0]
            month = line.split(',')[1]
            day = line.split(',')[2]
            date = year + month + day
            y = 0
      

            if i == 0:     
                for t in range(3,commas):
                    station_id = line.split(',')[t]
                    data[c,x,0] = station_id
                    x = x + 1
                    
            if i == 1:     
                for t in range(3,commas):
                    lat = line.split(',')[t]
                    data[c,q,1] = lat
                    q = q + 1
            
            if i == 2:     
                for t in range(3,commas):
                    lon = line.split(',')[t]
                    data[c,v,2] = lon
                    v = v + 1

            if i != 0 and i != 1 and i != 2:
                for t in range(3,commas):   
                    precip = line.split(',')[t]
                    precip = float(precip)
                    data[c,y,i] = precip

                    y = y + 1
            
            i = i + 1

data[np.isnan(data)] = 9999


inhouse_data = data[0,:,:] 
ncar_data75 = data[1,:,:] 
ncar_data90 = data[2,:,:] 


###############################################################################
############ Determine percentiles      #######################################
###############################################################################

percent  = np.array([75,90])
percentiles  = zeros((len(data[0,:,0]),3))
p_array = []
             


for y in range(len(data[0,:,0])):
            if data[0,y,0] != 0:
                #p_array is all precip days for one station.  Created to determine percentiles for each station
                p_array = data[0,y,3:185]

                p_array = np.delete(p_array, np.where(p_array < 2.54))
                p_array = np.delete(p_array, np.where(p_array > 1000))
                
                percentile75 = np.percentile(p_array,percent[0])
                percentile90 = np.percentile(p_array,percent[1])
            
                percentiles[y,0] = data[0,y,0]
                percentiles[y,1] = percentile75
                percentiles[y,2] = percentile90


###############################################################################
############ Determine on which days an upper quartile and decile event occured    
###############################################################################

inhouse_data75 = zeros((798,186))
inhouse_data90 = zeros((798,186))
inhouse_data75[:,0:3] = inhouse_data[:,0:3]
inhouse_data90[:,0:3] = inhouse_data[:,0:3]

for w in range(len(inhouse_data[:,0])):
    if inhouse_data[w,0] != 0:
        for i in range(3,186):
            if percentiles[w,1] <= inhouse_data[w,i] < 1000 :
                inhouse_data75[w,i] = 1
            elif percentiles[w,1] > inhouse_data[w,i]:
                inhouse_data75[w,i] = 0
            elif 1000 < inhouse_data[w,i]:
                inhouse_data75[w,i] = 9999
          
          
for w in range(len(inhouse_data[:,0])):
    if inhouse_data[w,0] != 0:
        for i in range(3,186):
            if percentiles[w,2] <= inhouse_data[w,i] < 1000 :
                inhouse_data90[w,i] = 1
            elif percentiles[w,2] > inhouse_data[w,i]:
                inhouse_data90[w,i] = 0
            elif 1000 < inhouse_data[w,i]:
                inhouse_data90[w,i] = 9999   
         
###############################################################################
############ Calc forecast freq as function of observed freq ##################   
###############################################################################
         
         
### First divide into regions
region = ['Pac. NW', 'Sierra Nevada','Blue Mountains, OR','ID/Western MT','NW WY','UT','CO' ,'AZ/NM']                                       
###Serrezze regions
                   


### Steenburgh/Lewis regions (only Pacific (Far NW[1] and Sierrza Nevada[2]) and Intermountain (CO Rockies[3], Intermountain[4], Intermountain NW[5], Soutwest ID[6]))                   
regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,44, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)

### Caluclate observed frequencies for each forcasted probability                    
         
freq = zeros((2,11,7))
ss = 0
tt = 0

pacificloc = zeros((700,2))
interloc = zeros((700,2))

for r in range(0,2):    
    freq[r,:,0] = np.arange(0,1.001,.1)
    for x in range(len(ncar_data75[:,0])):
        for w in range(len(inhouse_data75[:,0])):


################### PACIFIC ###################            
            if r == 0:
                if ((regions[0,0] <= inhouse_data75[w,1] <= regions[0,1] and regions[0,2] <= inhouse_data75[w,2] <= regions[0,3]) or ###Sierra Nevada
                    
                    (regions[1,0] <= inhouse_data75[w,1] <= regions[1,1] and regions[1,2] <= inhouse_data75[w,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                    (inhouse_data75[w,1] >= regions[1,4] or inhouse_data75[w,2] <= regions[1,5])):
                    
                        
               
                    if inhouse_data75[w,0] == ncar_data75[x,0]:
                        pacificloc[ss,:] = inhouse_data75[w,1:3]
                        ss = ss + 1   
                        if inhouse_data75[w,0] != 0:
                            for i in range(3,186):
                                if 0 < inhouse_data[w,i] < 1000:
                                    for t in range(11):
                                        if ncar_data75[x,i] == t/10.:
                                            freq[r,t,1] = freq[r,t,1] + inhouse_data75[w,i]
                                            freq[r,t,2] = freq[r,t,2] + 1
                                            
                                            
################  INTERMOUNTAIN #################                                        
            if r == 1:
                if ((regions[2,0] <= inhouse_data75[w,1] <= regions[2,1] and regions[2,2] <= inhouse_data75[w,2] <= regions[2,3]) or ## CO Rockies
                    
                    (regions[3,0] <= inhouse_data75[w,1] <= regions[3,1] and regions[3,2] <= inhouse_data75[w,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    (inhouse_data75[w,1] >= regions[3,4] or inhouse_data75[w,2] <= regions[3,5]) and 
                    (inhouse_data75[w,1] <= regions[3,6] or inhouse_data75[w,2] >= regions[3,7]) or
                    
                    (regions[4,0] <= inhouse_data75[w,1] <= regions[4,1] and regions[4,2] <= inhouse_data75[w,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    (inhouse_data75[w,1] >= regions[4,4] or inhouse_data75[w,2] >= regions[4,5]) and 
                    (inhouse_data75[w,1] >= regions[4,6] or inhouse_data75[w,2] <= regions[4,7]) or
                        
                    (regions[5,0] <= inhouse_data75[w,1] <= regions[5,1] and regions[5,2] <= inhouse_data75[w,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
                    (inhouse_data75[w,1] <= regions[5,4] or inhouse_data75[w,2] <= regions[5,5])): 
                     
                        
                        
                    if inhouse_data75[w,0] == ncar_data75[x,0]:
                        interloc[tt,:] = inhouse_data75[w,1:3]
                        tt = tt + 1
                        if inhouse_data75[w,0] != 0:
                            for i in range(3,186):
                                if 0 < inhouse_data[w,i] < 1000:
                                    for t in range(11):
                                        if ncar_data75[x,i] == t/10.:
                                            freq[r,t,1] = freq[r,t,1] + inhouse_data75[w,i]
                                            freq[r,t,2] = freq[r,t,2] + 1 

print ss
print tt

### Get Elevation data


        
NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/wrfinput_d02'
fh = Dataset(NCARens_file, mode='r')

elevation = fh.variables['HGT'][:]
lat_netcdf = fh.variables['XLAT'][:]
long_netcdf = fh.variables['XLONG'][:]     
levels_el = np.arange(0,5000,100)



######################### Plot snotel locations for reliability diagrams ################                                           
fig = plt.figure(figsize=(14,12))
plt.title('SNOTEL Regions', fontsize = 28)
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(inhouse_data75[:,2], inhouse_data75[:,1])
xi, yi = map(pacificloc[:,1], pacificloc[:,0])
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
csAVG2 = map.contourf(x2,y2,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
xii, yii = map(interloc[:,1], interloc[:,0])
levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 2.2]
cmap=plt.cm.BrBG

csAVG = map.scatter(xi,yi, marker='o',  c = 'blue',s = 150, vmin = 0.4, vmax = 2.2, label = "Pacific")
csAVG2 = map.scatter(xii,yii,  marker='o', c = 'red', s = 150, vmin = 0.4, vmax = 2.2, label = "Interior")
csAVG = map.scatter(x,y, marker='o',  c = 'white',alpha = 1, s = 150, vmin = 0.4, vmax = 2.2, label = "Not Included")
csAVG = map.scatter(xi,yi, marker='o',  c = 'blue',s = 150, vmin = 0.4, vmax = 2.2,)
csAVG2 = map.scatter(xii,yii,  marker='o', c = 'red', s = 150, vmin = 0.4, vmax = 2.2)
map.drawcoastlines() 
map.drawstates()
map.drawcountries()
#blue_line = mlines.Line2D([], [], color='blue',
#                          label='Pacific',   linewidth = 2,marker = "o")
#plt.legend(handles=[blue_line], loc = "lower left",prop={'size':10.5})
plt.legend(loc = "lower left", fontsize = 17)
plt.savefig("./plots/snotel_regions.png")    
plt.show()
'''




region_name = ['Pac. NW', 'Sierra Nevada','Blue Mountains, OR','ID/Western MT','NW WY','UT','CO' ,'AZ/NM', 'Not Included']                                       





#################################################
###### Regiuons used for skill scores   #########
#################################################
regions = np.array([[41.5,49.2, -123.0,-120.5],
                    [37.0,41.0, -121.0,-118.0], 
                    [43.7,46.2, -120.0,-116.8], 
                    [43.0,49.3, -116.8,-112.2], 
                    [41.8,47.0, -112.5,-105.5],
                    [37.2,41.8, -113.9,-109.2],
                    [35.6,41.5, -108.7,-104.5],
                    [32.5,35.5, -113.0,-107.0]])
                    
snotel_regions = zeros((8,500,3))

for w in range(0,8):
    t = 0
    for y in range(len(inhouse_data[:,0])):
        if regions[w,0] <= inhouse_data[y,1] <= regions[w,1] and regions[w,2] <= inhouse_data[y,2] <= regions[w,3]:

            snotel_regions[w,t,0:2] = inhouse_data[y,1:3]
            t = t + 1

                    
 ######################### Plot snotel locations for reliability diagrams ################                                           
fig = plt.figure(figsize=(14,12))
plt.title('SNOTEL Regions', fontsize = 28)
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(snotel_regions[0,:,1], snotel_regions[0,:,0])
x1, y1 = map(snotel_regions[1,:,1], snotel_regions[1,:,0])
x2, y2 = map(snotel_regions[2,:,1], snotel_regions[2,:,0])
x3, y3 = map(snotel_regions[3,:,1], snotel_regions[3,:,0])
x4, y4 = map(snotel_regions[4,:,1], snotel_regions[4,:,0])
x5, y5 = map(snotel_regions[5,:,1], snotel_regions[5,:,0])
x6, y6 = map(snotel_regions[6,:,1], snotel_regions[6,:,0])
x7, y7 = map(snotel_regions[7,:,1], snotel_regions[7,:,0])
x8, y8 = map(inhouse_data[:,2], inhouse_data[:,1])

region_name = ['Pacific Northwest', 'Sierra Nevada','Blue Mountains, OR','Idaho/Western MT','NW Wyoming','Utah','Colorado' ,'Arizona/New Mexico', 'Not Included']
linecolor = ['blue', 'green', 'red', 'c', 'y', 'darkred', 'purple', 'salmon', 'white']
x9, y9 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
csAVG2 = map.contourf(x9,y9,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)

csAVG = map.scatter(x,y, marker='o',  c = linecolor[0],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[0])
csAVG = map.scatter(x1,y1, marker='o',  c = linecolor[1],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[1])
csAVG = map.scatter(x2,y2, marker='o',  c = linecolor[2],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[2])
csAVG = map.scatter(x3,y3, marker='o',  c = linecolor[3],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[3])
csAVG = map.scatter(x4,y4, marker='o',  c = linecolor[4],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[4])
csAVG = map.scatter(x5,y5, marker='o',  c = linecolor[5],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[5])
csAVG = map.scatter(x6,y6, marker='o',  c = linecolor[6],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[6])
csAVG = map.scatter(x7,y7, marker='o',  c = linecolor[7],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[7])
csAVG = map.scatter(x8,y8, marker='o',  c = linecolor[8],s = 150, vmin = 0.4, vmax = 2.2, label = region_name[8])

csAVG = map.scatter(x,y, marker='o',  c = linecolor[0],s = 150, vmin = 0.4, vmax = 2.2,)
csAVG = map.scatter(x1,y1, marker='o',  c = linecolor[1],s = 150, vmin = 0.4, vmax = 2.2)
csAVG = map.scatter(x2,y2, marker='o',  c = linecolor[2],s = 150, vmin = 0.4, vmax = 2.2)
csAVG = map.scatter(x3,y3, marker='o',  c = linecolor[3],s = 150, vmin = 0.4, vmax = 2.2)
csAVG = map.scatter(x4,y4, marker='o',  c = linecolor[4],s = 150, vmin = 0.4, vmax = 2.2)
csAVG = map.scatter(x5,y5, marker='o',  c = linecolor[5],s = 150, vmin = 0.4, vmax = 2.2)
csAVG = map.scatter(x6,y6, marker='o',  c = linecolor[6],s = 150, vmin = 0.4, vmax = 2.2)
csAVG = map.scatter(x7,y7, marker='o',  c = linecolor[7],s = 150, vmin = 0.4, vmax = 2.2)



map.drawcoastlines() 
map.drawstates()
map.drawcountries()
#blue_line = mlines.Line2D([], [], color='blue',
#                          label='Pacific',   linewidth = 2,marker = "o")
#plt.legend(handles=[blue_line], loc = "lower left",prop={'size':10.5})
#plt.legend(loc = "lower left", fontsize = 20)
plt.savefig("../public_html/snotel_skillscore_regions.pdf")    
plt.show()                   




