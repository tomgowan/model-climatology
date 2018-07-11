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
from mpl_toolkits.basemap import Basemap, maskoceans


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
region_bias=zeros((4,8))


###############################################################################
############ Read in  12Z to 12Z data   #####################
###############################################################################
inhouse_data = zeros((798,186))
ncar_data = zeros((798,186))
nam4k_data = zeros((798,186))
nam12k_data = zeros((798,186))
hrrr_data = zeros((798,186))
            
x = 0
q = 0
v = 0
i = 0   

links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/snotel/Tom_in_house.csv", 
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/ncarens_precip_12Zto12Z.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/nam4k_precip_12Zto12Z.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/hrrr_precip_12Zto12Z.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/nam12k_precip_12Zto12Z.txt"]

#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        
data = zeros((5,798,186))

         
for c in range(5):
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
ncar_data = data[1,:,:] 
nam4k_data = data[2,:,:] 
hrrr_data = data[3,:,:]
nam12k_data = data[4,:,:]


###############################################################################
###########################   Calculate Bias  #################################
###############################################################################
bias = zeros((800,20))



for c in range(1,5):
    t = 7 + c
    w = -1
    for x in range(len(data[c,:,0])):
        for y in range(len(inhouse_data[:,0])):
                if data[c,x,0] == inhouse_data[y,0]:
                    w = w + 1
                    bias[w,0] = data[c,x,0]
                    bias[w,1] = inhouse_data[y,0]
                

                    for z in range(3,185):
                        if data[c,x,z] < 1000 and inhouse_data[y,z] < 1000:
                        
                            #lat/lon data
                            bias[w,2] = data[c,x,1]
                            bias[w,3] = data[c,x,2]
                            bias[w,4] = inhouse_data[y,1]
                            bias[w,5] = inhouse_data[y,2]   
                        
                            #precip data
                            bias[w,6] = bias[w,6] + inhouse_data[y,z]
                            bias[w,7] = bias[w,7] + data[c,x,z]
                            bias[w,t] = bias[w,7]/bias[w,6]

                       

                        
                        
                       
                        
###############################################################################
###########################   Divide into regions  ############################
###############################################################################


regions = np.array([[41.5,49.2, -123.0,-120.5],
                    [37.0,41.0, -121.0,-118.0], 
                    [43.7,46.2, -120.0,-116.8], 
                    [43.0,49.3, -116.8,-112.2], 
                    [41.8,47.0, -112.5,-105.5],
                    [37.2,41.8, -113.9,-109.2],
                    [35.6,41.5, -108.7,-104.5],
                    [32.5,35.5, -113.0,-107.0]])
                    
                    
region_bias = zeros((4,8))



for w in range(8):
    x = 0
    y = 0
    z = 0
    a = 0
    r = 0
    for i in range(649):

        if regions[w,0] <= bias[i,2] <= regions[w,1] and regions[w,2] <= bias[i,3] <= regions[w,3]:
 
            #NCAR
            x = x + bias[i,8]
            #NAM4k
            y = y + bias[i,9]
            #HRRR
            z = z + bias[i,10]
            #NAM12k
            a = a + bias[i,11]
            
            
            r = r + 1

    region_bias[0,w] = x/r
    region_bias[1,w] = y/r
    region_bias[2,w] = z/r
    region_bias[3,w] = a/r

'''


#####  NCAR Prism bias data ####################################
#############  NCAR   ######
precip_ncar = np.loadtxt('ncarens_dailymean.txt')
precip_tot = np.loadtxt('prism_hrrr_dailymean.txt')



#####   HRRR   ############
precip_hrrr = np.loadtxt('hrrr_dailymean.txt')
precip_tot = np.loadtxt('prism_hrrr_dailymean.txt')


#############  NAM4km   ######
precip_nam4k = np.loadtxt('nam4k_dailymean.txt')
precip_tot = np.loadtxt('prism_hrrr_dailymean.txt')



snotel_ncar = bias[:,8]
snotel_hrrr = bias[:,10]
snotel_nam4k = bias[:,9]



###############################################################################
##############   Calc biases for western US  ##################################
###############################################################################

#############  NCAR   ######
avg1 = precip_ncar[17:453, 0:540]/precip_tot[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_ncar = np.average(avg)
bias_stdev_ncar = np.std(avg)
#avg_low = avg1[(avg1 > 1) & (avg1 < 10)]
#avg_high = avg1[(avg1 > 0.1) & (avg1 < 1)]

#bias_mean_ncar_low = np.average(avg_low)
#bias_mean_ncar_high = np.average(avg_high)

snotel_mean_ncar = np.mean(snotel_ncar)
snotel_stdev_ncar = np.std(snotel_ncar)
#avg_low = snotel_ncar[(snotel_ncar < 1)]
#avg_high = snotel_ncar[(snotel_ncar > 1)]
#snotel_mean_ncar_low = np.mean(avg_low)
#snotel_mean_ncar_high = np.mean(avg_high)


#####   HRRR   ############
avg1 = precip_hrrr[17:453, 0:540]/precip_tot[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_hrrr = np.average(avg)
bias_stdev_hrrr = np.std(avg)


snotel_mean_hrrr = np.mean(snotel_hrrr)
snotel_stdev_hrrr = np.std(snotel_hrrr)



#####  NAM4km   ############
avg1 = precip_nam4k[17:453, 0:540]/precip_tot[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_nam4k = np.average(avg)
bias_stdev_nam4k = np.std(avg)


snotel_mean_nam4k = np.mean(snotel_nam4k)
snotel_stdev_nam4k = np.std(snotel_nam4k)







lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667







#cmap= custom_div_cmap(400, mincol='#5D3809', midcol='w' ,maxcol='Bg')


cmap=plt.cm.BrBG
#cmap.set_over('.4')
#cmap.set_under()
sub = 221    
    
titles = ['NCARens Control','NAM-4km', 'HRRR', 'NAM-12km']
fig1=plt.figure(num=None, figsize=(12,15), dpi=800, facecolor='w', edgecolor='k')
#fig1.subplots_adjust(hspace=0, bottom = 0)

for i in range(4):
    s = 8 + i
    
    ax = fig1.add_subplot(sub)
    plt.tight_layout()
    map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
    xi, yi = map(bias[:,5], bias[:,4])
    levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 2.2]
    csAVG = map.scatter(xi,yi, c = bias[:,s], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 75, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )
    

    
    #ax.set_cscale('log')
    map.drawcoastlines()  ###RdYlBu
    map.drawstates()
    map.drawcountries()

    ax.set_title(titles[i], fontsize = 22)


    if sub == 223 or sub == 224:   
        
        
        cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,2.2])
        cbar.ax.set_xlabel('Daily Precipitation Bias', fontsize  = 14)
        
        #cbar.ax.set_xlabel('mm', fontsize  = 14)
        cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
    sub = sub + 1
  
plt.savefig("./plots/regional_bias_map_allmodels.pdf")
plt.show()










##############################################################################
####  Bias form snotel and prism  for ncar  ##################################
##############################################################################



fig1=plt.figure(num=None, figsize=(25,10), dpi=800, facecolor='w', edgecolor='k')


ax = fig1.add_subplot(131)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[:,5], bias[:,4])
x, y = map(lons_prism, lats_prism)
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.contourf(x,y,precip_ncar/precip_tot, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG = map.scatter(xi,yi, c = bias[:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 65, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("NCAR Ensemble Control", fontsize = 20)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xlabel('Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (SNOTEL and PRISM)', fontsize  = 12)       
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2']) 
plt.annotate('Mean bias PRISM = %1.3f\n' % bias_mean_ncar  +
             'Std dev bias PRISM = %1.3f' % bias_stdev_ncar, xy=(0.013, .013),
             xycoords='axes fraction', fontsize = 12, backgroundcolor = 'w')
plt.annotate('Mean bias SNOTEL = %1.3f\n' % snotel_mean_ncar  +
             'Std dev bias SNOTEL = %1.3f' % snotel_stdev_ncar, xy=(0.67, .013),
             xycoords='axes fraction', fontsize = 12, backgroundcolor = 'w')



##################################     HRRR     ###############################


ax = fig1.add_subplot(132)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[:,5], bias[:,4])
x, y = map(lons_prism, lats_prism)
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.contourf(x,y,precip_hrrr/precip_tot, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG = map.scatter(xi,yi, c = bias[:,10], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 65, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("HRRR", fontsize = 20)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xlabel('Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (SNOTEL and PRISM)', fontsize  = 12)       
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2']) 
plt.annotate('Mean bias PRISM = %1.3f\n' % bias_mean_hrrr  +
             'Std dev bias PRISM = %1.3f' % bias_stdev_hrrr, xy=(0.013, .013),
             xycoords='axes fraction', fontsize = 12, backgroundcolor = 'w')
plt.annotate('Mean bias SNOTEL = %1.3f\n' % snotel_mean_hrrr  +
             'Std dev bias SNOTEL = %1.3f' % snotel_stdev_hrrr, xy=(0.67, .013),
             xycoords='axes fraction', fontsize = 12, backgroundcolor = 'w')






##################################     NAM4km     #############################

ax = fig1.add_subplot(133)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[:,5], bias[:,4])
x, y = map(lons_prism, lats_prism)
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.contourf(x,y,precip_nam4k/precip_tot, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG = map.scatter(xi,yi, c = bias[:,9], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 65, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("NAM-4km", fontsize = 20)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xlabel('Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (SNOTEL and PRISM)', fontsize  = 12)       
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2']) 
plt.annotate('Mean bias PRISM = %1.3f\n' % bias_mean_nam4k  +
             'Std dev bias PRISM = %1.3f' % bias_stdev_nam4k, xy=(0.013, .013),
             xycoords='axes fraction', fontsize = 12, backgroundcolor = 'w')
plt.annotate('Mean bias SNOTEL = %1.3f\n' % snotel_mean_nam4k  +
             'Std dev bias SNOTEL = %1.3f' % snotel_stdev_nam4k, xy=(0.67, .013),
             xycoords='axes fraction', fontsize = 12, backgroundcolor = 'w')


plt.tight_layout()
plt.savefig("./plots/regional_bias_ncar_prism_snotel.pdf")
plt.show()



'''




#######   Bias Bar Graph
N = len(region_bias[:,0])
x = range(N)
region = ['Pacific Northwest', 'Sierra Nevada','Blue Mountains, OR','Idaho/Western MT','NW Wyoming','Utah','Colorado' ,'Arizona/New Mexico']   
sub = 241

fig = plt.figure(figsize=(16,12))
######   11111111

for i in range(8):
    ax1 = fig.add_subplot(sub)
    ax1.set_title(region[i], fontsize = 20)
    ax1.bar(x,region_bias[:,i],width = 1, color = ['blue', 'green', 'red', 'c'], edgecolor ='none')
    if sub == 241 or sub == 245:
        ax1.set_ylabel('Bias', fontsize = 18)
    ax1.set_yticks(np.arange(0.4999,2,.1))
    plt.tight_layout()
    ax1.set_xticks([.5, 1.5, 2.5, 3.5])
    if sub == 245 or sub == 246 or sub == 247 or sub == 248:
        ax1.set_xticklabels(('NCARens\nControl', 'NAM-4km', 'HRRR', 'NAM-12km'), fontsize  = 15, rotation = 45)
    else:
        ax1.set_xticklabels(('', '', '', ''))
    plt.ylim([0.5999,1.5])
    plt.axhline(1, color = 'k', lw  = 2)
    ax1.yaxis.grid()
    sub = sub + 1 
    
plt.savefig("../plots/bias_by_region2.pdf")
plt.show()  





































