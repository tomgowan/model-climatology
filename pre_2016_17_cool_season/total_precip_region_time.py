
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
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter

'''

nearestNCAR = zeros((785,650))
totalprecip = zeros((125000))
totalprecip_hrrr = zeros((125000))
totalprecip_nam4k = zeros((125000))
Date = zeros((152))
frequency = zeros((100,10))
bias = zeros((100,10))
frequency_snotel = zeros((40,3))
daily_snotel_precip = zeros((125000))
snotel_rowloc = zeros((798))
snotel_colloc = zeros((798))



###############################################################################
############ Read in  12Z to 12Z data   #####################
###############################################################################




inhouse_data = zeros((649,186))
ncar_data = zeros((798,186))
nam4k_data = zeros((798,186))
nam12k_data = zeros((798,186))
hrrr_data = zeros((798,186))
hrrrv2_data = zeros((798,186))
            
x = 0
q = 0
v = 0
i = 0   

 # Bilinearly interpolated model data at snotel sites        
links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/snotel_precip_2015_2016_qc.csv", 
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ncarens_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/nam4km_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/hrrrV1_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/nam12km_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/gfs_precip_12Zto12Z_interp.txt"]
         #"/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/hrrrV2_precip_12Zto12Z_interp.txt"]

#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        
data = zeros((len(links),798,186))

         
for c in range(len(links)):
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
nam4km_data = data[2,:,:]
hrrrV1_data = data[3,:,:]
nam12km_data = data[4,:,:]
gfs_data = data[5,:,:]
#hrrrv2_data = data[5,:,:]


# Make sure all days for each station have good data
# If not, set equal to 0 for that day and station
for x in range(len(data[0,:,0])):
    for y in range(len(data[1,:,0])):
        if data[0,x,0] == data[1,y,0]:
                    num = 0
                    for r in range(3,len(data[0,0,:])):
                        if all(data[1:,y,r] < 1000) and data[0,x,r] < 1000:  # Make sure all models have data for that day and station
                            pass
                        else:
                            data[1:,y,r] = 0
                            data[0,x,r] = 0
                            
                            
###############################################################################
####################   Divide into regions  ###################################
###############################################################################
               
#region = ['Pacific Northwest', 'Sierra Nevada','Blue Mountains, OR','Idaho/Western MT','NW Wyoming','Utah','Colorado' ,'Arizona/New Mexico'] 

regions = np.array([[41.5,49.2, -123.0,-120.5],
                    [37.0,41.0, -121.0,-118.0], 
                    [43.7,46.2, -120.0,-116.8], 
                    [43.0,49.3, -116.8,-112.2], 
                    [41.8,47.0, -112.5,-105.5],
                    [37.2,41.8, -113.9,-109.2],
                    [35.6,41.5, -108.7,-104.5],
                    [32.5,35.5, -113.0,-107.0]])

                    
### This complex loop matches snotel sites for model forecasts and observed data, determines region of snotel site, then adds precip data for all snotel locations in that region                    
regions_precip = zeros((8,15,186))                

for model in range(1, len(links)): 
    for x in range(len(data[model,:,0])): #Snotel lat lon
            print x
            for y in range(len(data[0,:,0])): #Model lat lon
                for w in range(0,len(regions[:,0])): #Determine region of lat and lon
                    if regions[w,0] <= data[0,y,1] <= regions[w,1] and regions[w,2] <= data[0,y,2] <= regions[w,3]:
                        if data[model,x,0] == data[0,y,0]:
                            regions_precip[w,model-1,3:] =  regions_precip[w,model-1,3:] + data[model,x,3:]  
                            regions_precip[w,model-1,0] =  regions_precip[w,model-1,0] + 1  ###Counter
                            regions_precip[w,3+model,3:] =  regions_precip[w,3+model,3:] + data[0,y,3:]
                            regions_precip[w,3+model,0] =  regions_precip[w,3+model,0] + 1 ## Counter


### This loop determines the the mean accumukated precip for each snotel region
for w in range(0,len(regions[:,0])):
    for x in range(len(regions_precip[0,:,0])):
        for y in range(3,len(regions_precip[0,0,:])):
            regions_precip[w,x,y] = regions_precip[w,x,y] + regions_precip[w,x,y-1]
        regions_precip[w,x,3:] = regions_precip[w,x,3:]/regions_precip[w,x,0]

#Place final data into new, clean array
precip_final = zeros((8,183,6))
for w in range(0,len(regions[:,0])):
    for x in range(6):
        for y in range(3,len(regions_precip[0,0,:])):
            precip_final[w,y-3,x] = regions_precip[w,x,y]



###############################################################################
################################## PLOTS ######################################
###############################################################################

'''


linecolor = ['blue', 'green', 'red', 'c', 'gold','k']
t = 0


plt.gca().set_color_cycle(linecolor)



region = ['Pacific Northwest', 'Sierra Nevada','Blue Mountains, OR','Idaho/Western MT','NW Wyoming','Utah','CO' ,'AZ/NM']                
x = np.arange(0,183,1)

fig1=plt.figure(num=None, figsize=(21,15), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.2, bottom = 0.17)

for i in range(0,8):
    plot = 241+i
    ax1 = fig1.add_subplot(plot)
    if i != 0:
        plt.yticks(np.arange(0,np.max(precip_final[i,182,:]),100))
        ax1.set_yticklabels(np.arange(0,np.max(precip_final[i,182,:]),100), fontsize = 16)
    else:
        plt.yticks(np.arange(0,np.max(precip_final[i,182,:]),200))
        ax1.set_yticklabels(np.arange(0,np.max(precip_final[i,182,:]),200), fontsize = 16)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.gca().set_color_cycle(linecolor)
    ax1.plot(x,precip_final[i,:,:],linewidth = 2, marker = "None", markeredgecolor = 'none')
    plt.xlim([0,200])
    plt.xticks(np.arange(0,201,40))
    ax1.set_xticklabels(np.arange(0,201,40), fontsize = 16)
    ax1.yaxis.label.set_size(20)
    plt.grid(True)

    blue_line = mlines.Line2D([], [],linewidth = 3, marker = "None", markeredgecolor = 'none', color='blue',
                           label='NCARens Control')
    green_line = mlines.Line2D([], [],linewidth = 3, marker = "None", markeredgecolor = 'none', color='green',
                           label='NAM-4km')
    red_line = mlines.Line2D([], [],linewidth = 3, marker = "None", markeredgecolor = 'none', color='red',
                           label='HRRR')
    cyan_line = mlines.Line2D([], [], linewidth = 3, marker = "None", markeredgecolor = 'none',color='c',
                           label='NAM-12km')
    gold_line = mlines.Line2D([], [], linewidth = 3, marker = "None", markeredgecolor = 'none',color='gold',
                           label='GFS')
    black_line = mlines.Line2D([], [],linewidth = 3, marker = "None", markeredgecolor = 'none', color='k',
                           label='SNOTEL')
    
    plt.title(region[i], fontsize = 24)
            
    
    if i == 0 or i == 4:
        plt.ylabel('Accumulated Precip. (mm)', fontsize = 20, labelpad = 13)

    if i == 4 or  i == 5 or i == 6 or  i == 7:
        plt.xlabel('Day', fontsize = 20, labelpad = 13)

plt.legend(handles=[ blue_line, green_line, red_line, cyan_line, gold_line, black_line], loc='upper center', bbox_to_anchor=(-1.28, -0.16), 
           fancybox=True, shadow=True, ncol=6,fontsize = 21)
#plt.tight_layout()
plt.savefig('../../public_html/mean_accum_precip_region.png')












    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    