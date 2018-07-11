#%%
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
############ Read in  12Z to 12Z data   #######################################
###############################################################################

            
x = 0
q = 0
v = 0
i = 0   

         

links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/snotel_precip_2016_2017_qc.csv",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/ncarens_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/gfs13km_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/hrrr_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/nam3km_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_arw_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_nmb_precip_12Zto12Z_interp.txt"]

#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        
data = zeros((len(links),798,185))

         
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
gfs_data = data[2,:,:]
hrrr_data = data[3,:,:]
nam_data = data[4,:,:]
sref_arw_data = data[5,:,:]
sref_nmb_data = data[6,:,:]



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
               

regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,44, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)
                   

                    
### This complex loop matches snotel sites for model forecasts and observed data, determines region of snotel site, then adds precip data for all snotel locations in that region                    
regions_precip = zeros((2,12,185))                

for model in range(1, len(links)): 
    for x in range(len(data[model,:,0])): #Snotel lat lon
            print x
            for y in range(len(data[0,:,0])): #Model lat lon
                for w in range(0,2): #Determine region of lat and lon
                ################### PACIFIC ###################            
                    if w == 0:
                        if ((regions[0,0] <= data[model,x,1] <= regions[0,1] and regions[0,2] <= data[model,x,2] <= regions[0,3]) or ###Sierra Nevada
                            
                            (regions[1,0] <= data[model,x,1] <= regions[1,1] and regions[1,2] <= data[model,x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                            (data[model,x,1] >= regions[1,4] or data[model,x,2] <= regions[1,5])):



                            if data[model,x,0] == data[0,y,0]:
                                regions_precip[w,model-1,3:] =  regions_precip[w,model-1,3:] + data[model,x,3:]  
                                regions_precip[w,model-1,0] =  regions_precip[w,model-1,0] + 1  ###Counter
                                regions_precip[w,5+model,3:] =  regions_precip[w,5+model,3:] + data[0,y,3:]
                                regions_precip[w,5+model,0] =  regions_precip[w,5+model,0] + 1 ## Counter
                                
                                
                ################  INTERMOUNTAIN #################                                        
                    if w == 1:
                        if ((regions[2,0] <= data[model,x,1] <= regions[2,1] and regions[2,2] <= data[model,x,2] <= regions[2,3]) or ## CO Rockies
    
                            (regions[3,0] <= data[model,x,1] <= regions[3,1] and regions[3,2] <= data[model,x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                            (data[model,x,1] >= regions[3,4] or data[model,x,2] <= regions[3,5]) and 
                            (data[model,x,1] <= regions[3,6] or data[model,x,2] >= regions[3,7]) or
                            
                            (regions[4,0] <= data[model,x,1] <= regions[4,1] and regions[4,2] <= data[model,x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                            (data[model,x,1] >= regions[4,4] or data[model,x,2] >= regions[4,5]) and 
                            (data[model,x,1] >= regions[4,6] or data[model,x,2] <= regions[4,7]) or
                            
                            (regions[5,0] <= data[model,x,1] <= regions[5,1] and regions[5,2] <= data[model,x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
                            (data[model,x,1] <= regions[5,4] or data[model,x,2] <= regions[5,5])):  
                                
                            
                                
                            if data[model,x,0] == data[0,y,0]:
                                regions_precip[w,model-1,3:] =  regions_precip[w,model-1,3:] + data[model,x,3:]  
                                regions_precip[w,model-1,0] =  regions_precip[w,model-1,0] + 1  ###Counter
                                regions_precip[w,5+model,3:] =  regions_precip[w,5+model,3:] + data[0,y,3:]
                                regions_precip[w,5+model,0] =  regions_precip[w,5+model,0] + 1 ## Counter
                                
#%%                               


### This loop determines the the mean accumukated precip for each snotel region
for w in range(0,2):
    for x in range(len(regions_precip[0,:,0])):

        for y in range(3,len(regions_precip[0,0,:])):
            regions_precip[w,x,y] = regions_precip[w,x,y]/regions_precip[w,x,0]
#%%
#Place final data into new, clean array
precip_final = zeros((2,182,7))
for w in range(0,2):
    for x in range(7):
        for y in range(3,len(regions_precip[0,0,:])):
            precip_final[w,y-3,x] = regions_precip[w,x,y]

for w in range(2):
    for i in range(6):
        precip_final[w,:,i] = precip_final[w,:,i]/precip_final[w,:,6]

###############################################################################
################################## PLOTS ######################################
###############################################################################


#%%

linecolor = ['blue', 'green', 'red', 'c', 'gold','magenta','k']
t = 0


plt.gca().set_color_cycle(linecolor)



region = ['Pacific', 'Interior']                
x = np.arange(0,182,1)

fig1=plt.figure(num=None, figsize=(13,14), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.2, bottom = 0.17)

for i in range(0,2):
    plot = 211+i
    ax1 = fig1.add_subplot(plot)
    #if i == 1:
    #    plt.yticks(np.arange(0,np.max(precip_final[i,181,:]),100))
    #    ax1.set_yticklabels(np.arange(0,np.max(precip_final[i,181,:]),100), fontsize = 16)
        
    #if i == 0:
    #    plt.yticks(np.arange(0,np.max(precip_final[i,181,:]),200))
    #    ax1.set_yticklabels(np.arange(0,np.max(precip_final[i,181,:]),200), fontsize = 16)

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.gca().set_color_cycle(linecolor)
    ax1.scatter(x,precip_final[i,:,1])#, markeredgecolor = 'none')
    plt.xlim([0,182])
    plt.xticks(np.arange(0,182,20))
    ax1.set_xticklabels(np.arange(0,182,20), fontsize = 16)
    ax1.yaxis.label.set_size(20)
    plt.grid(True)
    blue = mlines.Line2D([], [], color='blue',
                           label='NCAR ENS CTL', linewidth = 2,marker = "o", markeredgecolor = 'none')
    green = mlines.Line2D([], [], color='green',
                           label='GFS', linewidth = 2,marker = "o", markeredgecolor = 'none')
    red = mlines.Line2D([], [], color='red',
                           label='HRRR', linewidth = 2,marker = "o", markeredgecolor = 'none')
    cyan = mlines.Line2D([], [], color='c',
                           label='NAM-3km', linewidth = 2,marker = "o", markeredgecolor = 'none')
    gold = mlines.Line2D([], [], color='gold',
                           label='SREF ARW CTL', linewidth = 2,marker = "o", markeredgecolor = 'none')
    magenta = mlines.Line2D([], [], color='magenta',
                           label='SREF NMB CTL', linewidth = 2,marker = "o", markeredgecolor = 'none')

    
    plt.title(region[i], fontsize = 24)
            
    

    plt.ylabel('Accumulated Precip. (mm)', fontsize = 20, labelpad = 13)

    if i == 1:
        plt.xlabel('Day of Cool-Season', fontsize = 20, labelpad = 13)

plt.legend(handles=[ blue, green, red, cyan, gold, magenta], loc='upper center', bbox_to_anchor=(.134, 2.2), 
           ncol=1,fontsize = 16)
#plt.tight_layout()
plt.savefig("../../../public_html/mean_accum_precip_region_2016_17_test.pdf")
plt.show()










    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    