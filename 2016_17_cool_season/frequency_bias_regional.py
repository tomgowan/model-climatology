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



frequency = zeros((2,80,8))
bias = zeros((2,80,8))
frequency_snotel = zeros((80,3))




###############################################################################
############ Read in  12Z to 12Z data   #####################
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



###############################################################################
##### Dvide into regionms#####################################################
###############################################################################
### Steenburgh/Lewis regions (only Pacific (Far NW[1] and Sierrza Nevada[2]) and Intermountain (CO Rockies[3], Intermountain[4], Intermountain NW[5], Soutwest ID[6]))                   
regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,44, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)



                    

                                            
                                            

   
j = 0
print c
# print ranges in array
for x in range(1,30):
        x = x*2.54
        y = x-2.54
        frequency[0,j,c+1] = x#+1.27
        bias[0,j,c+1] = x#+1.27
        j = j + 1


pm = 0
iim = 0
test = zeros((500,185))
ps = 0
iis = 0
## Frequency of event for each model
num = 0
for w in range(2):
    for c in range(1,len(links)): # loop over snotel, ncar, nam4k, hrrr, nam12k, hrrrv2
        print c
        j = 0
        for x in range(1,30):#(1,30)::

            x = (x*2.54)+1.27#3.81
            y = x-2.54#5.08
         
            for z in range(len(data[c,:,0])):
                for f in range(len(data[0,:,0])):
                    if data[c,z,0] == data[0,f,0]:
                        ################### PACIFIC ###################            
                        if w == 0:
                            if ((regions[0,0] <= data[c,z,1] <= regions[0,1] and regions[0,2] <= data[c,z,2] <= regions[0,3]) or ###Sierra Nevada
                    
                                (regions[1,0] <= data[c,z,1] <= regions[1,1] and regions[1,2] <= data[c,z,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                                (data[c,z,1] >= regions[1,4] or data[c,z,2] <= regions[1,5])):
                              
                                for r in range(3,185):
                                    if all(data[1:,z,r] < 1000) and data[0,f,r] < 1000:
                                        if data[c,z,r] > y and data[c,z,r] <= x:
                                            frequency[w,j,c] = frequency[w,j,c] + 1
                                            
                                          
                        ################  INTERMOUNTAIN #################                                        
                        if w == 1:
                            if ((regions[2,0] <= data[c,z,1] <= regions[2,1] and regions[2,2] <= data[c,z,2] <= regions[2,3]) or ## CO Rockies
                    
                                (regions[3,0] <= data[c,z,1] <= regions[3,1] and regions[3,2] <= data[c,z,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                                (data[c,z,1] >= regions[3,4] or data[c,z,2] <= regions[3,5]) and 
                                (data[c,z,1] <= regions[3,6] or data[c,z,2] >= regions[3,7]) or
                                
                                (regions[4,0] <= data[c,z,1] <= regions[4,1] and regions[4,2] <= data[c,z,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                                (data[c,z,1] >= regions[4,4] or data[c,z,2] >= regions[4,5]) and 
                                (data[c,z,1] >= regions[4,6] or data[c,z,2] <= regions[4,7]) or
                                    
                                (regions[5,0] <= data[c,z,1] <= regions[5,1] and regions[5,2] <= data[c,z,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
                                (data[c,z,1] <= regions[5,4] or data[c,z,2] <= regions[5,5])): 
                                 
                                 iim = iim + 1
                                 for r in range(3,185):
                                     if all(data[1:,z,r] < 1000) and data[0,f,r] < 1000:                           
                                         if data[c,z,r] > y and data[c,z,r] <= x:
                                             frequency[w,j,c] = frequency[w,j,c] + 1
                            
            j = j + 1






print('\n\n') 
j = 0
## Frequency of event for snotel data
for w in range(2):
    j = 0
    for x in range(1,30):#(1,30):

            x = (x*2.54)+1.27#3.81
            y = x-2.54#5.08
      
            for f in range(len(data[0,:,0])):
                for z in range(len(data[1,:,0])):
                    if data[1,z,0] == data[0,f,0]:
                        ################### PACIFIC ###################            
                        if w == 0:
                            if ((regions[0,0] <= data[c,z,1] <= regions[0,1] and regions[0,2] <= data[c,z,2] <= regions[0,3]) or ###Sierra Nevada
                    
                                (regions[1,0] <= data[c,z,1] <= regions[1,1] and regions[1,2] <= data[c,z,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                                (data[c,z,1] >= regions[1,4] or data[c,z,2] <= regions[1,5])):
                                
                                ps = ps + 1
                                print(data[0,f,0])
                                for r in range(3,185):
                                    if all(data[1:,z,r] < 1000) and data[0,f,r] < 1000:  # Make sure all models have data for that day
                                        if data[0,f,r] > y and data[0,f,r] <= x:
                                            frequency[w,j,0] = frequency[w,j,0] + 1
                        ################  INTERMOUNTAIN #################                                        
                        if w == 1:
                            if ((regions[2,0] <= data[c,z,1] <= regions[2,1] and regions[2,2] <= data[c,z,2] <= regions[2,3]) or ## CO Rockies
                    
                                (regions[3,0] <= data[c,z,1] <= regions[3,1] and regions[3,2] <= data[c,z,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                                (data[c,z,1] >= regions[3,4] or data[c,z,2] <= regions[3,5]) and 
                                (data[c,z,1] <= regions[3,6] or data[c,z,2] >= regions[3,7]) or
                                
                                (regions[4,0] <= data[c,z,1] <= regions[4,1] and regions[4,2] <= data[c,z,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                                (data[c,z,1] >= regions[4,4] or data[c,z,2] >= regions[4,5]) and 
                                (data[c,z,1] >= regions[4,6] or data[c,z,2] <= regions[4,7]) or
                                    
                                (regions[5,0] <= data[c,z,1] <= regions[5,1] and regions[5,2] <= data[c,z,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
                                (data[c,z,1] <= regions[5,4] or data[c,z,2] <= regions[5,5])): 
                                
                                iis = iis +1
                                for r in range(3,185):
                                    if all(data[1:,z,r] < 1000) and data[0,f,r] < 1000:  # Make sure all models have data for that day
                                        if data[0,f,r] > y and data[0,f,r] <= x:
                                            frequency[w,j,0] = frequency[w,j,0] + 1
            j = j + 1
#%%
print(pm)
print(iim)
print(ps)
print(iis)
#%%

np.save('frequency_totals', frequency)
frequency_save = frequency
test = frequency_save
#%%
for w in range(2):
    sum_inhouse = sum(frequency[w,:,0])
    sum_ncar = sum(frequency[w,:,1])
    sum_gfs = sum(frequency[w,:,2])
    sum_hrrr = sum(frequency[w,:,3])
    sum_nam = sum(frequency[w,:,4])
    sum_sref_arw = sum(frequency[w,:,5])
    sum_sref_nmb = sum(frequency[w,:,6])
    
    #### Calculate Frequencues



    for i in range(0,80):
 
         ### POSSIBLY FOUND MAJOR FLAW HERE.  WAS DIVIEDING BY SUM OF NUMBER OF EVENTS FOR EACH MODEL
        frequency[w,i,0]= frequency[w,i,0]/sum_inhouse*100 
        frequency[w,i,1]= frequency[w,i,1]/sum_inhouse*100 
        frequency[w,i,2]= frequency[w,i,2]/sum_inhouse*100 
        frequency[w,i,3]= frequency[w,i,3]/sum_inhouse*100 
        frequency[w,i,4]= frequency[w,i,4]/sum_inhouse*100  
        frequency[w,i,5]= frequency[w,i,5]/sum_inhouse*100
        frequency[w,i,6]= frequency[w,i,6]/sum_inhouse*100

        for i in range(0,80):
      
            bias[w,i,0] = frequency[w,i,1]/frequency[w,i,0] #NCARens
            bias[w,i,1] = frequency[w,i,2]/frequency[w,i,0] #gfs
            bias[w,i,2] = frequency[w,i,3]/frequency[w,i,0] #HRRRR
            bias[w,i,3] = frequency[w,i,4]/frequency[w,i,0] #nam
            bias[w,i,4] = frequency[w,i,5]/frequency[w,i,0] #sref_arw
            bias[w,i,5] = frequency[w,i,6]/frequency[w,i,0] #sef_nmb


#%%

sample = np.load('frequency_totals.npy')

### Pacific ##########
props = dict(boxstyle='square', facecolor='white', alpha=1)
fig1=plt.figure(num=None, figsize=(11, 11), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.15, bottom = 0.2)
ax1 = fig1.add_subplot(211)
plt.xlim([0,50])
plt.xticks(np.arange(1.27,80,2.54))


w = 0
plt.grid(True)
line1 = ax1.plot(bias[w,:,7],bias[w,:,0])
line2 = ax1.plot(bias[w,:,7],bias[w,:,1])
line3 = ax1.plot(bias[w,:,7],bias[w,:,2])
line4 = ax1.plot(bias[w,:,7],bias[w,:,3])
line5 = ax1.plot(bias[w,:,7],bias[w,:,4])
line6 = ax1.plot(bias[w,:,7],bias[w,:,5])

x = np.linspace(0, 100, 100)
y = np.linspace(1,1,100)
line7 = ax1.plot(x,y)

plt.yticks(np.arange(0.2,2.5001,0.2))
ax1.set_yticklabels(np.arange(0.2,2.5001,0.2), fontsize = 16)
plt.ylim([.4,2.501])
ax1.set_xticklabels(['1.3', ' ','6.4',' ','11.4',' ','16.5',' ','21.6',' ','26.7',
                     ' ','31.8',' ','36.8',' ','41.9',' ','47.0'], fontsize = 16)
plt.xlim([0,50])
#ax1.set_yticks([50,20,10,5,2,1,0.5,0.2,0.1,0.05])
#ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.2f'))

plt.setp(line1, color='blue', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line2, color='green', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line3, color='red', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line4, color='c', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line5, color='gold', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line6, color='magenta', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line7, color='k', linewidth=2)
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
                           label='SREF NMMB CTL', linewidth = 2,marker = "o", markeredgecolor = 'none')
ax1.fill_between(np.arange(0,55,1), 0.85, 0.4, facecolor = 'saddlebrown',alpha=0.25)
ax1.fill_between(np.arange(0,55,1), 1.2, 2.5, facecolor = 'darkgreen',alpha=0.25)
ax1.text(1, 2.24, '(a) Pacific Ranges', fontsize = 25, bbox = props)
plt.title('         ', fontsize = 22, y = 1.04)

plt.ylabel('Frequency Bias', fontsize = 18, labelpad = 10)












### Intermpountain ##########


ax2 = fig1.add_subplot(212)
plt.xlim([0,50])
plt.xticks(np.arange(1.27,80,2.54))


w = 1
plt.grid(True)
line1 = ax2.plot(bias[0,:,7],bias[w,:,0])
line2 = ax2.plot(bias[0,:,7],bias[w,:,1])
line3 = ax2.plot(bias[0,:,7],bias[w,:,2])
line4 = ax2.plot(bias[0,:,7],bias[w,:,3])
line5 = ax2.plot(bias[0,:,7],bias[w,:,4])
line6 = ax2.plot(bias[0,:,7],bias[w,:,5])

x = np.linspace(0, 100, 100)
y = np.linspace(1,1,100)
line7 = ax2.plot(x,y)

plt.yticks(np.arange(0.2,2.5001,0.2))
ax2.set_yticklabels(np.arange(0.2,2.5001,0.2), fontsize = 16)
plt.ylim([.4,2.501])
ax2.set_xticklabels(['1.3', ' ','6.4',' ','11.4',' ','16.5',' ','21.6',' ','26.7',
                     ' ','31.8',' ','36.8',' ','41.9',' ','47.0'], fontsize = 16)
plt.xlim([0,50])
#ax1.set_yticks([50,20,10,5,2,1,0.5,0.2,0.1,0.05])
#ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.2f'))

plt.setp(line1, color='blue', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line2, color='green', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line3, color='red', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line4, color='c', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line5, color='gold', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line6, color='magenta', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line7, color='k', linewidth=2)
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
                           label='SREF NMMB CTL', linewidth = 2,marker = "o", markeredgecolor = 'none')
#plt.legend(handles=[blue, green, red, cyan], loc = 'upper left',bbox_to_anchor=(0.03, 1), fontsize = 16)




        
    
ax2.fill_between(np.arange(0,55,1), 0.85, 0.4, facecolor = 'saddlebrown',alpha=0.25)
ax2.fill_between(np.arange(0,55,1), 1.2, 2.5, facecolor = 'darkgreen',alpha=0.25)
        
if i == 1:
    ax1.fill_between(x, precip_final[i,:,6], l, facecolor = 'darkgreen',alpha=0.25)
        
        
plt.xlabel('Event Size Bin (mm)', fontsize = 18)
plt.ylabel('Frequency Bias', fontsize = 18, labelpad = 10)
ax2.text(1, 2.24, '(b) Interior Ranges', fontsize = 25, bbox = props)
plt.legend(handles=[ blue, red, cyan,green, gold, magenta], loc='upper center', bbox_to_anchor=(0.5, -0.2), 
             ncol=3,fontsize = 15)

#Sample size
a = plt.axes([.21, .757, .2, .07], axisbg='white')
plt.bar(sample[0,:,7],sample[0,:,0],width = 1.6, color = 'k', edgecolor ='none',align='center')
plt.xlim([0,50])

#plt.title('SREF NMMB', y = 1.05, fontsize = 13)
#plt.text(0.56, 2000, 'Interior\nRanges', fontsize = 12)
plt.ylabel('# Obs.', fontsize = 11)
plt.xlabel('Precip. Bin (mm)',fontsize = 11)
a.set_yscale('log')
plt.ylim([1,100000])
#plt.xticks()
plt.xticks(np.arange(0,51,10), fontsize = 10)
#plt.yticks(np.arange(0,5001,600), fontsize = 10)
#a.set_yticklabels(['0', '600', '1200', '1800', '2400', '>3000'])
plt.grid(True)


a = plt.axes([.21, .382, .2, .07], axisbg='white')
plt.bar(sample[0,:,7],sample[1,:,0],width = 1.6, color = 'k', edgecolor ='none',align='center')
plt.xlim([0,50])
#plt.ylim([0,3000])
#plt.title('SREF NMMB', y = 1.05, fontsize = 13)
#plt.text(0.56, 2000, 'Interior\nRanges', fontsize = 12)
plt.ylabel('# Obs.', fontsize = 11)
plt.xlabel('Precip. Bin (mm)',fontsize = 11)
a.set_yscale('log')
#plt.xticks()
plt.xticks(np.arange(0,51,10), fontsize = 10)
#plt.yticks(np.arange(0,5001,600), fontsize = 10)
#a.set_yticklabels(['0', '600', '1200', '1800', '2400', '>3000'])
plt.grid(True)

plt.savefig("../../../public_html/ms_thesis_plots/event_frequency_2016_17_regional_no_overlap_bins.pdf")
plt.close(fig1)


#%%




