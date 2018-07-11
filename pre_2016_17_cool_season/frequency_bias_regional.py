
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
'''
nearestNCAR = zeros((785,650))
totalprecip = zeros((125000))
totalprecip_hrrr = zeros((125000))
totalprecip_nam4k = zeros((125000))
Date = zeros((152))
frequency = zeros((8,80,7))
bias = zeros((8,80,7))
frequency_snotel = zeros((80,3))
daily_snotel_precip = zeros((125000))
snotel_rowloc = zeros((798))
snotel_colloc = zeros((798))



###############################################################################
############ Read in  12Z to 12Z data   #####################
###############################################################################

            
x = 0
q = 0
v = 0
i = 0   

         

links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/snotel_precip_2015_2016_qc.csv", 
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ncarens_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/nam4km_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/hrrrV1_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/nam12km_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/gfs_precip_12Zto12Z_interp.txt"]

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
nam4k_data = data[2,:,:]
hrrr_data = data[3,:,:]
nam12k_data = data[4,:,:]
gfs_data = data[5,:,:]






###############################################################################
##### Dvide into regionms#####################################################
###############################################################################

regions = np.array([[41.5,49.2, -123.0,-120.5],
                    [37.0,41.0, -121.0,-118.0], 
                    [43.7,46.2, -120.0,-116.8], 
                    [43.0,49.3, -116.8,-112.2], 
                    [41.8,47.0, -112.5,-105.5],
                    [37.2,41.8, -113.9,-109.2],
                    [35.6,41.5, -108.7,-104.5],
                    [32.5,35.5, -113.0,-107.0]])






        
j = 0
print c
# print ranges in array
for x in range(1,30):
        x = x*2.54
        y = x-2.54
        frequency[0,j,c+1] = x+1.27
        bias[0,j,c+1] = x+1.27
        j = j + 1

## Frequency of event for each model
num = 0
for w in range(8):
    for c in range(1,len(links)): # loop over snotel, ncar, nam4k, hrrr, nam12k, hrrrv2
        print c
        j = 0
        for x in range(1,30):

            x = (x*2.54)+3.81
            y = x-5.08
         
            for z in range(len(data[c,:,0])):
                for f in range(len(data[0,:,0])):
                    if data[c,z,0] == data[0,f,0]:
                        if regions[w,0] <= data[c,z,1] <= regions[w,1] and regions[w,2] <= data[c,z,2] <= regions[w,3]:
                            print x,y
                            for r in range(3,186):
                         
                                if all(data[1:,z,r] < 1000) and data[0,f,r] < 1000:
    
                                    if data[c,z,r] > y and data[c,z,r] <= x:
                                        frequency[w,j,c] = frequency[w,j,c] + 1
            j = j + 1

j = 0
## Frequency of event for snotel data
for w in range(8):
    j = 0
    for x in range(1,30):

            x = (x*2.54)+3.81
            y = x-5.08
      
            for f in range(len(data[0,:,0])):
                for z in range(len(data[1,:,0])):
                    if data[1,z,0] == data[0,f,0]:
                        if regions[w,0] <= data[0,z,1] <= regions[w,1] and regions[w,2] <= data[0,z,2] <= regions[w,3]:
                            print x,y
                            for r in range(3,186):
                                if all(data[1:,z,r] < 1000) and data[0,f,r] < 1000:  # Make sure all models have data for that day
                                    if data[0,z,r] > y and data[0,z,r] <= x:
                                        frequency[w,j,0] = frequency[w,j,0] + 1
            j = j + 1


for w in range(8):
    sum_inhouse = sum(frequency[w,:,0])
    sum_ncar = sum(frequency[w,:,1])
    sum_nam4k = sum(frequency[w,:,2])
    sum_hrrr = sum(frequency[w,:,3])
    sum_nam12k = sum(frequency[w,:,4])
    sum_gfs = sum(frequency[w,:,5])
    
    #### Calculate Frequencues



    for i in range(0,80):
 
         ### POSSIBLY FOUND MAJOR FLAW HERE.  WAS DIVIEDING BY SUM OF NUMBER OF EVENTS FOR EACH MODEL
        frequency[w,i,0]= frequency[w,i,0]/sum_inhouse*100 
        frequency[w,i,1]= frequency[w,i,1]/sum_inhouse*100 
        frequency[w,i,2]= frequency[w,i,2]/sum_inhouse*100 
        frequency[w,i,3]= frequency[w,i,3]/sum_inhouse*100 
        frequency[w,i,4]= frequency[w,i,4]/sum_inhouse*100  
        frequency[w,i,5]= frequency[w,i,5]/sum_inhouse*100

        for i in range(0,80):
   
           bias[w,i,0] = frequency[w,i,1]/frequency[w,i,0] #NCARens
           bias[w,i,1] = frequency[w,i,2]/frequency[w,i,0] #nam4k
           bias[w,i,2] = frequency[w,i,3]/frequency[w,i,0] #HRRRR
           bias[w,i,3] = frequency[w,i,4]/frequency[w,i,0] #nam12k
           bias[w,i,4] = frequency[w,i,5]/frequency[w,i,0] #gfs






np.savetxt('regional_frequency_bias1.txt', bias[0,:,:])
np.savetxt('regional_frequency_bias2.txt', bias[1,:,:])
np.savetxt('regional_frequency_bias3.txt', bias[2,:,:])
np.savetxt('regional_frequency_bias4.txt', bias[3,:,:])
np.savetxt('regional_frequency_bias5.txt', bias[4,:,:])
np.savetxt('regional_frequency_bias6.txt', bias[5,:,:])
np.savetxt('regional_frequency_bias7.txt', bias[6,:,:])
np.savetxt('regional_frequency_bias8.txt', bias[7,:,:])

'''

##############  Bias Plots
linecolor = ['blue', 'green', 'red', 'c', 'gold']
t = 0


plt.gca().set_color_cycle(linecolor)



region = ['Pacific Northwest', 'Sierra Nevada','Blue Mountains, OR','Idaho/Western MT','NW Wyoming','Utah','CO' ,'AZ/NM']                
x = np.arange(5,95.1,5)

fig1=plt.figure(num=None, figsize=(16,12), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=0.35, bottom = 0.2)
levels = [0.3,0.45, 0.7,  1,  1.4, 2,3,4]
levels_label = ['0.3','0.45' ,'0.7',  '1',  '1.4',  '2','3','4']
for i in range(1,9):
    plot = 420+i
    w = 4*i
    t = w - 4
    ax1 = fig1.add_subplot(plot)
    ax1.set_yscale('log')
    
    plt.xticks(np.arange(0,42,6))
    plt.xlim([0,42])
    ax1.set_xticks(np.arange(0,42,6))
    ax1.set_xticklabels(np.arange(0,43,6), fontsize = 14)
    plt.yticks(levels)#np.arange(0.2,4.001,0.6))
    ax1.set_yticklabels(levels_label, fontsize = 14)
    plt.ylim([0.3,4])
    
    plt.gca().set_color_cycle(linecolor)
    ax1.plot(bias[0,:28,6],bias[i-1,:28,:],linewidth = 2, marker = "o", markeredgecolor = 'none')
    
    plt.grid(True)
    blue_line = mlines.Line2D([], [],linewidth = 2, marker = "o", markeredgecolor = 'none', color='blue',
                           label='NCARens Control')
    green_line = mlines.Line2D([], [],linewidth = 2, marker = "o", markeredgecolor = 'none', color='green',
                           label='NAM-4km')
    red_line = mlines.Line2D([], [],linewidth = 2, marker = "o", markeredgecolor = 'none', color='red',
                           label='HRRR')
    cyan_line = mlines.Line2D([], [], linewidth = 2, marker = "o", markeredgecolor = 'none',color='c',
                           label='NAM-12km')
    gold_line = mlines.Line2D([], [], linewidth = 2, marker = "o", markeredgecolor = 'none',color='gold',
                           label='GFS')
    x = np.linspace(0, 100, 100)
    y = np.linspace(1,1,100)
    line5 = ax1.plot(x,y)
    plt.title(region[i-1], fontsize = 16)
    plt.setp(line5, color='k', linewidth=2) 
    
    if i == 1 or i == 3 or i == 5 or i == 7:
        plt.ylabel('Bias', fontsize = 16, labelpad = 13)

    if i == 8 or  i == 7:
        plt.xlabel('24-hour Precip. Event (Percentile)', fontsize = 16, labelpad = 13)

plt.legend(handles=[ blue_line, green_line, red_line, cyan_line, gold_line], loc='upper center', bbox_to_anchor=(-0.13, -0.4), 
           fancybox=True, shadow=True, ncol=5,fontsize = 16)
plt.savefig('../plots/short_range_frequency_regional_bias_interp.pdf')





'''

#fig1 = plt.figure()

fig1=plt.figure(num=None, figsize=(11, 11), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.4, bottom = 0.1)



ax2 = fig1.add_subplot(212)
plt.xlim([0,80])
plt.xticks(np.arange(0,80,6))



plt.grid(True)
line1 = ax2.plot(bias[:,5],bias[:,0])
line2 = ax2.plot(bias[:,5],bias[:,1])
line3 = ax2.plot(bias[:,5],bias[:,2])
line4 = ax2.plot(bias[:,5],bias[:,3])

x = np.linspace(0, 100, 100)
y = np.linspace(1,1,100)
line5 = ax2.plot(x,y)

plt.yticks(np.arange(0.2,2.5001,0.2))
ax2.set_yticklabels(np.arange(0.2,2.5001,0.2), fontsize = 16)
plt.ylim([.4,2])
ax2.set_xticklabels(np.arange(0, 80, 6), fontsize = 16)
#ax1.set_yticks([50,20,10,5,2,1,0.5,0.2,0.1,0.05])
#ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.2f'))

plt.setp(line1, color='blue', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line2, color='green', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line3, color='red', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line4, color='c', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line5, color='k', linewidth=2)
blue = mlines.Line2D([], [], color='blue',
                           label='NCAR Ens Control', linewidth = 2,marker = "o", markeredgecolor = 'none')
green = mlines.Line2D([], [], color='green',
                           label='NAM-4km', linewidth = 2,marker = "o", markeredgecolor = 'none')
red = mlines.Line2D([], [], color='red',
                           label='HRRR', linewidth = 2,marker = "o", markeredgecolor = 'none')
cyan = mlines.Line2D([], [], color='c',
                           label='NAM-12km', linewidth = 2,marker = "o", markeredgecolor = 'none')
#plt.legend(handles=[blue, green, red, cyan], loc = 'upper left',bbox_to_anchor=(0.03, 1), fontsize = 16)
plt.title('Event Size Frequency Bias', fontsize = 22)
plt.xlabel('24-hour Precipitation (mm)', fontsize = 18, labelpad = 10)
plt.ylabel('Bias Ratio', fontsize = 18, labelpad = 10)

plt.savefig("../../public_html/dailyprecip_bias_binned_interp.pdf")
plt.show()

'''











