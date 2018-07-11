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

nearestNCAR = zeros((785,650))
totalprecip = zeros((125000))
totalprecip_hrrr = zeros((125000))
totalprecip_nam4k = zeros((125000))
Date = zeros((152))
frequency = zeros((80,8))
bias = zeros((80,8))
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


'''
a = 0
b = 0
for i in range(len(nam4k_data[609,:])):
    if nam4k_data[609,i] != 9999 and ncar_data[609,i] != 9999:
        a = a + nam4k_data[3:,i] 
        b = b + ncar_data[3:,i]
test = sum(a-b)
print test
'''

j = 0
print c
# print ranges in array
for x in range(1,30):
        x = x*2.54
        y = x-2.54
        frequency[j,c+1] = x+1.27
        bias[j,c+1] = x+1.27
        j = j + 1

## Frequency of event for each model
num = 0

for c in range(1,len(links)): # loop over snotel, ncar, nam4k, hrrr, nam12k, gfs
    print c
    j = 0
    for x in range(1,30):
        print x
        x = (x*2.54)+3.81
        y = x-5.08
     
        for z in range(len(data[c,:,0])):
            for f in range(len(data[0,:,0])):
                    if data[c,z,0] == data[0,f,0]:
                 
                        for r in range(3,185):
                     
                            if all(data[1:,z,r] < 1000) and data[0,f,r] < 1000:

                                if data[c,z,r] > y and data[c,z,r] <= x:
                                    frequency[j,c] = frequency[j,c] + 1
        j = j + 1

j = 0
## Frequency of event for snotel data
for x in range(1,30):
        x = (x*2.54)+3.81#6.35
        y = x-5.08#7.62
  
        for f in range(len(data[0,:,0])):
            for z in range(len(data[1,:,0])):
                if data[1,z,0] == data[0,f,0]:
                   
                    for r in range(3,185):
                        if all(data[1:,z,r] < 1000) and data[0,f,r] < 1000:  # Make sure all models have data for that day
                            if data[0,f,r] > y and data[0,f,r] <= x:
                                frequency[j,0] = frequency[j,0] + 1
        j = j + 1



sum_inhouse = sum(frequency[:,0])
sum_ncar = sum(frequency[:,1])
sum_gfs = sum(frequency[:,2])
sum_hrrr = sum(frequency[:,3])
sum_nam = sum(frequency[:,4])
sum_sref_arw = sum(frequency[:,5])
sum_sref_nmb = sum(frequency[:,6])

#### Calculate Frequencues



for i in range(0,80):
 
### POSSIBLY FOUND MAJOR FLAW HERE.  WAS DIVIEDING BY SUM OF NUMBER OF EVENTS FOR EACH MODEL
   frequency[i,0]= frequency[i,0]/sum_inhouse*100 
   frequency[i,1]= frequency[i,1]/sum_inhouse*100 
   frequency[i,2]= frequency[i,2]/sum_inhouse*100 
   frequency[i,3]= frequency[i,3]/sum_inhouse*100 
   frequency[i,4]= frequency[i,4]/sum_inhouse*100  
   frequency[i,5]= frequency[i,5]/sum_inhouse*100
   frequency[i,6]= frequency[i,6]/sum_inhouse*100

for i in range(0,80):
   
   bias[i,0] = frequency[i,1]/frequency[i,0] #NCARens
   bias[i,1] = frequency[i,2]/frequency[i,0] #gfs
   bias[i,2] = frequency[i,3]/frequency[i,0] #HRRRR
   bias[i,3] = frequency[i,4]/frequency[i,0] #nam
   bias[i,4] = frequency[i,5]/frequency[i,0] #sref_arw
   bias[i,5] = frequency[i,6]/frequency[i,0] #sef_nmb









##############  Bias Plots


#fig1 = plt.figure()

fig1=plt.figure(num=None, figsize=(11, 11), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.4, bottom = 0.1)
ax1 = fig1.add_subplot(211)
plt.xlim([0,50])
plt.xticks(np.arange(0,61,6))
ax1.set_xticks(np.arange(0,61,6))
ax1.set_xticklabels(np.arange(0,61,6), fontsize = 16)
plt.ylim([.03,60])
#ax1.set_yscale('log')
linecolor = ['k','blue', 'green', 'red', 'c', 'gold', 'magenta']

plt.gca().set_color_cycle(linecolor)
plt.grid(True)
ax1.plot(frequency[:,7],frequency[:,0:7],linewidth = 2, marker = "o", markeredgecolor = 'none')
#,['50','30','20','10','5','1','0.75','0.5','0.2','0.1'])

ax1.set_yscale('log')
ax1.set_yticks([60,35,20,10,5,2,1,0.5,0.2,0.1,0.05])
ax1.set_yticklabels(['60','35','20','10','5','2','1','0.5','0.2','0.1','0.05'], fontsize = 14)
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.2f'))
black = mlines.Line2D([], [], color='k',
                           label='SNOTEL', linewidth = 2,marker = "o", markeredgecolor = 'none')
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



plt.legend(handles=[black, blue, green, red, cyan, gold, magenta], fontsize = 12.5)
plt.title('Event Size Frequency', fontsize = 22)
plt.xlabel('24-hour Precipitation (mm)', fontsize = 17, labelpad = 10)
plt.ylabel('Frequency (%)', fontsize = 17, labelpad = 10)








ax2 = fig1.add_subplot(212)
plt.xlim([0,50])
plt.xticks(np.arange(0,80,6))



plt.grid(True)
line1 = ax2.plot(bias[:,7],bias[:,0])
line2 = ax2.plot(bias[:,7],bias[:,1])
line3 = ax2.plot(bias[:,7],bias[:,2])
line4 = ax2.plot(bias[:,7],bias[:,3])
line5 = ax2.plot(bias[:,7],bias[:,4])
line6 = ax2.plot(bias[:,7],bias[:,5])

x = np.linspace(0, 100, 100)
y = np.linspace(1,1,100)
line7 = ax2.plot(x,y)

plt.yticks(np.arange(0.2,2.5001,0.2))
ax2.set_yticklabels(np.arange(0.2,2.5001,0.2), fontsize = 16)
plt.ylim([.4,2.201])
ax2.set_xticklabels(np.arange(0, 80, 6), fontsize = 16)
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
                           label='SREF NMB CTL', linewidth = 2,marker = "o", markeredgecolor = 'none')
#plt.legend(handles=[blue, green, red, cyan], loc = 'upper left',bbox_to_anchor=(0.03, 1), fontsize = 16)
plt.title('Event Size Frequency Bias', fontsize = 22)
plt.xlabel('24-hour Precipitation (mm)', fontsize = 18, labelpad = 10)
plt.ylabel('Bias Ratio', fontsize = 18, labelpad = 10)

plt.savefig("../../../public_html/event_frequency_2016_17.pdf")
plt.show()














