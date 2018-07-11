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
Date = zeros((152))
frequency = zeros((80,5))
frequency_snotel = zeros((80,2))
daily_snotel_precip = zeros((125000))
snotel_rowloc = zeros((798))
snotel_colloc = zeros((798))






##### Read in SNOTEL data

i = 0
#j = 0
#k = 0 

with open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/all_nov2015_march2016_english.txt", "rt") as f:
    for line in f:
        snotel_precip = float(line.split(',')[8])
        daily_snotel_precip[i] = snotel_precip*25.4
        
        i = i + 1
        print i
'''
        snotel_lat1 = float(line.split(',')[5])
        snotel_lat1 = round(snotel_lat1, 3)
        
        snotel_lon1 = float(line.split(',')[6])
        snotel_lat1 = round(snotel_lat1,3)
        
        
        if snotel_lat1 != snotel_lat2[j-1]:
            snotel_lat2[j] = snotel_lat1
            j = j + 1

        if snotel_lon1 != snotel_lon2[k-1]:
            snotel_lon2[k] = snotel_lon1
            k = k + 1
 '''         
            
        
        


#### Read in snotel lat and lon locations for ncarens
i = 0
with open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/snotel_ncarens_precip.txt", "rt") as f:
    for line in f:
        snotel_precip = line.split(',')[2]
        totalprecip[i] = snotel_precip
        i = i + 1
        

### Read in snotel precip data from ncarens  
        
i = 0
with open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/snotel_ncarens_latlonloc.txt", "rt") as f:
    for line in f:
        snotel_row = line.split(',')[2]
        snotel_col= line.split(',')[3]
        
        snotel_rowloc[i] = snotel_row
        snotel_colloc[i] = snotel_col
        
        i = i + 1
        

### Read in NCAR data

NCARens_file = '/uufs/chpc.utah.edu/common/home/horel-group/archive/20160222/models/ncarens/ncar_3km_westus_2016022200_mem9_f047.nc'
fhcoord = Dataset(NCARens_file, mode='r')

lonNCAR = fhcoord.variables['gridlon_0'][:]
latNCAR = fhcoord.variables['gridlat_0'][:]

#### Loop over all lats and lons

'''
for w in range(0,797)

    for i in range(0,785):
        for j in range(0,650):
            nearestNCAR[i,j] = abs(lonNCAR[i,j]-snotel_lon2[w]) + abs(latNCAR[i,j]-snotel_lat2[w])
        
    locationNCAR =  np.where(nearestNCAR == nearestNCAR.min())
        
    row[w] = locationNCAR[0]
    col[w] = locationNCAR[1]
'''




Date2= '20151101'


for i in range(0,152):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date[i] = int(Date3)

x = 24


##### there are 151 days here...all days from start of november to end of march except for 2/2/2016

##Hour 8 to 31 captures midnight to midnight
'''
f = open("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/SNOTEL/snotel_ncarens_precip.txt", "w")
for w in range(0,50):
    for i in range(0,152):
        

    
        if x != 24:
            totalprecip[i-1,0] = 99999
            x = 1
   
        for j in range(8,9):
            NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/ncarens/ncar_3km_westus_%08d' % Date[i] + '00_mem1_f0%02d.nc' %  j
            if os.path.exists(NCARens_file):

               # print(NCARens_file) 
                print w
                fh = Dataset(NCARens_file, mode='r')
                
                row = int(snotel_rowloc[w])
                col = int(snotel_colloc[w])
            
                if j == 1:
                    precipI = fh.variables['APCP_P8_L1_GLC0_acc'][row, col]
                    precipNCAR = precipI*0.0393689*25.4

                else:
                    precipI = fh.variables['APCP_P8_L1_GLC0_acc1h'][row, col]
                    precipNCAR = precipI*0.0393689*25.4


                totalprecip[i,w] = totalprecip[i,w]+precipNCAR
        
                x = x + 1
 
        f.write(str(row))
        f.write(",")
        f.write(str(col))
        f.write(",")
        f.write(str(totalprecip[i,w]))
        f.write("\n")

    

      
fh.close()
f.close()
'''



## NCAR frequency of event

j = 0
for x in range(1,80):
    x = x*2.54
    y = x-2.54
    frequency[j,2] = x-1.27
    for i in range(0,121125):
        if totalprecip[i] > y and totalprecip[i] <= x:
            frequency[j,0] = frequency[j,0] + 1
    j = j + 1
    
    
    
### SNOTEL frequency of event
j = 0
for x in range(1,80):
    x = x*2.54
    y = x-2.54
    for i in range(0,121125):
        if daily_snotel_precip[i] > y and daily_snotel_precip[i] <= x:
            frequency[j,1] = frequency[j,1] + 1
    j = j + 1


sum_ncar = sum(frequency[:,0])
sum_snotel = sum(frequency[:,1])

for i in range(0,80):
 

   frequency[i,0]= frequency[i,0]/sum_ncar*100
   frequency[i,1]= frequency[i,1]/sum_snotel*100

    







#fig1 = plt.figure()

fig1=plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig1.add_subplot(111)
plt.xlim([3,50])
plt.xticks(np.arange(3,75,3))
plt.ylim([.03,50])
#ax1.set_yscale('log')

plt.grid(True)
ax1.plot(frequency[:,2],frequency[:,0:2])
#,['50','30','20','10','5','1','0.75','0.5','0.2','0.1'])

ax1.set_yscale('log')
ax1.set_yticks([50,20,10,5,2,1,0.5,0.2,0.1,0.05])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.2f'))
blue_line = mlines.Line2D([], [], color='blue',
                           label='NCAR Ensemble')
green_line = mlines.Line2D([], [], color='green',
                           label='SNOTEL')
plt.legend(handles=[blue_line, green_line])
plt.title('Daily Precipitation Frequency Diagram', fontsize = 16)
plt.xlabel('24-hour Precipitation (mm)')
plt.ylabel('Frequncy (%)')
plt.savefig("ncarens_snotel_validation.pdf")
plt.show()














