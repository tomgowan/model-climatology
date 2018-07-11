
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
import matplotlib.lines as mlines

inhouse = zeros((649,185))
ncar = zeros((798,185))
gfs = zeros((798,185))
hrrr = zeros((798,185))
nam3km = zeros((798,185))
sref_arw = zeros((798,185))
sref_nmb = zeros((798,185))



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

inhouse = data[0,:,:] 
ncar = data[1,:,:] 
gfs = data[2,:,:] 
hrrr = data[3,:,:]
nam3km = data[4,:,:]
sref_arw = data[5,:,:]
sref_nmb = data[6,:,:]

#%%


###############################################################################
########################### NCAR forecast vs. observed ########################
###############################################################################



regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,44, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)
    
    
    
    
i = 0
u = 0
end = 19   
precip = zeros((2000,185))

bins = np.arange(2.54,50, 2.54)
ncar_array = zeros((len(bins)-1, len(bins)-1))
ncar_array_norm = zeros((len(bins)-1, len(bins)-1))

for x in range(len(ncar[:,0])):
        for y in range(len(inhouse[:,0])):
                    ################### PACIFIC ###################            
                if ((regions[0,0] <= ncar[x,1] <= regions[0,1] and regions[0,2] <= ncar[x,2] <= regions[0,3]) or ###Sierra Nevada
                            
                    (regions[1,0] <= ncar[x,1] <= regions[1,1] and regions[1,2] <= ncar[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                    (ncar[x,1] >= regions[1,4] or ncar[x,2] <= regions[1,5])):
                    
                    
                    
#                    ################  INTERMOUNTAIN #################                                        
#                if ((regions[2,0] <= ncar[x,1] <= regions[2,1] and regions[2,2] <= ncar[x,2] <= regions[2,3]) or ## CO Rockies
#    
#                    (regions[3,0] <= ncar[x,1] <= regions[3,1] and regions[3,2] <= ncar[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
#                    (ncar[x,1] >= regions[3,4] or ncar[x,2] <= regions[3,5]) and 
#                    (ncar[x,1] <= regions[3,6] or ncar[x,2] >= regions[3,7]) or
#                        
#                    (regions[4,0] <= ncar[x,1] <= regions[4,1] and regions[4,2] <= ncar[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
#                    (ncar[x,1] >= regions[4,4] or ncar[x,2] >= regions[4,5]) and 
#                    (ncar[x,1] >= regions[4,6] or ncar[x,2] <= regions[4,7]) or
#                        
#                    (regions[5,0] <= ncar[x,1] <= regions[5,1] and regions[5,2] <= ncar[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
#                    (ncar[x,1] <= regions[5,4] or ncar[x,2] <= regions[5,5])):
                    
                    if ncar[x,0] == inhouse[y,0]:
                        print ncar[x,0]
                        print inhouse[y,0]
                    
                        ### Just to investigate data
                        precip[i,:] = ncar[x,:]
                        precip[i+1,:] = inhouse[y,:]
                        i = i +2


u = 0               


for x in range(len(ncar[:,0])):
    for y in range(len(inhouse[:,0])):
        ################### PACIFIC ################### 
        if ((regions[0,0] <= ncar[x,1] <= regions[0,1] and regions[0,2] <= ncar[x,2] <= regions[0,3]) or ###Sierra Nevada
                            
            (regions[1,0] <= ncar[x,1] <= regions[1,1] and regions[1,2] <= ncar[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
            (ncar[x,1] >= regions[1,4] or ncar[x,2] <= regions[1,5])):
                    
                    
                    
#            ################  INTERMOUNTAIN #################                                        
#        if ((regions[2,0] <= ncar[x,1] <= regions[2,1] and regions[2,2] <= ncar[x,2] <= regions[2,3]) or ## CO Rockies
#    
#            (regions[3,0] <= ncar[x,1] <= regions[3,1] and regions[3,2] <= ncar[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
#            (ncar[x,1] >= regions[3,4] or ncar[x,2] <= regions[3,5]) and 
#            (ncar[x,1] <= regions[3,6] or ncar[x,2] >= regions[3,7]) or
#                        
#            (regions[4,0] <= ncar[x,1] <= regions[4,1] and regions[4,2] <= ncar[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
#            (ncar[x,1] >= regions[4,4] or ncar[x,2] >= regions[4,5]) and 
#            (ncar[x,1] >= regions[4,6] or ncar[x,2] <= regions[4,7]) or
#                        
#            (regions[5,0] <= ncar[x,1] <= regions[5,1] and regions[5,2] <= ncar[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
#            (ncar[x,1] <= regions[5,4] or ncar[x,2] <= regions[5,5])):        
            if ncar[x,0] == inhouse[y,0]:
            

### Create forecat and observed array (forecast on x axis and oberved on y axis)     
                for z in range(3,185):
                            # Excludes bad data 
                            if all(data[1:,x,z] < 1000) and inhouse[y,z] < 1000:
                                for i in range(len(bins)-1):
                                    for j in range(len(bins)-1):
                                        if bins[i] - 1.27 <= ncar[x,z] < bins[i+1] -1.27 and bins[j] -1.27 <= inhouse[y,z] < bins[j+1]-1.27:

                                            ncar_array[j,i] = ncar_array[j,i] + 1

print('test')

### To nomalize the data
for i in range(len(bins)-1):
    for j in range(len(bins)-1):
        ncar_array_norm[j,i] = ncar_array[j,i]/((max(ncar_array[j,:])+max(ncar_array[:,i]))/2)











          





    
   
                       

###############################################################################
################### gfs forecast vs. observed ########################
###############################################################################

end = 19   
u = u + 1
gfs_array = zeros((len(bins)-1, len(bins)-1))
gfs_array_norm = zeros((len(bins)-1, len(bins)-1))
precip = zeros((2000,185))


for x in range(len(gfs[:,0])):
    for y in range(len(inhouse[:,0])):
        ################### PACIFIC ################### 
        if ((regions[0,0] <= gfs[x,1] <= regions[0,1] and regions[0,2] <= gfs[x,2] <= regions[0,3]) or ###Sierra Nevada
                            
            (regions[1,0] <= gfs[x,1] <= regions[1,1] and regions[1,2] <= gfs[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
            (gfs[x,1] >= regions[1,4] or gfs[x,2] <= regions[1,5])):
                    
                    
                    
#            ################  INTERMOUNTAIN #################                                        
#        if ((regions[2,0] <= gfs[x,1] <= regions[2,1] and regions[2,2] <= gfs[x,2] <= regions[2,3]) or ## CO Rockies
#    
#            (regions[3,0] <= gfs[x,1] <= regions[3,1] and regions[3,2] <= gfs[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
#            (gfs[x,1] >= regions[3,4] or gfs[x,2] <= regions[3,5]) and 
#            (gfs[x,1] <= regions[3,6] or gfs[x,2] >= regions[3,7]) or
#                        
#            (regions[4,0] <= gfs[x,1] <= regions[4,1] and regions[4,2] <= gfs[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
#            (gfs[x,1] >= regions[4,4] or gfs[x,2] >= regions[4,5]) and 
#            (gfs[x,1] >= regions[4,6] or gfs[x,2] <= regions[4,7]) or
#                        
#            (regions[5,0] <= gfs[x,1] <= regions[5,1] and regions[5,2] <= gfs[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
#            (gfs[x,1] <= regions[5,4] or gfs[x,2] <= regions[5,5])): 
            if gfs[x,0] == inhouse[y,0]:
            
                for z in range(3,185):
                    # Excludes bad data 
                    if all(data[1:,x,z] < 1000) and inhouse[y,z] < 1000:
                        for i in range(len(bins)-1):
                            for j in range(len(bins)-1):
                                if bins[i]-1.27 <= gfs[x,z] < bins[i+1]-1.27 and bins[j]-1.27 <= inhouse[y,z] < bins[j+1]-1.27:

                                    gfs_array[j,i] = gfs_array[j,i] + 1



### To nomalize the data
for i in range(len(bins)-1):
    for j in range(len(bins)-1):
        gfs_array_norm[j,i] = gfs_array[j,i]/((max(gfs_array[j,:])+max(gfs_array[:,i]))/2)


    
        
                       


###############################################################################
##################### HRRR forecast vs. observed ########################
###############################################################################

end = 19   
u = u + 1
hrrr_array = zeros((len(bins)-1, len(bins)-1))
hrrr_array_norm = zeros((len(bins)-1, len(bins)-1))
p_array = []
precip = zeros((2000,185))



for x in range(len(hrrr[:,0])):
    for y in range(len(inhouse[:,0])):
        ################### PACIFIC ################### 
        if ((regions[0,0] <= hrrr[x,1] <= regions[0,1] and regions[0,2] <= hrrr[x,2] <= regions[0,3]) or ###Sierra Nevada
                            
            (regions[1,0] <= hrrr[x,1] <= regions[1,1] and regions[1,2] <= hrrr[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
            (hrrr[x,1] >= regions[1,4] or hrrr[x,2] <= regions[1,5])):
                    
                    
                    
#            ################  INTERMOUNTAIN #################                                        
#        if ((regions[2,0] <= hrrr[x,1] <= regions[2,1] and regions[2,2] <= hrrr[x,2] <= regions[2,3]) or ## CO Rockies
#    
#            (regions[3,0] <= hrrr[x,1] <= regions[3,1] and regions[3,2] <= hrrr[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
#            (hrrr[x,1] >= regions[3,4] or hrrr[x,2] <= regions[3,5]) and 
#            (hrrr[x,1] <= regions[3,6] or hrrr[x,2] >= regions[3,7]) or
#                        
#            (regions[4,0] <= hrrr[x,1] <= regions[4,1] and regions[4,2] <= hrrr[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
#            (hrrr[x,1] >= regions[4,4] or hrrr[x,2] >= regions[4,5]) and 
#            (hrrr[x,1] >= regions[4,6] or hrrr[x,2] <= regions[4,7]) or
#                        
#            (regions[5,0] <= hrrr[x,1] <= regions[5,1] and regions[5,2] <= hrrr[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
#            (hrrr[x,1] <= regions[5,4] or hrrr[x,2] <= regions[5,5])): 

            if hrrr[x,0] == inhouse[y,0]:
            
                for z in range(3,185):
                    
                    # Excludes bad data 
                    if all(data[1:,x,z] < 1000) and inhouse[y,z] < 1000:
                        for i in range(len(bins)-1):
                            for j in range(len(bins)-1):
                                if bins[i]-1.27 <= hrrr[x,z] < bins[i+1]-1.27 and bins[j]-1.27 <= inhouse[y,z] < bins[j+1]-1.27:
                                   
                                    hrrr_array[j,i] = hrrr_array[j,i] + 1



### To nomalize the data
for i in range(len(bins)-1):
    for j in range(len(bins)-1):
        hrrr_array_norm[j,i] = hrrr_array[j,i]/((max(hrrr_array[j,:])+max(hrrr_array[:,i]))/2)









###############################################################################
#################### NAM3k forecast vs. observed ########################
###############################################################################
precip = zeros((2000,185))
end = 19
u = u + 1
nam3km_array = zeros((len(bins)-1, len(bins)-1))
nam3km_array_norm = zeros((len(bins)-1, len(bins)-1))


for x in range(len(nam3km[:,0])):
    for y in range(len(inhouse[:,0])):
        ################### PACIFIC ################### 
        if ((regions[0,0] <= nam3km[x,1] <= regions[0,1] and regions[0,2] <= nam3km[x,2] <= regions[0,3]) or ###Sierra Nevada
                            
            (regions[1,0] <= nam3km[x,1] <= regions[1,1] and regions[1,2] <= nam3km[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
            (nam3km[x,1] >= regions[1,4] or nam3km[x,2] <= regions[1,5])):
                    
                    
                    
#            ################  INTERMOUNTAIN #################                                        
#        if ((regions[2,0] <= nam3km[x,1] <= regions[2,1] and regions[2,2] <= nam3km[x,2] <= regions[2,3]) or ## CO Rockies
#    
#            (regions[3,0] <= nam3km[x,1] <= regions[3,1] and regions[3,2] <= nam3km[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
#            (nam3km[x,1] >= regions[3,4] or nam3km[x,2] <= regions[3,5]) and 
#            (nam3km[x,1] <= regions[3,6] or nam3km[x,2] >= regions[3,7]) or
#                        
#            (regions[4,0] <= nam3km[x,1] <= regions[4,1] and regions[4,2] <= nam3km[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
#            (nam3km[x,1] >= regions[4,4] or nam3km[x,2] >= regions[4,5]) and 
#            (nam3km[x,1] >= regions[4,6] or nam3km[x,2] <= regions[4,7]) or
#                        
#            (regions[5,0] <= nam3km[x,1] <= regions[5,1] and regions[5,2] <= nam3km[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
#            (nam3km[x,1] <= regions[5,4] or nam3km[x,2] <= regions[5,5])): 
            if nam3km[x,0] == inhouse[y,0]:
                for z in range(3,185):
                    
                    # Excludes bad data 
                    if all(data[1:,x,z] < 1000) and inhouse[y,z] < 1000:
                        for i in range(len(bins)-1):
                            for j in range(len(bins)-1):
                                if bins[i]-1.27 <= nam3km[x,z] < bins[i+1]-1.27 and bins[j]-1.27 <= inhouse[y,z] < bins[j+1]-1.27:
                                    nam3km_array[j,i] = nam3km_array[j,i] + 1
                                            


### To nomalize the data
for i in range(len(bins)-1):
    for j in range(len(bins)-1):
        nam3km_array_norm[j,i] = nam3km_array[j,i]/((max(nam3km_array[j,:])+max(nam3km_array[:,i]))/2)
        



###############################################################################
#################### sref_arw forecast vs. observed ########################
###############################################################################
precip = zeros((2000,185))
end = 19
u = u + 1
sref_arw_array = zeros((len(bins)-1, len(bins)-1))
sref_arw_array_norm = zeros((len(bins)-1, len(bins)-1))


for x in range(len(sref_arw[:,0])):
    for y in range(len(inhouse[:,0])):
        ################### PACIFIC ################### 
        if ((regions[0,0] <= sref_arw[x,1] <= regions[0,1] and regions[0,2] <= sref_arw[x,2] <= regions[0,3]) or ###Sierra Nevada
                            
            (regions[1,0] <= sref_arw[x,1] <= regions[1,1] and regions[1,2] <= sref_arw[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
            (sref_arw[x,1] >= regions[1,4] or sref_arw[x,2] <= regions[1,5])):
                    
                    
                    
#            ################  INTERMOUNTAIN #################                                        
#        if ((regions[2,0] <= sref_arw[x,1] <= regions[2,1] and regions[2,2] <= sref_arw[x,2] <= regions[2,3]) or ## CO Rockies
#    
#            (regions[3,0] <= sref_arw[x,1] <= regions[3,1] and regions[3,2] <= sref_arw[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
#            (sref_arw[x,1] >= regions[3,4] or sref_arw[x,2] <= regions[3,5]) and 
#            (sref_arw[x,1] <= regions[3,6] or sref_arw[x,2] >= regions[3,7]) or
#                        
#            (regions[4,0] <= sref_arw[x,1] <= regions[4,1] and regions[4,2] <= sref_arw[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
#            (sref_arw[x,1] >= regions[4,4] or sref_arw[x,2] >= regions[4,5]) and 
#            (sref_arw[x,1] >= regions[4,6] or sref_arw[x,2] <= regions[4,7]) or
#                        
#            (regions[5,0] <= sref_arw[x,1] <= regions[5,1] and regions[5,2] <= sref_arw[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
#            (sref_arw[x,1] <= regions[5,4] or sref_arw[x,2] <= regions[5,5])): 
            if sref_arw[x,0] == inhouse[y,0]:
                for z in range(3,185):
                    
                    # Excludes bad data 
                    if all(data[1:,x,z] < 1000) and inhouse[y,z] < 1000:
                        for i in range(len(bins)-1):
                            for j in range(len(bins)-1):
                                if bins[i]-1.27 <= sref_arw[x,z] < bins[i+1]-1.27 and bins[j]-1.27 <= inhouse[y,z] < bins[j+1]-1.27:
                                    sref_arw_array[j,i] = sref_arw_array[j,i] + 1
                                            


### To nomalize the data
for i in range(len(bins)-1):
    for j in range(len(bins)-1):
        sref_arw_array_norm[j,i] = sref_arw_array[j,i]/((max(sref_arw_array[j,:])+max(sref_arw_array[:,i]))/2)
        
        
        
        

###############################################################################
#################### sref_nmb forecast vs. observed ########################
###############################################################################
precip = zeros((2000,185))
end = 19
u = u + 1
sref_nmb_array = zeros((len(bins)-1, len(bins)-1))
sref_nmb_array_norm = zeros((len(bins)-1, len(bins)-1))


for x in range(len(sref_nmb[:,0])):
    for y in range(len(inhouse[:,0])):
                ################### PACIFIC ################### 
        if ((regions[0,0] <= sref_nmb[x,1] <= regions[0,1] and regions[0,2] <= sref_nmb[x,2] <= regions[0,3]) or ###Sierra Nevada
                            
            (regions[1,0] <= sref_nmb[x,1] <= regions[1,1] and regions[1,2] <= sref_nmb[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
            (sref_nmb[x,1] >= regions[1,4] or sref_nmb[x,2] <= regions[1,5])):
                    
                    
                    
#            ################  INTERMOUNTAIN #################                                        
#        if ((regions[2,0] <= sref_nmb[x,1] <= regions[2,1] and regions[2,2] <= sref_nmb[x,2] <= regions[2,3]) or ## CO Rockies
#    
#            (regions[3,0] <= sref_nmb[x,1] <= regions[3,1] and regions[3,2] <= sref_nmb[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
#            (sref_nmb[x,1] >= regions[3,4] or sref_nmb[x,2] <= regions[3,5]) and 
#            (sref_nmb[x,1] <= regions[3,6] or sref_nmb[x,2] >= regions[3,7]) or
#                        
#            (regions[4,0] <= sref_nmb[x,1] <= regions[4,1] and regions[4,2] <= sref_nmb[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
#            (sref_nmb[x,1] >= regions[4,4] or sref_nmb[x,2] >= regions[4,5]) and 
#            (sref_nmb[x,1] >= regions[4,6] or sref_nmb[x,2] <= regions[4,7]) or
#                        
#            (regions[5,0] <= sref_nmb[x,1] <= regions[5,1] and regions[5,2] <= sref_nmb[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
#            (sref_nmb[x,1] <= regions[5,4] or sref_nmb[x,2] <= regions[5,5])): 
            if sref_nmb[x,0] == inhouse[y,0]:
                for z in range(3,185):
                    
                    # Excludes bad data 
                    if all(data[1:,x,z] < 1000) and inhouse[y,z] < 1000:
                        for i in range(len(bins)-1):
                            for j in range(len(bins)-1):
                                if bins[i]-1.27 <= sref_nmb[x,z] < bins[i+1]-1.27 and bins[j]-1.27 <= inhouse[y,z] < bins[j+1]-1.27:
                                    sref_nmb_array[j,i] = sref_nmb_array[j,i] + 1
                                                


### To nomalize the data
for i in range(len(bins)-1):
    for j in range(len(bins)-1):
        sref_nmb_array_norm[j,i] = sref_nmb_array[j,i]/((max(sref_nmb_array[j,:])+max(sref_nmb_array[:,i]))/2)        
        
        
        
        
        


###### Change to frequencies
array = zeros((len(links)-1,18,18))
array[0,:,:] = ncar_array
array[1,:,:] = gfs_array
array[2,:,:] = hrrr_array
array[3,:,:] = nam3km_array
array[4,:,:] = sref_arw_array
array[5,:,:] = sref_nmb_array

for i in range(len(links)-1):
    tot = np.sum(array[i])
    array[i] = array[i]/tot
  
#####  Determine most likely forecasted amounts for each snotel value
#array_32 = zeros((4,18,18))
array2 = array
array3 = array


for i in range(len(links)-1):
    for j in range(18):
        #m = np.percentile(array2[i,:,j],95)
        m = max(array2[i,:,j])
        for t in range(18):
            if m == array2[i,t,j]:
                pass
            else:
                array2[i,t,j] = 0

array3 = zeros((len(links)-1,18,18))
array3[0,:,:] = ncar_array
array3[1,:,:] = gfs_array
array3[2,:,:] = hrrr_array
array3[3,:,:] = nam3km_array
array3[4,:,:] = sref_arw_array
array3[5,:,:] = sref_nmb_array

for i in range(len(links)-1):
    for j in range(18):
        #m = np.percentile(array2[i,:,j],95)
        m = max(array3[i,j,:])
        for t in range(18):
            if m == array3[i,j,t]:
                pass
            else:
                array3[i,j,t] = 0 
                
array2 = array2+array3
                
#%%
###### Change to frequencies
array4 = zeros((len(links),18,18))
array4[0,:,:] = ncar_array
array4[1,:,:] = gfs_array
array4[2,:,:] = hrrr_array
array4[3,:,:] = nam3km_array
array4[4,:,:] = sref_arw_array
array4[5,:,:] = sref_nmb_array

for i in range(len(links)-1):
    tot = np.sum(array4[i])
    array4[i] = array4[i]/tot
    
    
array5 = array4

np.save('heatmaps_pacific', array5)    

'''
   

###############################################################################
#########################  PLots   ############################################
###############################################################################


###############################################################################
#####  Code found online to discritize any color map ##########################
###############################################################################

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)




model = ['NCAR Ensemble Control', 'HRRR', 'NAM-4km','NAM-12km', 'GFS'] 
labels = np.arange(2.54,60,2.54)
x = np.arange(0,20,2)
y = np.arange(0,20,2)
fig1=plt.figure(num=None, figsize=(18,12), dpi=500, facecolor='w', edgecolor='k')
ax2 = fig1.add_axes([0.025,0.12,1.08,0.78])
ax2.axis('off')
plot = 330
plot_loc = [232,233,234,235,236]
#array = [ncar_array, gfs_array, hrrr_array, nam3km_array]

for i in range(5):
    g = array4[i]
    #ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    #ax1.xaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    print i
    plot = plot_loc[i]
    ax1 = fig1.add_subplot(plot)
    plt.title(model[i], fontsize = 22)
    #ax1.tick_params(axis='x', which='both', bottom='on', labelbottom='on') 
    plt.grid(True)  
    ##plt.tick_params(axis='y', which='both', bottom='on')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if i == 2 or i == 3 or i == 4:

        ax1.set_xlabel('Forecasted 24-hour Events (mm)', fontsize  = 18, labelpad = 12)
    if i == 0 or i == 2:

        ax1.set_ylabel('Observed 24-hour Events (mm)', fontsize  = 18, labelpad = 12)
    if i == 2 or i == 3 or i == 4:
        ax1.set_xticklabels([(i*2.54)+1.27 for i in x], fontsize  = 16, rotation = 60)
    else:
        ax1.set_xticklabels([])
    if i == 2 or i == 0 or i == 4:
        ax1.set_yticklabels([(i*2.54)+1.27 for i in y], fontsize = 16)
    else:
        ax1.set_yticklabels([])

    if plot == 234: 
        ax1.set_yticklabels([(i*2.54)+1.27 for i in y], fontsize = 16)
 


   
    heatmap = ax1.pcolor(g, vmin=0.00005, vmax=.1, cmap=discrete_cmap(12, 'jet'), norm=matplotlib.colors.LogNorm())

    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]

    # now plot both limits against eachother
    ax1.plot(lims, lims, 'k', alpha=1, zorder=1)
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
formatter = LogFormatter(10)#, labelOnlyBase=False) 
#cbar = plt.colorbar(heatmap, ax = ax2, ticks=[1,1.77,3.16,10**.75, 10, 10**1.25, 10**1.5, 10**1.75, 10**2, 10**2.25, 10**2.5, 10**2.75, 10**3],format = formatter)
ticks = np.logspace(1, 0.232503, num = 13, base = 0.00005)
cbar = plt.colorbar(heatmap, ax = ax2,ticks = ticks,format = formatter)
cbar.ax.set_yticklabels(['0.00005', '0.00009', '0.0002', '0.0003','0.0006','0.001','0.002','0.004','0.008','0.015','0.03','0.05','0.1'], fontsize = 16)
cbar.ax.set_xlabel('\nFrequency', fontsize  = 16)
#plt.tight_layout()
plt.savefig("../../public_html/heatmaps_interp.pdf", bbox_inches='tight')
plt.show()




'''

#%%

###############################################################################
#########################  PLots   ############################################
###############################################################################


###############################################################################
#####  Code found online to discritize any color map ##########################
###############################################################################

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


xlab = np.arange(1.27, 100, 2.54)
xlab = np.round(xlab, decimals = 1)

model = ['NCAR ENS CTL', 'GFS', 'HRRR','NAM-3km', 'SREF ARW CTL', 'SREF NMB CTL'] 
labels = np.arange(2.54,60,2.54)
x = np.arange(0,20,1)
y = np.arange(0,20,1)
fig1=plt.figure(num=None, figsize=(9.5,14), dpi=500, facecolor='w', edgecolor='k')
ax2 = fig1.add_axes([0.0,0.19,1.2,0.62])
#ax2 = fig1.add_axes([0,0.7,1,7])
plt.axis('off')
plot = 220
#array = [ncar_array, gfs_array, hrrr_array, nam3km_array]

for i in range(6):
    g = array4[i]

    print i
    plot = 321 + i
    ax1 = fig1.add_subplot(plot)
    ax1.set_xticks(x)
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1 = fig1.add_subplot(plot)
    plt.title(model[i], fontsize = 18)

    if i == 4 or i == 5:
        ax1.set_xlabel('Forecasted Event Size (mm)', fontsize  = 16, labelpad = 12)
    if i == 0 or i == 2 or i == 4:
        ax1.set_ylabel('Observed Event Size (mm)', fontsize  = 16, labelpad = 12)



    if i == 0 or i == 2:
        ax1.set_yticklabels(xlab, fontsize = 14)
        for label in ax1.get_yticklabels()[1::2]:
            label.set_visible(False)
    else:
        ax1.set_yticklabels([])
        
    if i == 5:
        ax1.set_xticklabels(xlab, fontsize  = 14, rotation = 45)
        for label in ax1.get_xticklabels()[1::2]:
            label.set_visible(False)
    else:
        ax1.set_xticklabels([])


    if i == 4:
        ax1.set_xticklabels(xlab, fontsize  = 14, rotation = 45)
        for label in ax1.get_xticklabels()[1::2]:
            label.set_visible(False)
        ax1.set_yticklabels(xlab, fontsize = 14)
        for label in ax1.get_yticklabels()[1::2]:
            label.set_visible(False)
            
    cmap=discrete_cmap(13, 'jet')
    cmap.set_bad(color='navy')
    heatmap = ax1.pcolor(g, vmin=0.00005, vmax=.1, cmap=cmap, norm=matplotlib.colors.LogNorm())

    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]

    # now plot both limits against eachother
    ax1.plot(lims, lims, 'k', alpha=1, zorder=1)
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
formatter = LogFormatter(10)#, labelOnlyBase=False) 
#cbar = plt.colorbar(heatmap, ax = ax2, ticks=[1,1.77,3.16,10**.75, 10, 10**1.25, 10**1.5, 10**1.75, 10**2, 10**2.25, 10**2.5, 10**2.75, 10**3],format = formatter)
ticks = np.logspace(1, 0.232503, num = 14, base = 0.00005)
cbar = plt.colorbar(heatmap, ax = ax2,ticks = ticks,format = formatter, orientation='vertical')
cbar.ax.set_yticklabels([' ', '<0.009', '0.02', '0.03','0.06','0.1','0.2','0.4','0.8','1.5','3','5','>10'], fontsize = 14)
plt.tight_layout()
cbar.ax.set_xlabel('\nFrequency \nof Event (%)', fontsize  = 14)
plt.savefig("../../../public_html/heatmaps_2016_17_pacific.pdf", bbox_inches='tight')
plt.show()

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
import matplotlib.lines as mlines

####   Ideal example plots

ideal = zeros((18,18))
for i in range(18):
    ideal[i,i] = 9-.5*i
    if i > 0:
        ideal[i,i-1] = 9-.5*i
        ideal[i-1,i] = 9-.5*i
        
notideal = zeros((18,18))
for i in range(18):
        notideal[17-i,:] = 19-np.arange(1,162,9) +10*i



x = np.arange(0,20,1)
y = np.arange(0,20,1)
fig1=plt.figure(num=None, figsize=(12,12), dpi=500, facecolor='w', edgecolor='k')
ax2 = fig1.add_axes([.11,0.335,.97,0.355])
plt.axis('off')
plot = 220
#array = [ncar_array, gfs_array, hrrr_array, nam3km_array]
model = ['High Accuracy', 'Low Accuracy'] 

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


#colors = [
# (235, 246, 255),
# (214, 226, 255),
# (181, 201, 255),
# (142, 178, 255),
# (127, 150, 255),
# (114, 133, 248),
# (99, 112, 248),
# (0, 158,  30),
# (60, 188,  61),
# (179, 209, 110),
# (185, 249, 110),
# (255, 249,  19),
# (255, 163,   9),
# (229,   0,   0),
# (189,   0,   0),
# (129,   0,   0),
# (0,   0,   0)
# ]
colors = [
 (215, 227, 238),
 (181, 202, 255),
 (143, 179, 255),
 (127, 151, 255),
 (171, 207,  99),
 (232, 245, 158),
 (255, 250,  20),
 (255, 209,  33),
 (255, 163,  10),
 (255,  76,   0),
 ]
#colors = [
# (255, 255, 255),
# (237, 250, 194),
# (205, 255, 205),
# (153, 240, 178),
#  (83, 189, 159),
#  (50, 166, 150),
#  (50, 150, 180),
#   (5, 112, 176),
#   (5,  80, 140),
#  (10,  31, 15),
#  (44,   2,  70),
# (106,  44,  90),
#]
cmap = make_cmap(colors, bit=True)


for i in range(2):


    print i
    plot = 121 + i
    ax1 = fig1.add_subplot(plot)
    plt.title(model[i], fontsize = 23, y = 1.03)

    if i == 1 or i ==0:
        ax1.set_xlabel('Forecasted Event Size', fontsize  = 20, labelpad = 15)


    if i == 0:
        ax1.set_ylabel('Observed Event Size', fontsize  = 20, labelpad = 15)

    #ax1.set_xticks(x)
    #ax1.set_yticks(y)
    #ax1.set_xticklabels([(i*2.54)+1.27 for i in x], fontsize  = 10, rotation = 45)
    #ax1.set_yticklabels([(i*2.54)+1.27 for i in y], fontsize = 10)

    
    if i == 0:
        heatmap = ax1.pcolor(ideal, vmin=1, vmax=9, cmap=cmap, norm=matplotlib.colors.LogNorm())
    if i == 1:
        heatmap = ax1.pcolor(notideal, vmin = 20, vmax=170, cmap=cmap, norm=matplotlib.colors.LogNorm())
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])


    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]
    # now plot both limits against eachother
    ax1.plot(lims, lims, 'k', alpha=1, zorder=1)
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)

formatter = LogFormatter(10)#, labelOnlyBase=False) 
#cbar = plt.colorbar(heatmap, ax = ax2, ticks=[1,1.77,3.16,10**.75, 10, 10**1.25, 10**1.5, 10**1.75, 10**2, 10**2.25, 10**2.5, 10**2.75, 10**3],format = formatter)
ticks = np.logspace(1, 0.232503, num = 13, base = 0.00005)
cbar = plt.colorbar(heatmap, ax = ax2,ticks = ticks,format = formatter)
#cbar.ax.set_yticklabels(['0.00005', '0.00009', '0.0002', '0.0003','0.0006','0.001','0.002','0.004','0.008','0.015','0.03','0.05','0.1'], fontsize = 12)

#cbar.ax.set_xlabel('\nFrequency', fontsize  = 19)
plt.savefig("../../../public_html/heatmaps_examples.pdf", bbox_inches='tight')
plt.show()



