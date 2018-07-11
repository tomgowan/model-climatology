
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, maskoceans
import pygrib, os, sys
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta
import pyart
from matplotlib import animation
import matplotlib.animation as animation
import types
import matplotlib.lines as mlines


###############################################################################
##############   Read in hrrr  and prism precip   #############################
###############################################################################
Date2= '20150930'
Date = zeros((184))
precip_gfs = zeros((621,1405))
precip_tot = zeros((621,1405))
num_days = 184

for i in range(0,num_days):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date[i] = int(Date3)  


y = 0
for i in range(0,num_days-1): 
    x = 0
    z = 0


    #### Make sure all nam12k and prism files are present
    for r in range(3,7):
        j = r*6
        
        ### GFS files regridded to PRISM grid
        try:
            gfs_file1 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_raw_output/gfs004/regridded.precip.%08d' % Date[i] + '00F0%02d' % j + 'gfs004.grib2'
            if os.path.exists(gfs_file1):
                x = x + 1
        except:
            pass
        
        try:
            gfs_file2 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_raw_output/gfs004/regridded.precip.gfs_4_%08d' % Date[i] + '_0000_0%02d' % j + '.grb2'
            if os.path.exists(gfs_file2):
                x = x + 1
        except:
            pass
    
    ### PRISM Files    
    try:
        prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_stable_4kmD2_%08d' % Date[i] + '_asc.asc'
        if os.path.exists(prism_file):
            z = 1
    except:
        pass
    
    try:
        prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_provisional_4kmD2_%08d' % Date[i] + '_asc.asc'
        if os.path.exists(prism_file):
            z = 1
    except:
        pass


     
    if x == 4 and z == 1:
        y = y + 1
         
        for r in range(3,7):
            j = r*6
            
            try:
                gfs_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_raw_output/gfs004/regridded.precip.%08d' % Date[i] + '00F0%02d' % j + 'gfs004.grib2'
                grbs = pygrib.open(gfs_file)
            except:
                pass
            try:
                gfs_file ='/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_raw_output/gfs004/regridded.precip.gfs_4_%08d' % Date[i] + '_0000_0%02d' % j + '.grb2'
                grbs = pygrib.open(gfs_file)
            except: 
                pass
                
            
            print gfs_file
            grb = grbs.select(name='Total Precipitation')[0]
            lat_gfs,lon_gfs = grb.latlons()
            
         
            msg = grbs[1]
            gfs_precipi = (msg.values)*0.0393689*25.4
            precip_gfs = precip_gfs + gfs_precipi
      



        ############### Prism #####################################
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_stable_4kmD2_%08d" % Date[i+1] + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_provisional_4kmD2_%08d" % Date[i+1] + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        
        precip_tot = precip_tot + precip
    


print y   
precip_tot = precip_tot/y
precip_gfs = precip_gfs/y
## Attempt to fix notation of lons so basemap understands it




###############################################################################
##############   Create lat lon grid for psirm    #############################
###############################################################################




lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667


##################   Saveprism and ncar array  ################################

np.savetxt('gfs_dailymean.txt', precip_gfs)
np.savetxt('prism_gfs_dailymean.txt', precip_tot)



