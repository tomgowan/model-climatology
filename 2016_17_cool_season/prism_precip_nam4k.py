
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
precip_nam4k = zeros((621,1405))
#precip_hrrr = zeros((1033,842))
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

    #### Make sure all ncar and prism files are present
    for t in range(1,9):
        j = (t * 3) + 12
        try:
            nam4k_file1 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/nam4km/prism.regridded.p.%08d' % Date[i] + '00F%02d' % j + 'namhires.grib2'
            if os.path.exists(nam4k_file1):
                x = x + 1
        except:
            pass
    
    
    
    
    
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


     
    if x == 8 and z == 1:
        y = y + 1
         
        for x in range(1,9):
            j = (x * 3) + 12

            nam4k_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/nam4km/prism.regridded.p.%08d' % Date[i] + '00F%02d' % j + 'namhires.grib2'
      
            grbs = pygrib.open(nam4k_file)
            print nam4k_file
            grb = grbs.select(name='Total Precipitation')[0]
            lat_nam4k,lon_nam4k = grb.latlons()
            tmpmsgs = grbs.select(name='Total Precipitation')
            msg = grbs[1]
            print(msg)
            precip_vals = msg.values
            precip_vals = precip_vals*0.0393689*25.4
            precip_nam4k = precip_nam4k + precip_vals
            



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
precip_nam4k = precip_nam4k/y
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

np.savetxt('nam4k_dailymean.txt', precip_nam4k)
np.savetxt('prism_nam4k_dailymean.txt', precip_tot)

























