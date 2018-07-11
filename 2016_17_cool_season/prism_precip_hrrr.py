import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
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



WM = [-117.0, 43.0, -108.5, 49.0]
UT = [-114.7, 36.7, -108.9, 42.5]
CO = [-110.0, 36.0, -104.0, 41.9]
NU = [-113.4, 39.5, -110.7, 41.9]
NW = [-125.3, 42.0, -116.5, 49.1]
WE = [-125.3, 31.0, -102.5, 49.2]
US = [-125, 24.0, -66.5, 49.5]
SN = [-123.5, 33.5, -116.0, 41.0]

region = sys.argv[1]

if region == 'WM':
    latlon = WM
    
if region == 'US':
    latlon = US
    
if region == 'UT':
    latlon = UT
    
if region == 'CO':
    latlon = CO
    
if region == 'NU':
    latlon = NU
    
if region == 'NW':
    latlon = NW

if region == 'WE':
    latlon = WE

if region == 'SN':
    latlon = SN

###############################################################################
##############   Read in hrrr  and prism precip   #############################
###############################################################################
Date2= '20150930'
Date = zeros((184))
precip_hrrr = zeros((621,1405))
#precip_hrrr = zeros((1033,842))
precip_tot = zeros((621,1405))
num_days = 184

for i in range(0,num_days):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date[i] = int(Date3)  

y = 0
for i in range(0,num_days-1):#num_days-1): 
    x = 0
    z = 0

    #### Make sure all ncar and prism files are present
    for j in range(3,15):
        try:
            hrrr_file1 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/regridded.precip.%08d10' % Date[i] + 'F%02d' % j + 'hrrr.grib2'
            if os.path.exists(hrrr_file1):
                x = x + 1
        except:
            pass
    
        try:
            hrrr_file1 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/regridded.precip.%08d22' % Date[i] + 'F%02d' % j + 'hrrr.grib2'
            if os.path.exists(hrrr_file1):
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



    if x == 24 and z == 1:
        y = y + 1
       
            #############  HRRR   ############################
        for j in range(3,15):
            hrrr_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/regridded.precip.%08d10' % Date[i] + 'F%02d' % j + 'hrrr.grib2'
            #hrrr_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%08d07' % Date[i] + 'F12hrrr.grib2'
            grbs = pygrib.open(hrrr_file)
            print hrrr_file
            grb = grbs.select(name='Total Precipitation')[1]
            lat_hrrr,lon_hrrr = grb.latlons()
            tmpmsgs = grbs.select(name='Total Precipitation')
            msg = grbs[2]
            print(msg)
            precip_vals = msg.values
            precip_vals = precip_vals*0.0393689*25.4
            precip_hrrr = precip_hrrr + precip_vals



            hrrr_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/regridded.precip.%08d22' % Date[i] + 'F%02d' % j + 'hrrr.grib2'
            #hrrr_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%08d05' % Date[i+1] + 'F12hrrr.grib2'
            grbs = pygrib.open(hrrr_file)
            print hrrr_file
            grb = grbs.select(name='Total Precipitation')[1]
            lat_hrrr,lon_hrrr = grb.latlons()
            tmpmsgs = grbs.select(name='Total Precipitation')
            msg = grbs[2]
            print(msg)
            precip_vals = msg.values
            precip_vals = precip_vals*0.0393689*25.4
            precip_hrrr = precip_hrrr + precip_vals 



   

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
    
 

   
precip_tot = precip_tot/y#num_days
precip_hrrr = precip_hrrr/y#num_days
## Attempt to fix notation of lons so basemap understands it
#lon_hrrr = lon_hrrr-360



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

np.savetxt('hrrr_dailymean.txt', precip_hrrr)
np.savetxt('prism_hrrr_dailymean.txt', precip_tot)




#avg = precip_hrrr/precip_tot
#bias_mean = np.mean(avg, weights = (avg > 0))
###############################################################################
########################   Plot   #############################################
###############################################################################



cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
fig = plt.figure(figsize=(20,10))
levels = np.arange(.0001,37,.5)
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
########################   hrrr   #############################################


ax = fig.add_subplot(131)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon_hrrr, lat_hrrr)
precip_hrrr = maskoceans(lon_hrrr, lat_hrrr, precip_hrrr)
#map.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='deeppink', lakes=False)
csAVG = map.contourf(x,y,precip_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()

cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('HRRR Precipitation Climatology', fontsize = 14)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)






########################   prism   #############################################
ax = fig.add_subplot(132)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

x, y = map(lons_prism, lats_prism)
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('PRISM Precipitation Climatology', fontsize = 14)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)













### Calcualte bias mean of whole array
avg = precip_hrrr/precip_tot
avg = avg[(avg > 0.1) & (avg < 5)]
bias_mean = np.average(avg)

########################   bias   #############################################
ax = fig.add_subplot(133)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]

#levels = np.arange(.45.000001,.1)
x, y = map(lon_hrrr, lat_hrrr)
csAVG = map.contourf(x,y,precip_hrrr/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.title('HRRR/PRISM Precipitation Climatology', fontsize = 14)
cbar.ax.set_xlabel('Mean Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)
#leg = ([], [], label='Mean bias = %1.3f' % bias_mean)
#plt.legend(handles = [leg],loc = "lower left")

#plt.text(.5,.5,'Mean bias = %1.3f' % bias_mean,rotation = 0, fontsize = 12)
#plt.annotate('Mean bias = %1.3f' % bias_mean, xy=(0.01, .01), xycoords='axes fraction', fontsize = 12)
plt.savefig("./plots/hrrr_prism_climo_%s" % region + ".pdf")
plt.show()

























