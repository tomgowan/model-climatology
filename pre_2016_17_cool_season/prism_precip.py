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
##############   Read in ncar  and prism precip   #############################
###############################################################################
Date2= '20151001'
Date = zeros((184))
precip_ncar = zeros((621,1405))
precip_tot = zeros((621,1405))
num_days = 183

for i in range(0,183):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date[i] = int(Date3)  


for i in range(0,num_days): 
    x = 0
    y = 0

    #### Make sure all ncar and prism files are present
    for j in range(8,32):
        NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/ncarens/regridded.precip.ncar_3km_%08d00' % Date[i] + '_mem1_f0%02d' % j + '.grb2'
        if os.path.exists(NCARens_file):
            x = x + 1
    try:
        prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_stable_4kmD2_%08d' % Date[i] + '_asc.asc'
        if os.path.exists(prism_file):
            y = 1
    except:
        pass
    
    try:
        prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_provisional_4kmD2_%08d' % Date[i] + '_asc.asc'
        if os.path.exists(prism_file):
            y = 1
    except:
        pass



    if x == 24 and y == 1:
        for j in range(8,32):#32
            #############  NCAR   ############################
            NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/ncarens/regridded.precip.ncar_3km_%08d00' % Date[i] + '_mem1_f0%02d' % j + '.grb2'
            print NCARens_file
            grb = grbs.select(name='Total Precipitation')[0]

            lat_ncar,lon_ncar = grb.latlons()

            grbs = pygrib.open(NCARens_file)

            tmpmsgs = grbs.select(name='Total Precipitation')
            msg = grbs[1]
            precip_vals = msg.values
            precip_vals = precip_vals*0.0393689*25.4
            precip_ncar = precip_ncar + precip_vals 

   

        ############### Prism #####################################
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_stable_4kmD2_%08d" % Date[i] + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_provisional_4kmD2_%08d" % Date[i] + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        
        precip_tot = precip_tot + precip
    
 

   
precip_tot = precip_tot/num_days
precip_ncar = precip_ncar/num_days
## Attempt to fix notation of lons so basemap understands it
lon_ncar = lon_ncar-360



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
'''
np.savetxt('ncarens_dailymean.txt', precip_ncar)
np.savetxt('prism_dailymean.txt', precip_tot)
precip_ncar = np.loadtxt('ncarens_dailymean.txt')

bias_mean = np.mean(precip_ncar/precip_tot)
###############################################################################
########################   Plot   #############################################
###############################################################################




my_cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
fig = plt.figure(figsize=(20,10))
levels = np.arange(.0001,7,.1)
########################   NCAR   #############################################


ax = fig.add_subplot(131)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon_ncar, lat_ncar)
precip_ncar = maskoceans(lon_ncar, lat_ncar, precip_ncar)
#map.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='deeppink', lakes=False)
csAVG = map.contourf(x,y,precip_ncar, levels, cmap = my_cmap)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()

cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('NCAR Ensemble Control Precipitation Climatology', fontsize = 14)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)






########################   prism   #############################################
ax = fig.add_subplot(132)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

x, y = map(lon_ncar, lat_ncar)
precip_tot = maskoceans(lon_ncar, lat_ncar, precip_tot)
csAVG = map.contourf(x,y,precip_tot, levels, cmap = my_cmap)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('PRISM Precipitation Climatology', fontsize = 14)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)


















########################   bias   #############################################
ax = fig.add_subplot(133)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
#plt.text(1,1,'Mean bias = %1.3f' % bias_mean,rotation = 0, fontsize = 12)
#levels = np.arange(.45.000001,.1)
x, y = map(lon_ncar, lat_ncar)
csAVG = map.contourf(x,y,precip_ncar/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.title('NCAR/PRISM Precipitation Climatology', fontsize = 14)
cbar.ax.set_xlabel('Mean Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)
plt.savefig("./plots/ncar_prism_climo_%s" % region + ".pdf")
plt.show()


























