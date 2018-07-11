
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
precip_nam12k = zeros((621,1405))
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



    #### Make sure all nam12k and prism files are present
    for t in range(1,9):
        j = (t * 3) + 12
        try:
            nam12k_file1 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/nam12km/regridded.nam_218_%08d_' % Date[i] + '0000_0%02d' % j + '.grb'
            if os.path.exists(nam12k_file1):
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

            nam12k_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/nam12km/regridded.nam_218_%08d_' % Date[i] + '0000_0%02d' % j + '.grb'
      
            grbs = pygrib.open(nam12k_file)
            print nam12k_file
            grb = grbs.select(name='Total Precipitation')[0]
            lat_nam12k,lon_nam12k = grb.latlons()
            
            try: 
                    msg = grbs[4]
                    nam12k_precip = (msg.values)*0.0393689*25.4
                    precip_nam12k = precip_nam12k + nam12k_precip
            except:
                    print 'hi'
                    msg = grbs[3]
                    nam12k_precip = (msg.values)*0.0393689*25.4
                    precip_nam12k = precip_nam12k + nam12k_precip
                    




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
precip_nam12k = precip_nam12k/y
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

np.savetxt('nam12k_dailymean.txt', precip_nam12k)
np.savetxt('prism_nam12k_dailymean.txt', precip_tot)




###############################################################################
########################   Plot   #############################################
###############################################################################




cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
fig = plt.figure(figsize=(25,9))


levels = [.0001, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
########################   NAM12k   #############################################


ax = fig.add_subplot(131)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon_nam12k, lat_nam12k)
precip_nam12k = maskoceans(lon_nam12k, lat_nam12k, precip_nam12k)
#map.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='deeppink', lakes=False)
csAVG = map.contourf(x,y,precip_nam12k, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()

cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('NCAR Ensemble Control', fontsize = 18)
#cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)






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
plt.title('PRISM', fontsize = 18)
#cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)















########################   bias   #############################################
ax = fig.add_subplot(133)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
#plt.text(1,1,'Mean bias = %1.3f' % bias_mean,rotation = 0, fontsize = 12)
#levels = np.arange(.45.000001,.1)
ax.set_title('NCAR/PRISM', fontsize = 18)
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_nam12k/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
#set(cbar,'visible','off')

#cbar.ax.set_xlabel('Mean Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)

plt.savefig("./plots/nam12k_prism_climo_%s" % region + ".pdf")
plt.show()




















