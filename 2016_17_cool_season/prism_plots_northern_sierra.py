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
from matplotlib import animation
import matplotlib.animation as animation
import types
import matplotlib.lines as mlines


wmont = [-117.0, 43.0, -108.5, 49.0]
#utah = [-114.7, 36.7, -108.9, 42.5]
colorado = [-110.0, 36.0, -104.0, 41.9]
wasatch = [-113.4, 39.5, -110.7, 41.9]
#cascades = [-125.3, 42.0, -116.5, 49.1]
cascades = [-123.96, 46.06, -120.38, 48.99]
west = [-125.3, 31.0, -102.5, 49.2]
utah = [-112.4, 40.2, -111.2, 41.3]
north_rockies = [-115.736, 46.90, -113.198, 48.98]
sierra = [-120.494, 38.558, -119.382, 39.471]

region = 'sierra'



if region == 'wmont':
    latlon = wmont
    
if region == 'utah':
    latlon = utah
    
if region == 'colorado':
    latlon = colorado
    
if region == 'wasatch':
    latlon = wasatch
    
if region == 'cascades':
    latlon = cascades

if region == 'west':
    latlon = west
    
if region == 'north_rockies':
    latlon = north_rockies
    
if region == 'sierra':
    latlon = sierra


#####  Prism and model data for prism comparison ####################################



#############  NCAR   ######
precip_ncar = np.loadtxt('ncar_dailymean.txt')
precip_tot_ncar = np.loadtxt('prism_ncar_dailymean.txt')



#####   HRRR   ############
precip_hrrr = np.loadtxt('hrrr_dailymean.txt')
precip_tot_hrrr = np.loadtxt('prism_hrrr_dailymean.txt')


#############  NAM3km   ######
precip_nam3km = np.loadtxt('nam3km_dailymean.txt')
precip_tot_nam3km = np.loadtxt('prism_nam3km_dailymean.txt')



#############  GFS   ######
precip_gfs = np.loadtxt('gfs_dailymean.txt')
precip_tot_gfs = np.loadtxt('prism_gfs_dailymean.txt')



#############  sref_arw   ######
precip_sref_arw = np.loadtxt('sref_arw_ctl_dailymean.txt')
precip_tot_sref_arw = np.loadtxt('prism_sref_arw_dailymean.txt')

#############  sref_nmb   ######
precip_sref_nmb = np.loadtxt('sref_nmb_ctl_dailymean.txt')
precip_tot_sref_nmb = np.loadtxt('prism_sref_nmb_dailymean.txt')



###############################################################################
##############   Calc biases for western US  ##################################
###############################################################################

#############  NCAR   ######
avg1 = precip_ncar[17:453, 0:540]/precip_tot_ncar[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_ncar = np.average(avg)
bias_stdev_ncar = np.std(avg)





#####   HRRR   ############
avg1 = precip_hrrr[17:453, 0:540]/precip_tot_hrrr[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_hrrr = np.average(avg)
bias_stdev_hrrr = np.std(avg)


#####   gfs   ############
avg1 = precip_gfs[17:453, 0:540]/precip_tot_gfs[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_gfs = np.average(avg)
bias_stdev_gfs = np.std(avg)


#####   nam3km   ############
avg1 = precip_nam3km[17:453, 0:540]/precip_tot_nam3km[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_nam3km = np.average(avg)
bias_stdev_nam3km = np.std(avg)



#####   sref_arw   ############
avg1 = precip_sref_arw[17:453, 0:540]/precip_tot_sref_arw[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_sref_arw = np.average(avg)
bias_stdev_sref_arw = np.std(avg)



#####   sref_nmb   ############
avg1 = precip_sref_nmb[17:453, 0:540]/precip_tot_sref_nmb[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_sref_nmb = np.average(avg)
bias_stdev_sref_nmb = np.std(avg)








lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667




#%%
#####   ELevation Data   #########

NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/random/wrfinput_d02'
fh = Dataset(NCARens_file, mode='r')

elevation = fh.variables['HGT'][:]
lat_netcdf = fh.variables['XLAT'][:]
long_netcdf = fh.variables['XLONG'][:] 

##Lbel Salt Lake City and Seattle
# Cities names and coordinates
cities = ['SLC', 'SEA']
lat = [40.774, 47.593]
lon = [-111.913, -122.306]


##############################################################################
##############################  Plot ##################################
##############################################################################

levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 10]

fig1=plt.figure(num=None, figsize=(11,18.5), dpi=800, facecolor='w', edgecolor='k')
levels_el = np.arange(400,3500,200)
thick = 1.1
cmap = plt.cm.BrBG

title = 21
dot = 75
info = 16
label = 16
axis_title = 20

##################################     NCAR     #############################
ax = fig1.add_subplot(321)
plt.tight_layout()
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
#precip_tot_ncar = maskoceans(lons_prism, lats_prism, precip_tot_ncar)
x, y = map(lons_prism, lats_prism)
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
levels = levels
csAVG = map.contourf(x,y,precip_ncar/precip_tot_ncar, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)
map.drawcoastlines()  
map.drawstates()
map.drawcountries()
#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
ax.set_title("NCAR ENS CTL", fontsize = title)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5]) 
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_ncar  +
#             'SD = %1.3f' % bias_stdev_ncar, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')



##################################     GFS     #############################

ax = fig1.add_subplot(322)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
#precip_tot_gfs = maskoceans(lons_prism, lats_prism, precip_tot_gfs)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_gfs/precip_tot_gfs, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
ax.set_title("GFS", fontsize = title)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_gfs  +
#             'SD = %1.3f' % bias_stdev_gfs, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')






##################################     HRRR     ###############################


ax = fig1.add_subplot(323)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
#precip_tot_hrrr = maskoceans(lons_prism, lats_prism, precip_tot_hrrr)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_hrrr/precip_tot_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N)) 
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1) 
cmap = plt.cm.BrBG 
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
ax.set_title("HRRR", fontsize = title)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_hrrr  +
#             'SD = %1.3f' % bias_stdev_hrrr, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')






##################################     NAM     #############################

ax = fig1.add_subplot(324)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
#precip_tot_nam3km = maskoceans(lons_prism, lats_prism, precip_tot_nam3km)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_nam3km/precip_tot_nam3km, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG 

map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()


#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
  
ax.set_title("NAM-3km", fontsize = title)           
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_nam3km  +
#             'SD = %1.3f' % bias_stdev_nam3km, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')
             
             
             
             
##################################     SREF ARW     #############################

ax = fig1.add_subplot(325)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
#precip_tot_sref_arw = maskoceans(lons_prism, lats_prism, precip_tot_sref_arw)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_sref_arw/precip_tot_sref_arw, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG 


map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
  
ax.set_title("SREF ARW CTL", fontsize = title)
             
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
  
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xlabel('Bias Ratio', fontsize  = axis_title) 
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_sref_arw  +
#             'SD = %1.3f' % bias_stdev_sref_arw, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')


             
             
##################################     SREF NMB    #############################

ax = fig1.add_subplot(326)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
#precip_tot_sref_nmb = maskoceans(lons_prism, lats_prism, precip_tot_sref_nmb)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_sref_nmb/precip_tot_sref_nmb, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)
cmap = plt.cm.BrBG 


map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
  
ax.set_title("SREF NMB CTL", fontsize = title)
             
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xlabel('Bias Ratio', fontsize  = axis_title)  
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_sref_nmb  +
#             'SD = %1.3f' % bias_stdev_sref_nmb, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')
 

           
plt.tight_layout()
plt.savefig("../../../public_html/bias_prism_allmodels_interp_2016_17_sierra.png")














































