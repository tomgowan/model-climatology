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






#####  Prism and model data for prism comparison ####################################



#############  NCAR   ######
precip_ncar = np.loadtxt('ncar_dailymean.txt')
precip_tot_ncar = np.loadtxt('prism_ncar_dailymean.txt')



#####   HRRR   ############
precip_hrrr = np.loadtxt('hrrr_dailymean.txt')
precip_tot_hrrr = np.loadtxt('prism_hrrr_dailymean.txt')


#############  NAM4km   ######
precip_nam4k = np.loadtxt('nam4k_dailymean.txt')
precip_tot_nam4k = np.loadtxt('prism_nam4k_dailymean.txt')


#############  NAM12km   ######
precip_nam12k = np.loadtxt('nam12k_dailymean.txt')
precip_tot_nam12k = np.loadtxt('prism_nam12k_dailymean.txt')


#############  GFS   ######
precip_nam12k = np.loadtxt('gfs_dailymean.txt')
precip_tot_gfs = np.loadtxt('prism_gfs_dailymean.txt')


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






#####  NAM4km   ############
avg1 = precip_nam4k[17:453, 0:540]/precip_tot_nam4k[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_nam4k = np.average(avg)
bias_stdev_nam4k = np.std(avg)





#####  NAM12km   ############
avg1 = precip_nam12k[17:453, 0:540]/precip_tot_nam12k[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_nam12k = np.average(avg)
bias_stdev_nam12k = np.std(avg)


#####  GFS   ############
avg1 = precip_gfs[17:453, 0:540]/precip_tot_gfs[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_gfs = np.average(avg)
bias_stdev_gfs = np.std(avg)







lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667


'''


#############    PRISM  DAILY PRECIP   ########################################


fig1=plt.figure(num=None, figsize=(12,14), dpi=800, facecolor='w', edgecolor='k')
levels_el = np.arange(0,5000,100)
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]

cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

ax = fig1.add_subplot(111)
x, y = map(lons_prism, lats_prism)

csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  

#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("PRISM Precipitation", fontsize = 28)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.set_xlabel('Mean Daily Precip. from 10/01/15 to 03/31/16 (mm/day)', fontsize = 24, labelpad = 15)
cbar.ax.set_xticklabels(['0.0', '1.0', '2.0', '4.0', '6.5', '8.5', '11.0', '15.0', '22.0', '38.0'], fontsize = 20) 

plt.tight_layout()
plt.savefig("./plots/prism_precip.png")




'''







##############################################################################
####  Bias form snotel and prism  for ncar  ##################################
##############################################################################

levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]



cmap=plt.cm.BrBG

fig1=plt.figure(num=None, figsize=(27.5,10), dpi=800, facecolor='w', edgecolor='k')


ax = fig1.add_subplot(151)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
#xi, yi = map(bias[:,5], bias[:,4])
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_ncar/precip_tot_ncar, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("NCAR Ensemble Control", fontsize = 22)
             
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xlabel('Daily Precip. Bias', fontsize  = 18)  
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = 16) 
plt.annotate('Mean bias = %1.3f\n' % bias_mean_ncar  +
             'Std dev bias = %1.3f' % bias_stdev_ncar, xy=(0.015, .024),
             xycoords='axes fraction', fontsize = 18, backgroundcolor = 'w')




##################################     HRRR     ###############################


ax = fig1.add_subplot(152)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_hrrr/precip_tot_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("HRRR", fontsize = 22)    
        
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xlabel('Daily Precip. Bias', fontsize  = 18)   
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = 16) 
plt.annotate('Mean bias = %1.3f\n' % bias_mean_hrrr  +
             'Std dev bias = %1.3f' % bias_stdev_hrrr, xy=(0.015, .024),
             xycoords='axes fraction', fontsize = 18, backgroundcolor = 'w')







##################################     NAM4km     #############################

ax = fig1.add_subplot(153)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_nam4k/precip_tot_nam4k, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("NAM-4km", fontsize = 22)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xlabel('Daily Precip. Bias', fontsize  = 18)       
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = 16) 
plt.annotate('Mean bias = %1.3f\n' % bias_mean_nam4k  +
             'Std dev bias = %1.3f' % bias_stdev_nam4k, xy=(0.015, .024),
             xycoords='axes fraction', fontsize = 18, backgroundcolor = 'w')
             
             
             
##################################     NAM12km     #############################

ax = fig1.add_subplot(154)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_nam12k/precip_tot_nam12k, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("NAM-12km", fontsize = 22)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xlabel('Daily Precip. Bias', fontsize  = 18)       
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = 16) 
plt.annotate('Mean bias = %1.3f\n' % bias_mean_nam12k  +
             'Std dev bias = %1.3f' % bias_stdev_nam12k, xy=(0.015, .024),
             xycoords='axes fraction', fontsize = 18, backgroundcolor = 'w')

##################################     GFS     #############################

ax = fig1.add_subplot(155)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_gfs/precip_tot_gfs, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("GFS", fontsize = 22)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xlabel('Daily Precip. Bias', fontsize  = 18)       
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = 16) 
plt.annotate('Mean bias = %1.3f\n' % bias_mean_gfs  +
             'Std dev bias = %1.3f' % bias_stdev_gfs, xy=(0.015, .024),
             xycoords='axes fraction', fontsize = 18, backgroundcolor = 'w')

plt.tight_layout()
plt.savefig("../plots/bias_prism_allmodels.pdf")
plt.show()

