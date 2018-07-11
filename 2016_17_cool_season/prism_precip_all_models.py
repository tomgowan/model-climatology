# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:00:17 2016

@author: u1013082
"""

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
##############   Create lat lon grid for psirm    #############################
###############################################################################




lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667




###############################################################################
##############    Read in modeled and prism data   ############################
###############################################################################




#############  NCAR   ######
precip_ncar = np.loadtxt('ncarens_dailymean.txt')
precip_tot = np.loadtxt('prism_hrrr_dailymean.txt')



#####   HRRR   ############
precip_hrrr = np.loadtxt('hrrr_dailymean.txt')
precip_tot = np.loadtxt('prism_hrrr_dailymean.txt')


#############  NAM4km   ######
precip_nam4k = np.loadtxt('nam4k_dailymean.txt')
precip_tot = np.loadtxt('prism_hrrr_dailymean.txt')






###############################################################################
##############   Calc biases for western US  ##################################
###############################################################################

#############  NCAR   ######
avg1 = precip_ncar[17:453, 0:540]/precip_tot[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
avg_high = avg1[(avg1 > 1) & (avg1 < 10)]
avg_low = avg1[(avg1 > 0.1) & (avg1 < 1)]
bias_mean_ncar = np.average(avg)
bias_mean_ncar_low = np.average(avg_low)
bias_mean_ncar_high = np.average(avg_high)




#####   HRRR   ############
avg1 = precip_hrrr[17:453, 0:540]/precip_tot[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
avg_high = avg1[(avg1 > 1) & (avg1 < 10)]
avg_low = avg1[(avg1 > 0.1) & (avg1 < 1)]
bias_mean_hrrr = np.average(avg)
bias_mean_hrrr_low = np.average(avg_low)
bias_mean_hrrr_high = np.average(avg_high)



#####  NAM4km   ############
avg1 = precip_nam4k[17:453, 0:540]/precip_tot[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
avg_high = avg1[(avg1 > 1) & (avg1 < 10)]
avg_low = avg1[(avg1 > 0.1) & (avg1 < 1)]
bias_mean_nam4k = np.average(avg)
bias_mean_nam4k_low = np.average(avg_low)
bias_mean_nam4k_high = np.average(avg_high)








###############################################################################
########################   Plot   #############################################
###############################################################################


fig = plt.figure(figsize=(20,20))
cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]




########################   NCAR   #############################################
ax = fig.add_subplot(331)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
precip_ncar = maskoceans(lons_prism, lats_prism, precip_ncar)
csAVG = map.contourf(x,y,precip_ncar, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('NCAR Ensemble Control', fontsize = 18)




########################   prism (ncar)   #####################################
ax = fig.add_subplot(332)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
x, y = map(lons_prism, lats_prism)
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('PRISM', fontsize = 18)



########################   bias (NCAR)   ######################################
ax = fig.add_subplot(333)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
ax.set_title('NCAR/PRISM', fontsize = 18)
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_ncar/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.annotate('Mean bias = %1.3f\n' % bias_mean_ncar  +
             'Mean dry bias (bias < 1) = %1.3f\n' % bias_mean_ncar_low + 
             'Mean wet bias (bias > 1) = %1.3f' % bias_mean_ncar_high, xy=(0.02, .02),
             xycoords='axes fraction', fontsize = 10, backgroundcolor = 'w')







########################   hrrr   #############################################
cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
levels = np.arange(.0001,37,.5)
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]

ax = fig.add_subplot(334)
x, y = map(lons_prism, lats_prism)
precip_hrrr = maskoceans(lons_prism, lats_prism, precip_hrrr)
csAVG = map.contourf(x,y,precip_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('HRRR', fontsize = 18)





########################   prism (hrrr)   #####################################
ax = fig.add_subplot(335)
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




########################   bias (hrrr)   ######################################
ax = fig.add_subplot(336)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_hrrr/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.title('HRRR/PRISM', fontsize = 18)
plt.annotate('Mean bias = %1.3f\n' % bias_mean_hrrr  +
             'Mean dry bias (bias < 1) = %1.3f\n' % bias_mean_hrrr_low + 
             'Mean wet bias (bias > 1) = %1.3f' % bias_mean_hrrr_high, xy=(0.02, .02),
             xycoords='axes fraction', fontsize = 10, backgroundcolor = 'w')








########################   nam4km   #############################################


cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]


ax = fig.add_subplot(337)
x, y = map(lons_prism, lats_prism)
precip_nam4k = maskoceans(lons_prism, lats_prism, precip_nam4k)
csAVG = map.contourf(x,y,precip_nam4k, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('NAM4km', fontsize = 18)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)






########################   prism (nam4k)   ####################################
ax = fig.add_subplot(338)
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
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)









########################   bias (nam4k)   #####################################
ax = fig.add_subplot(339)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_nam4k/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.title('NAM4km/PRISM', fontsize = 18)
cbar.ax.set_xlabel('Mean Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)
plt.annotate('Mean bias = %1.3f' % bias_mean_nam4k, xy=(0.01, .01), xycoords='axes fraction', fontsize = 11)
plt.annotate('Mean bias = %1.3f\n' % bias_mean_nam4k +
             'Mean dry bias (bias < 1) = %1.3f\n' % bias_mean_nam4k_low + 
             'Mean wet bias (bias > 1) = %1.3f' % bias_mean_nam4k_high, xy=(0.02, .02),
             xycoords='axes fraction', fontsize = 10, backgroundcolor = 'w')
plt.tight_layout()
plt.savefig("./plots/prism_climo_allmodels%s" % region + ".pdf")
plt.show()










'''
fig = plt.figure(figsize=(10,10))

sub = 331

for i in range(9):
    sub = sub + i
    cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
    ax = fig.add_subplot(sub)
    map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
    x, y = map(lons_prism, lats_prism)


###   NCAR  ###  
    if sub == 331:
        levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
        precip_ncar = maskoceans(lons_prism, lats_prism, precip_ncar)
        csAVG = map.contourf(x,y,precip_ncar, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('NCAR Ensemble Control', fontsize = 18)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%")
        cbar.ax.tick_params(labelsize=12)
        
    if sub == 332:
        levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
        precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
        csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('PRISM', fontsize = 18)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%")
        cbar.ax.tick_params(labelsize=12)

    if sub == 333:
        levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
        csAVG = map.contourf(x,y,precip_ncar/precip_tot, levels, cmap = plt.cm.BrBG, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('NCAR/PRISM', fontsize = 18)
        cbar.ax.tick_params(labelsize=12)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
        cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
        plt.annotate('Mean bias = %1.3f' % bias_mean_ncar, xy=(0.01, .01), xycoords='axes fraction', fontsize = 11)
        
        
        
###   HRRR  ###  
    if sub == 334:
        levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
        precip_hrrr = maskoceans(lons_prism, lats_prism, precip_hrrr)
        csAVG = map.contourf(x,y,precip_hrrr, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('HRRR', fontsize = 18)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%")
        cbar.ax.tick_params(labelsize=12)
        
    if sub == 335:   
        levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
        precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
        csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('PRISM', fontsize = 18)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%")
        cbar.ax.tick_params(labelsize=12)

    if sub == 336:
        levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
        csAVG = map.contourf(x,y,precip_hrrr/precip_tot, levels, cmap = plt.cm.BrBG, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('HRRR/PRISM', fontsize = 18)
        cbar.ax.tick_params(labelsize=12)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
        cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
        plt.annotate('Mean bias = %1.3f' % bias_mean_hrrr, xy=(0.01, .01), xycoords='axes fraction', fontsize = 11)
        
        
        
        
        
###   NAM4k  ###  
    if sub == 337:
        levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
        precip_nam4k = maskoceans(lons_prism, lats_prism, precip_nam4k)
        csAVG = map.contourf(x,y,precip_nam4k, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('NAM4km', fontsize = 18)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%")
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)
        
    if sub == 338:
        levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
        precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
        csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('PRISM', fontsize = 18)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%")
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)

    if sub == 339:
        levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
        csAVG = map.contourf(x,y,precip_nam4k/precip_tot, levels, cmap = plt.cm.BrBG, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
        plt.title('NAM4km/PRISM', fontsize = 18)
        cbar.ax.tick_params(labelsize=12)
        cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
        cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
        plt.annotate('Mean bias = %1.3f' % bias_mean_nam4k, xy=(0.01, .01), xycoords='axes fraction', fontsize = 11)
        cbar.ax.set_xlabel('Mean Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)

map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
plt.savefig("./plots/prism_climo_allmodels%s" % region + ".pdf")
plt.show()
'''


