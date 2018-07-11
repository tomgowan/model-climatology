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




    
### SNOTEL info ####
bias = np.load('snotel_bias_wasatch.npy')





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


print('hi')

### Lat and Lon to Plot Prism
lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667





#%%
#####   ELevation Data   #########

## For SNOTEL
el_file = '/uufs/chpc.utah.edu/common/home/horel-group/archive/terrain/WesternUS_terrain.nc'
fh = Dataset(el_file, mode='r')
elevation = fh.variables['elevation'][:]
lat = fh.variables['latitude'][:]
lon = fh.variables['longitude'][:]

lat_netcdf = zeros((3600,3600))
long_netcdf = zeros((3600,3600))
for i in range(3600):
    lat_netcdf[:,i] = lat
    long_netcdf[i,:] = lon
    
    
### For PRISM
NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/random/wrfinput_d02'
fh = Dataset(NCARens_file, mode='r')
elevation_p = fh.variables['HGT'][:]
lat_netcdf_p = fh.variables['XLAT'][:]
long_netcdf_p = fh.variables['XLONG'][:] 


##Labels for Salt Lake City and Seattle and Mountains
# Cities names and coordinates
cities = ['SLT']
lat = [38.93]
lon = [-119.97]#-119.83]







#%%

##############################################################################
##############################  Plot ##################################
##############################################################################

levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 10]

fig1=plt.figure(num=None, figsize=(20,20.5), dpi=800, facecolor='w', edgecolor='k')

levels_el = np.arange(400,3500,200)
thick = 1.1
cmap = plt.cm.BrBG

title = 26
dot = 170
info = 16
label = 22
axis_title = 28

##################################     NCAR     #############################
ax = fig1.add_subplot(3,4,1)
plt.tight_layout()
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
x2, y2 = map(long_netcdf_p[0,:,:], lat_netcdf_p[0,:,:])
levels = levels
csAVG = map.contourf(x,y,precip_ncar/precip_tot_ncar, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG2 = map.contour(x2,y2,elevation_p[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)
map.drawcoastlines()  
map.drawstates()
map.drawcountries()
#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)
  
ax.set_title("(a) NCAR ENS CTL", fontsize = title, y = 1.02)



##################################     GFS     #############################

ax = fig1.add_subplot(3,4,7)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_gfs/precip_tot_gfs, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation_p[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG
map.drawcoastlines()  
map.drawstates()
map.drawcountries()
#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3) 
ax.set_title("(g) GFS", fontsize = title, y = 1.02)





##################################     HRRR     ###############################


ax = fig1.add_subplot(3, 4, 3)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_hrrr/precip_tot_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N)) 
csAVG2 = map.contour(x2,y2,elevation_p[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1) 
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
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)
  
ax.set_title("(c) HRRR", fontsize = title, y = 1.02)



##################################     NAM     #############################

ax = fig1.add_subplot(3, 4, 5)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_nam3km/precip_tot_nam3km, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation_p[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
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
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)

ax.set_title("(e) NAM-3km", fontsize = title, y = 1.02)           

    
##################################     SREF ARW     #############################

ax = fig1.add_subplot(3, 4,  9)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_sref_arw/precip_tot_sref_arw, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation_p[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
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
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)
  
ax.set_title("(i) SREF ARW CTL", fontsize = title, y = 1.02)      

             
             
##################################     SREF NMB    #############################

ax = fig1.add_subplot(3, 4, 11)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_sref_nmb/precip_tot_sref_nmb, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG2 = map.contour(x2,y2,elevation_p[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)
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
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)
ax.set_title("(k) SREF NMMB CTL", fontsize = title, y = 1.02)         







##################################     SNOTEL Bias    #############################
##Labels for Salt Lake City and Seattle and Mountains
# Cities names and coordinates
cities = ['SLC', 'SEA', 'SLT']
lat = [40.774, 47.593, 38.93, 39.07, 39.2, 39.35, 39.07, 39.35]
lon = [-111.913, -122.306, -119.97, -120.35, -119.56, -119.67, -119.764, -120.5]#-119.83]


levels_el = np.arange(1200,3401,100)
cmap = plt.cm.BrBG


##################################     NCAR     #############################
ax = fig1.add_subplot(3, 4, 2)
plt.tight_layout()
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')


#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)

plt.text(x[3], y[3], 'Sierra Crest', fontsize = label+3, rotation = -60)
plt.text(x[4], y[4], 'Pine Nut Mtns', fontsize = label+3, rotation = -90)
plt.text(x[5], y[5], 'Carson\nRange', fontsize = label+3, rotation = 0)
ax.arrow(x[5], y[5], -20000, -39000, fc="k", ec="k",head_width=3000, head_length=3000)
ax.arrow(x[5], y[5], -20000, 0, fc="k", ec="k",head_width=3500, head_length=3500)


#precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[0,:,5], bias[0,:,4])
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[0,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("(b) NCAR ENS CTL", fontsize = title, y = 1.02)



##################################     GFS     #############################

ax = fig1.add_subplot(3, 4, 8)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)
  
  
#precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
cmap = plt.cm.BrBG
csAVG = map.scatter(xi,yi, c = bias[1,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("(h) GFS", fontsize = title, y = 1.02)



##################################     HRRR     ###############################


ax = fig1.add_subplot(3, 4, 4)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)
  
#precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[0,:,5], bias[0,:,4])
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[2,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("(d) HRRR", fontsize = title, y = 1.02)



##################################     NAM     #############################

ax = fig1.add_subplot(3, 4, 6)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+3000, city, fontsize = label+3)

xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
csAVG = map.scatter(xi,yi, c = bias[3,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("(f) NAM-3km", fontsize = title, y = 1.02)           

             
             
##################################     SREF ARW     #############################

ax = fig1.add_subplot(3, 4, 10)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)

xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
cmap = plt.cm.BrBG
csAVG = map.scatter(xi,yi, c = bias[4,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("(j) SREF ARW CTL", fontsize = title, y = 1.02)
             


             
             
##################################     SREF NMB    #############################

ax = fig1.add_subplot(3, 4, 12)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3)
  
  xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
cmap = plt.cm.BrBG
csAVG = map.scatter(xi,yi, c = bias[5,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("(l) SREF NMMB CTL", fontsize = title, y = 1.02)
  



           
#plt.tight_layout(pad=1, w_pad=0.5, h_pad=0)
plt.tight_layout()
subplots_adjust(left=None, bottom=.06, right=None, top=None, wspace=.1, hspace=-.25)

ax2 = fig1.add_axes([0.03,0.04,.44,.4], visible = None)         
cbar = plt.colorbar(csAVG, ax = ax2, ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5],orientation='horizontal')
cbar.ax.set_xlabel('Bias Ratio', fontsize  = axis_title, labelpad = 18)  
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label+2) 

ax3 = fig1.add_axes([0.53,0.04,.44,.4], visible = None)
cbar = plt.colorbar(csAVG2, ax = ax3, ticks= np.arange(1200,3401,400),orientation='horizontal')
cbar.ax.set_xlabel('Elevation (m)', fontsize  = axis_title, labelpad = 18)  
cbar.ax.set_xticklabels(np.arange(1200,3401,400), fontsize = label+2) 

plt.savefig("../../../public_html/bias_prism_snotel_allmodels_interp_2016_17_sierra.png")











