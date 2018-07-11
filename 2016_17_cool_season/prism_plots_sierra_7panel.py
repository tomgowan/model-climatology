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
import matplotlib.patheffects as path_effects



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

### For SNOTEL
#el_file = '/uufs/chpc.utah.edu/common/home/horel-group/archive/terrain/WesternUS_terrain.nc'
el_file = '/uufs/chpc.utah.edu/common/home/u1013082/elevation_data/60second_westernUS.nc'
fh = Dataset(el_file, mode='r')
elevation_p = fh.variables['Band1'][:]
lat_p = fh.variables['lat'][:]
lon_p = fh.variables['lon'][:]

xlen = len(elevation_p[:,0])
ylen = len(elevation_p[0,:])

lat_netcdf_p = zeros((xlen, ylen))
long_netcdf_p = zeros((xlen,ylen))
for i in range(xlen):
    long_netcdf_p[i,:] = lon_p
for j in range(ylen):
    lat_netcdf_p[:,j] = lat_p
    
elevation_p = new_array


##Labels for Salt Lake City and Seattle and Mountains
# Cities names and coordinates
cities = ['SLT']
lat = [38.93]
lon = [-119.97]#-119.83]





##################################     SNOTEL Bias    #############################
##Labels for Salt Lake City and Seattle and Mountains
# Cities names and coordinates
cities = ['SLC', 'SEA', 'SLT']
lat = [1, 47.593, 1, 39.07, 39.2, 39.35, 39.07, 39.35, 39.08]
lon = [-111.913, -122.306, -119.97, -120.35, -119.56, -119.67, -119.764, -120.5, -120.13]#-119.83]


levels_el = np.arange(1200,3401,100)
cmap = plt.cm.BrBG





#%%

##############################################################################
##############################  Plot ##################################
##############################################################################

levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 10]

fig1=plt.figure(num=None, figsize=(11,20), dpi=800, facecolor='w', edgecolor='k')

levels_el = np.arange(1000,2900,200)
thick = 1.1
cmap = plt.cm.BrBG

title = 22
dot = 140
info = 16
label = 18
axis_title = 28






##################################     NCAR     #############################
ax = fig1.add_subplot(3,2,1)
plt.tight_layout()
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
x2, y2 = map(long_netcdf_p[:,:], lat_netcdf_p[:,:])
levels = levels
csAVG = map.contourf(x,y,precip_ncar/precip_tot_ncar, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG2 = map.contour(x2,y2,elevation_p[:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)

xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[0,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot+60, vmin = 0.4, vmax = 2.2, zorder = 10)


#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])

plt.text(x[3], y[3], 'Sierra Crest', fontsize = label+3, rotation = -60, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])
plt.text(x[4], y[4], 'Pine Nut Mtns', fontsize = label+3, rotation = -90, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])
plt.text(x[5], y[5], 'Carson\nRange', fontsize = label+3, rotation = 0, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])
ax.arrow(x[5], y[5], -20000, -39000, fc="k", ec="k",head_width=3000, head_length=3000, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()], zorder = 11)
ax.arrow(x[5], y[5], -20000, 0, fc="k", ec="k",head_width=3500, head_length=3500, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()], zorder = 11)
plt.text(x[8], y[8], 'Lake\nTahoe', fontsize = label+3, rotation = 0, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])



map.drawcoastlines()  
map.drawstates()
map.drawcountries()
#For cities
#x, y = map(lon, lat)
#map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
## for each city,
#for city, xc, yc in zip(cities, x, y):
## draw the city name in a yellow (shaded) box
#  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
ax.set_title("(a) NCAR ENS CTL", fontsize = title)



##################################     GFS     #############################

ax = fig1.add_subplot(3,2,4)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_gfs/precip_tot_gfs, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation_p[:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG

xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[1,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot+60, vmin = 0.4, vmax = 2.2, zorder = 10)#,norm=matplotlib.colors.LogNorm() )  
    
    
map.drawcoastlines()  
map.drawstates()
map.drawcountries()
#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()]) 
ax.set_title("(d) GFS", fontsize = title)





##################################     HRRR     ###############################


ax = fig1.add_subplot(3, 2, 2)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_hrrr/precip_tot_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N)) 
csAVG2 = map.contour(x2,y2,elevation_p[:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1) 
cmap = plt.cm.BrBG 

xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[2,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot+60, vmin = 0.4, vmax = 2.2, zorder = 10)#,norm=matplotlib.colors.LogNorm() ) 
    
    
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])
  
ax.set_title("(b) HRRR", fontsize = title)



##################################     NAM     #############################

ax = fig1.add_subplot(3, 2, 3)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_nam3km/precip_tot_nam3km, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation_p[:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG 

xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[3,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot+60, vmin = 0.4, vmax = 2.2, zorder = 10)#,norm=matplotlib.colors.LogNorm() )  
    
    
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])

ax.set_title("(c) NAM-3km", fontsize = title)           

    
##################################     SREF ARW     #############################

ax = fig1.add_subplot(3, 2,  5)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_sref_arw/precip_tot_sref_arw, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation_p[:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG 

xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[4,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot+60, vmin = 0.4, vmax = 2.2, zorder = 10)#,norm=matplotlib.colors.LogNorm() )  
    
    
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])
  
ax.set_title("(e) SREF ARW CTL", fontsize = title)      

             
             
##################################     SREF NMB    #############################

ax = fig1.add_subplot(3, 2, 6)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_sref_nmb/precip_tot_sref_nmb, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG2 = map.contour(x2,y2,elevation_p[:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)
cmap = plt.cm.BrBG

xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[5,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot+60, vmin = 0.4, vmax = 2.2, zorder = 10)#,norm=matplotlib.colors.LogNorm() )  
    
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-12000, yc+5000, city, fontsize = label+3, path_effects = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                       path_effects.Normal()])
ax.set_title("(f) SREF NMMB CTL", fontsize = title)         




   

plt.tight_layout()
subplots_adjust(left=None, bottom=.08, right=None, top=.98, wspace=.07, hspace=.02) 
ax2 = fig1.add_axes([0.1,0.01,0.8,.4], visible = None)       
cbar = plt.colorbar(csAVG, ax = ax2, ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5],orientation='horizontal')
cbar.ax.set_xlabel('Bias Ratio', fontsize  = 22)  
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = 18) 


#cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
#cbar.ax.set_xlabel('Bias Ratio', fontsize  = axis_title)  
#cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 


plt.savefig("../../../public_html/bias_prism_snotel_allmodels_interp_2016_17_sierra.png")











