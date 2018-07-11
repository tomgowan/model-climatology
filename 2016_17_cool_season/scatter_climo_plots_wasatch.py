#%%

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import pygrib, os, sys, glob
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from matplotlib.mlab import bivariate_normal
from matplotlib import colors, ticker, cm
from mpl_toolkits.basemap import Basemap, maskoceans


wmont = [-117.0, 43.0, -108.5, 49.0]
#utah = [-114.7, 36.7, -108.9, 42.5]
colorado = [-110.0, 36.0, -104.0, 41.9]
wasatch = [-113.6, 39.5, -110.7, 41.9]
#cascades = [-125.3, 42.0, -116.5, 49.1]
cascades = [-124.987, 47.47, -119.987, 48.945]
west = [-125.3, 31.0, -102.5, 49.2]
utah = [-112.73, 40.2, -111.5, 41.25]

region = 'utah'



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
    
#%%
    
    
nearestNCAR = zeros((785,650))
totalprecip = zeros((152,2))
Date = zeros((152))
frequency = zeros((80,2))
frequency_snotel = zeros((80,2))
daily_snotel_precip = zeros((798,20))
snotel_lat2 = zeros((798))
snotel_lon2 = zeros((798))
row = zeros((798))
col = zeros((798))
region_bias=zeros((4,8))


###############################################################################
############ Read in  12Z to 12Z data   #####################
###############################################################################

            
x = 0
q = 0
v = 0
i = 0   



links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/snotel_precip_2016_2017_qc.csv",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/ncarens_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/gfs13km_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/hrrr_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/nam3km_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_arw_precip_12Zto12Z_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_nmb_precip_12Zto12Z_interp.txt"]

     
data = zeros((len(links),798,185))

         
for c in range(len(links)):
    x = 0
    q = 0
    v = 0
    i = 0     
    with open(links[c], "rt") as f:
        for line in f:
            commas = line.count(',') + 1
            year = line.split(',')[0]
            month = line.split(',')[1]
            day = line.split(',')[2]
            date = year + month + day
            y = 0
      

            if i == 0:     
                for t in range(3,commas):
                    station_id = line.split(',')[t]
                    data[c,x,0] = station_id
                    x = x + 1
                    
            if i == 1:     
                for t in range(3,commas):
                    lat = line.split(',')[t]
                    data[c,q,1] = lat
                    q = q + 1
            
            if i == 2:     
                for t in range(3,commas):
                    lon = line.split(',')[t]
                    data[c,v,2] = lon
                    v = v + 1

            if i != 0 and i != 1 and i != 2:
                for t in range(3,commas):   
                    precip = line.split(',')[t]
                    precip = float(precip)
                    data[c,y,i] = precip

                    y = y + 1
            
            i = i + 1

data[np.isnan(data)] = 9999




###############################################################################
###########################   Calculate Bias  #################################
###############################################################################
bias = zeros((len(links)-1,800,20))



for c in range(1,len(links)):
    w = -1
    for x in range(len(data[c,:,0])):
        for y in range(len(data[0,:,0])):
                if data[c,x,0] == data[0,y,0]:
                    w = w + 1
                    bias[c-1,w,0] = data[c,x,0]
                    bias[c-1,w,1] = data[0,y,0]
                

                    for z in range(3,185):
                        if all(data[1:,x,z] < 1000) and data[0,y,z] < 1000:
                        
                            #lat/lon data
                            bias[c-1,w,2] = data[c-1,x,1]
                            bias[c-1,w,3] = data[c-1,x,2]
                            bias[c-1,w,4] = data[0,y,1]
                            bias[c-1,w,5] = data[0,y,2]   
                        
                            #precip data
                            bias[c-1,w,6] = bias[c-1,w,6] + data[0,y,z] #sum SNOTEL precip
                            bias[c-1,w,9] = bias[c-1,w,9] + 1 #number of days
                            bias[c-1,w,7] = bias[c-1,w,7] + data[c,x,z] #sum model precip
                            bias[c-1,w,8] = bias[c-1,w,7]/bias[c-1,w,6] #bias


np.save('snotel_bias_wasatch', bias)                       
#%%
                        
            

ncar = bias[0,:w,8]
gfs = bias[1,:w,8]
hrrr = bias[2,:w,8]
nam = bias[3,:w,8]
sref_arw = bias[4,:w,8]
sref_nmb = bias[5,:w,8]



###############################################################################
##############   Calc biases for western US  ##################################
###############################################################################

#############  NCAR   ######
snotel_mean_ncar = np.mean(ncar)
snotel_stdev_ncar = np.std(ncar)


#### GFS #######
snotel_mean_gfs = np.mean(gfs)
snotel_stdev_gfs = np.std(gfs)

#####   HRRR   ############
snotel_mean_hrrr = np.mean(hrrr)
snotel_stdev_hrrr = np.std(hrrr)



#####  NAM   ############
snotel_mean_nam = np.mean(nam)
snotel_stdev_nam = np.std(nam)


#####  SREF ARW   ############
snotel_mean_sref_arw = np.mean(sref_arw)
snotel_stdev_sref_arw = np.std(sref_arw)

#####  SREF NMB   ############
snotel_mean_sref_nmb = np.mean(sref_nmb)
snotel_stdev_sref_nmb = np.std(sref_nmb)





#### Get mean prism precip for total accum plots ####
precip_prism = np.loadtxt('prism_ncar_dailymean.txt')



#####   ELevation Data   #########

#NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/random/wrfinput_d02'
#fh = Dataset(NCARens_file, mode='r')
#
#elevation = fh.variables['HGT'][:]
#lat_netcdf = fh.variables['XLAT'][:]
#long_netcdf = fh.variables['XLONG'][:] 
#%%
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
    
    
    
##Lbel Salt Lake City and Seattle
# Cities names and coordinates
cities = ['SLC', 'Seattle']
lat = [40.774, 47.593,0, 40.696, 40.694, 41.02, 41.05]
lon = [-111.913, -122.306, 0, -112.574, -112.131, -112.54, -112.23]

##############################################################################
##############################  Plot ##################################
##############################################################################



fig1=plt.figure(num=None, figsize=(11,18.5), dpi=800, facecolor='w', edgecolor='k')
levels_el = np.arange(1000,3500,100)
cmap = plt.cm.BrBG

title = 21
dot = 130
info = 16
label = 16
axis_title = 20

##################################     NCAR     #############################
ax = fig1.add_subplot(321)
plt.tight_layout()
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')


#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)

plt.text(x[3], y[3], 'Stansbury Mtns', fontsize = label+3, rotation = -85)
plt.text(x[4], y[4], 'Oquirrh Mtns', fontsize = label+3, rotation = -85)
plt.text(x[5], y[5], 'Wasatch\nRange', fontsize = label+3, rotation = 0)
ax.arrow(x[6], y[6], 42000, -65000, fc="k", ec="k",head_width=3000, head_length=3000)
ax.arrow(x[6], y[6], 33000, 0, fc="k", ec="k",head_width=3500, head_length=3500)


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
ax.set_title("NCAR ENS CTL", fontsize = title)
             
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5]) 
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % snotel_mean_ncar  +
#             'SD = %1.3f' % snotel_stdev_ncar, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')



##################################     GFS     #############################

ax = fig1.add_subplot(322)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
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
ax.set_title("GFS", fontsize = title)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % snotel_mean_gfs  +
#             'SD = %1.3f' % snotel_stdev_gfs, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')






##################################     HRRR     ###############################


ax = fig1.add_subplot(323)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
#precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[0,:,5], bias[0,:,4])
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
csAVG = map.scatter(xi,yi, c = bias[2,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("HRRR", fontsize = title)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % snotel_mean_hrrr  +
#             'SD = %1.3f' % snotel_stdev_hrrr, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')






##################################     NAM     #############################

ax = fig1.add_subplot(324)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
#precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
csAVG = map.scatter(xi,yi, c = bias[3,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("NAM-3km", fontsize = title)           
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % snotel_mean_nam  +
#             'SD = %1.3f' % snotel_stdev_nam, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')
             
             
             
             
##################################     SREF ARW     #############################

ax = fig1.add_subplot(325)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
#precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
cmap = plt.cm.BrBG
csAVG = map.scatter(xi,yi, c = bias[4,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("SREF ARW CTL", fontsize = title)
             
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
  
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xlabel('Bias Ratio', fontsize  = axis_title) 
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % snotel_mean_sref_arw  +
#             'SD = %1.3f' % snotel_stdev_sref_arw, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')


             
             
##################################     SREF NMB    #############################

ax = fig1.add_subplot(326)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

#For cities
x, y = map(lon, lat)
map.scatter(x[:3], y[:3], marker = '*',c = 'r', s = dot+80)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-10000, yc+5000, city, fontsize = label+3)
  
  
#precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
xi, yi = map(bias[0,:,5], bias[0,:,4])
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x2, y2 = map(long_netcdf[:,:], lat_netcdf[:,:])
csAVG2 = map.contourf(x2,y2,elevation[:,:], levels_el, cmap = 'Greys', zorder = 0)
cmap = plt.cm.BrBG
csAVG = map.scatter(xi,yi, c = bias[5,:,8], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = dot, vmin = 0.4, vmax = 2.2)#,norm=matplotlib.colors.LogNorm() )  
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("SREF NMB CTL", fontsize = title)
             
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xlabel('Bias Ratio', fontsize  = axis_title)  
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % snotel_mean_sref_nmb  +
#             'SD = %1.3f' % snotel_stdev_sref_nmb, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')
 

           
plt.tight_layout()
plt.savefig("../../../public_html/bias_snotel_allmodels_interp_2016_17_wasatch.png")
plt.show()














#%%


def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


colors = [
 (235, 246, 255),
 (214, 226, 255),
 (181, 201, 255),
 (142, 178, 255),
 (127, 150, 255),
 (114, 133, 248),
 (99, 112, 248),
 (0, 158,  30),
 (60, 188,  61),
 (179, 209, 110),
 (185, 249, 110),
 (255, 249,  19),
 (255, 163,   9),
 (229,   0,   0),
 (189,   0,   0),
 (129,   0,   0),
 (0,   0,   0)
 ]
   
cmap = make_cmap(colors, bit=True)






#############    SNOTEL  DAILY PRECIP   ########################################


fig1=plt.figure(num=None, figsize=(12,15), dpi=800, facecolor='w', edgecolor='k')
levels_el = np.arange(0,5000,100)
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7.5,  9,11, 13,  16, 22, 30,50]
levels_ticks = [0, 0.5,  1,  1.5, 2,  3,  4,  5,  7.5,  11,   16, 30]

#cmap = matplotlib.cm.get_cmap('pyart_NWSRef')

ax = fig1.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(bias[0,:,5], bias[0,:,4])
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
csAVG = map.scatter(xi,yi, c = bias[0,:,6]/bias[0,:,9], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 180)#,norm=matplotlib.colors.LogNorm() )  

csAVG2 = map.contourf(x2,y2,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)

#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("SNOTEL", fontsize = 28)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks = levels_ticks)
cbar.ax.set_xlabel('Mean-Daily Precipitation (mm)', fontsize = 24, labelpad = 15)
cbar.ax.set_xticklabels(levels_ticks, fontsize = 20) 

plt.tight_layout()
plt.savefig("../../../public_html/snotel_precip_2016_17_wasatch.png")



#%%






























