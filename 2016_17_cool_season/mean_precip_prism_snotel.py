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
utah = [-114.7, 36.7, -108.9, 42.5]
colorado = [-110.0, 36.0, -104.0, 41.9]
wasatch = [-113.4, 39.5, -110.7, 41.9]
cascades = [-125.3, 42.0, -116.5, 49.1]
west = [-125.3, 31.0, -102.5, 49.2]


region = 'west'



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

                       

                        
            

ncar = bias[0,:w,8]
gfs = bias[1,:w,8]
hrrr = bias[2,:w,8]
nam = bias[3,:w,8]
sref_arw = bias[4,:w,8]
sref_nmb = bias[5,:w,8]





#### Get mean prism precip for total accum plots ####
precip_prism = np.loadtxt('prism_ncar_dailymean.txt')



#####   ELevation Data   #########

NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/random/wrfinput_d02'
fh = Dataset(NCARens_file, mode='r')

elevation = fh.variables['HGT'][:]
lat_netcdf = fh.variables['XLAT'][:]
long_netcdf = fh.variables['XLONG'][:] 



lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667
    
    
    

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



'''


#############    SNOTEL  AND PRISM DAILY PRECIP   ########################################


fig1=plt.figure(num=None, figsize=(17,12), dpi=800, facecolor='w', edgecolor='k')
levels_el = np.arange(0,5000,100)
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7.5,  9,11, 13,  16, 22, 30,50]
levels_ticks = [0, 0.5,  1,  1.5, 2,  3,  4,  5,  7.5,  11,   16, 30]



ax = fig1.add_subplot(121)
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




ax = fig1.add_subplot(122)
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_prism, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("PRISM", fontsize = 28)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks = levels_ticks)
cbar.ax.set_xlabel('Mean-Daily Precipitation (mm)', fontsize = 24, labelpad = 15)
cbar.ax.set_xticklabels(levels_ticks, fontsize = 20) 

plt.tight_layout()
plt.savefig("../../../public_html/snotel_prism_precip_2016_17.png")


'''


fig = plt.figure(figsize=(17,13))


#levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7.5,  9,11, 13,  16, 22, 30,50]
levels_ticks = [0, 0.5,  1,  1.5, 2,  3,  4,  5,  7.5,  11,   16, 30]
levels_el = np.arange(0,4801,200)
#######################   PRISM event frequency  #############################################

#%%
ax = fig.add_subplot(122)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_prism, levels,  cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
props = dict(boxstyle='square', facecolor='white', alpha=1)
ax.text(80000, 100000, '(b) PRISM', fontsize = 28, bbox = props)





ax = fig.add_subplot(121)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
xi, yi = map(bias[0,:,5], bias[0,:,4])
csAVG = map.scatter(xi,yi, c = bias[0,:,6]/bias[0,:,9], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 120)#,norm=matplotlib.colors.LogNorm() )  
csAVG2 = map.contourf(x,y,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
#cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks = np.arange(0.000001, 1.0001, 0.1))
#cbar.ax.tick_params(labelsize=12)

#cbar.ax.set_xticklabels(np.arange(0,1.0001,0.1), fontsize = 17)
#cbar.ax.set_xlabel('Frequnecy of > 2.54 mm/24-h', fontsize = 20, labelpad = 15)

ax.text(80000, 100000, '(a) SNOTEL', fontsize = 28, bbox = props)


ax2 = fig.add_axes([.025,.09,.45,.04])
ax3 = fig.add_axes([.525,.09,.45,.04])
cbar = fig.colorbar(csAVG,cax=ax3, orientation='horizontal',ticks = levels_ticks)
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_xticklabels(levels_ticks, fontsize = 20)
cbar.ax.set_xlabel('Mean Daily Precipitation (mm)', fontsize = 24, labelpad = 15)

cbar = fig.colorbar(csAVG2,cax=ax2, orientation='horizontal',ticks = np.arange(0,4801,600))
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_xticklabels(np.arange(0,4801,600), fontsize = 20)
cbar.ax.set_xlabel('Elevation (m)', fontsize = 24, labelpad = 15)
plt.tight_layout()
plt.savefig("../../../public_html/snotel_prism_precip_2016_17.png")






