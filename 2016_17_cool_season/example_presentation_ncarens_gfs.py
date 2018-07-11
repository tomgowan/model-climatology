import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pygrib, os, sys
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta

maxx = zeros((802,722))
minn = zeros((802,722))
countlow = zeros((802,722))
countmed = zeros((802,722))
counthigh = zeros((802,722))
preciptotalMEAN = zeros((802,722))
preciptotalMax = zeros((11,802,722))
preciptotalMin = zeros((11,802,722))
PrecipProbLow = 0.01
PrecipProbMed = 1
PrecipProbHigh = 2
Prob = 'Greater'

Date = '20170122'
Date2 = Date[0:8]

#####REGIONS(lowlow, lowlat, upperlon, upperlat)


WM = [-117.0, 43.0, -108.5, 49.0]
UT = [-114.7, 36.7, -108.9, 42.5]
CO = [-110.0, 36.0, -104.0, 41.9]
NU = [-113.4, 39.5, -110.7, 41.9]
NW = [-125.3, 42.0, -116.5, 49.1]
WE = [-125.3, 31.0, -102.5, 49.2]
SN = [-123.5, 33.5, -116.0, 41.0]

region = 'UT'

if region == 'WM':
    latlon = WM
    
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

#############Probabbility Maps (Greater than 0.01, 1, 2)

for i in range(1,2):#11
    preciptotal = zeros((802,722))
    for j in range(13,37):

        NCARens_file = '/uufs/chpc.utah.edu/common/home/horel-group/archive/%s' % Date2 + '/models/ncarens/ncar_3km_west_us_%s' % Date2 + '00_mem%d' % i + '_f0%02d.grb2' %  j
        print NCARens_file
        grbs = pygrib.open(NCARens_file)
        grb = grbs.select(name='Total Precipitation')[0]
        


        lat,lon = grb.latlons()

        grbs = pygrib.open(NCARens_file)
        #for g in grbs:
        #    print g

        tmpmsgs = grbs.select(name='Total Precipitation')
        msg = grbs[4]

        precip_vals = msg.values
        precip = precip_vals*0.0393689*25.4
        preciptotal = preciptotal + precip

        preciptotalMax[i,:,:] = preciptotalMax[i,:,:]+precip
        
        
        
        
preciptotalsref = zeros((155,185))
for hour in range(12,34,3):
                    sref_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_raw_output/2016_17_coolseason/sref/jan2017/%s' % Date2 + '03arw_ctlF%2d' % hour + '.grib2'
                    grbs = pygrib.open(sref_file)
                    msg = grbs[6]
                    precip_vals = msg.values
                    precip = precip_vals*0.0393689*25.4
                    
                    preciptotalsref = preciptotalsref + precip
                    
                    latsref,lonsref = msg.latlons()
                        



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





fig = plt.figure(figsize=(17,14))


#levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7.5,  9,11, 13,  16,19, 22,25, 30,35, 40, 45, 50, 60, 70, 80]
levels_ticks = [0, 0.5,  1,  1.5, 2,  3,  4,  5,  7.5,  11,   16, 22, 30, 40, 50, 70]
levels_el = np.arange(0,4801,200)
#######################   PRISM event frequency  #############################################

#%%
ax = fig.add_subplot(121)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon, lat)
csAVG = map.contourf(x,y,preciptotal, levels,  cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()







ax = fig.add_subplot(122)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lonsref, latsref)

csAVG = map.contourf(x,y, preciptotalsref, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))#,norm=matplotlib.colors.LogNorm() )  

map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()

ax2 = fig.add_axes([.2,.09,.6,.04])

cbar = fig.colorbar(csAVG,cax=ax2, orientation='horizontal',ticks = levels_ticks)
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_xticklabels(levels_ticks, fontsize = 20)
cbar.ax.set_xlabel('Precipitation (mm)', fontsize = 24, labelpad = 15)


plt.tight_layout()
plt.savefig("../../../public_html/exampleplot_ncar_sref.png")







                        
        
        
        
        
        
        
        
