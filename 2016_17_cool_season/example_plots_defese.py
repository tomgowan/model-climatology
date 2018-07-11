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

for i in range(1,11):#11
    preciptotal = zeros((802,722))
    for j in range(0,49):

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
        precip = precip_vals*0.0393689
        preciptotal = preciptotal + precip

        preciptotalMax[i,:,:] = preciptotalMax[i,:,:]+precip
        preciptotalMin[i,:,:] = preciptotalMin[i,:,:]+precip
        
    preciptotalMEAN = preciptotalMEAN+preciptotal
    maxx = preciptotalMax[1,:,:]
    minn = preciptotalMin[1,:,:]

## Low

    for a in range(0,802):
        for b in range(0,722):
            if preciptotal[a,b] > PrecipProbLow:
                countlow[a,b] = countlow[a,b]+1
                    
### Med

    for c in range(0,802):
        for d in range(0,722):
            if preciptotal[c,d] > PrecipProbMed:
                countmed[c,d] = countmed[c,d]+1
                    
### High

    for e in range(0,802):
        for f in range(0,722):
            if preciptotal[e,f] > PrecipProbHigh:
                counthigh[e,f] = counthigh[e,f]+1


    for g in range(0,802):
            for h in range(0,722):
                if preciptotalMax[i,g,h] > maxx[g,h]:
                    maxx[g,h] = preciptotalMax[i,g,h]


    for m in range(0,802):
            for n in range(0,722):
                if preciptotalMin[i,m,n] < minn[m,n]:
                    minn[m,n] = preciptotalMin[i,m,n]       



#Convert into percentage

countlow = countlow*10
countmed = countmed*10
counthigh = counthigh*10



precipAVG = preciptotalMEAN/10

#####   ELevation Data   #########

NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/random/wrfinput_d02'
fh = Dataset(NCARens_file, mode='r')

elevation = fh.variables['HGT'][:]
lat_netcdf = fh.variables['XLAT'][:]
long_netcdf = fh.variables['XLONG'][:] 

levels_el = np.arange(0,5000,100)

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


#%%

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

from matplotlib import animation
import matplotlib.animation as animation
import types


################################################11111111111111111111

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(231)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(lon, lat)
levels = [0, 0.01, 0.1, 0.2, 0.4, 0.75, 1, 1.25, 1.5, 2.0, 2.5, 3, 4, 5, 7, 100]
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
csAVG2 = map.contourf(x2,y2,elevation[0,:,:], levels_el, cmap = 'Greys')
csAVG = map.contourf(xi,yi,precipAVG,levels, colors=('w', 'lightgrey', 'dimgray','palegreen',
                                           'limegreen', 'g','blue', 
                                           'royalblue', 'lightskyblue', 'cyan', 'navajowhite', 
                                            'darkorange', 'orangered', 'sienna', 'maroon'), alpha = 0.6)
map.drawcoastlines()
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks=[0, 0.01, 0.1, 0.2, 0.4, 0.75, 1, 1.25, 1.5, 2.0, 2.5, 3, 4, 5, 7])
cbar.ax.set_xticklabels(['0', '0.01', '0.1', '0.2', '0.4', '0.75', '1', '1.25', '1.5', '2.0', '2.5', '3', '4', '5', '7+'])
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_xlabel('Inches')
ax.set_title('Ensemble Mean 48-hr Precip')



################################################222222222222222222222




ax = fig.add_subplot(232)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(lon, lat)
levels = [0, 0.01, 0.1, 0.2, 0.4, 0.75, 1, 1.25, 1.5, 2.0, 2.5, 3, 4, 5, 7, 100]
csmax = map.contourf(xi,yi,maxx,levels,cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N)) 
map.drawcoastlines()
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csmax, location='bottom', pad="5%", ticks=[0, 0.01, 0.1, 0.2, 0.4, 0.75, 1, 1.25, 1.5, 2.0, 2.5, 3, 4, 5, 7])
cbar.ax.set_xticklabels(['0', '0.01', '0.1', '0.2', '0.4', '0.75', '1', '1.25', '1.5', '2.0', '2.5', '3', '4', '5', '7+'])
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_xlabel('Inches')
ax.set_title('Ensemble Max 48-hr Precip')







################################################33333333333333333333333








ax = fig.add_subplot(233)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(lon, lat)
levels = [0, 0.01, 0.1, 0.2, 0.4, 0.75, 1, 1.25, 1.5, 2.0, 2.5, 3, 4, 5, 7, 100]
csmin = map.contourf(xi,yi,minn,levels,  colors=('w', 'lightgrey', 'dimgray','palegreen',
                                             'limegreen', 'g','blue', 
                                            'royalblue', 'lightskyblue', 'cyan', 'navajowhite', 
                                            'darkorange', 'orangered', 'sienna', 'maroon'))
map.drawcoastlines()
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csmin, location='bottom', pad="5%", ticks=[0, 0.01, 0.1, 0.2, 0.4, 0.75, 1, 1.25, 1.5, 2.0, 2.5, 3, 4, 5, 7])
cbar.ax.set_xticklabels(['0', '0.01', '0.1', '0.2', '0.4', '0.75', '1', '1.25', '1.5', '2.0', '2.5', '3', '4', '5', '7+'])
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_xlabel('Inches')
ax.set_title('Ensemble Min 48-hr Precip')




##########################################################################################3

#####                             PROB PLOTS

##########################################################################################

######################### LOW



ax = fig.add_subplot(234)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon, lat)                                           
levels = [-5,5,15,25,35,45,55,65,75,85,95,105]
cslow = map.contourf(x,y,countlow,levels, colors=('w', 'darkslateblue', 'darksage','mediumseagreen',
                                             'lightgoldenrodyellow', 'tan', 
                                            'gold', 'orange', 'orangered','sienna', 
                                            'maroon'))
map.drawcoastlines()
map.drawstates()
map.drawcountries()
cbar = map.colorbar(cslow, location='bottom', pad="5%",ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.ax.set_xlabel('Percentage (%)')
plt.tight_layout(pad = 2)
ax.set_title('48-hr Precip ' + str(Prob) + ' than ' + str(PrecipProbLow) + '"')



######################### MED


ax = fig.add_subplot(235)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon, lat)                                          
levels = [-5,5,15,25,35,45,55,65,75,85,95,105]
csmed = map.contourf(x,y,countmed,levels, colors=('w', 'darkslateblue', 'darksage','mediumseagreen',
                                             'lightgoldenrodyellow', 'tan', 
                                            'gold', 'orange', 'orangered','sienna', 
                                            'maroon'))
map.drawcoastlines()
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csmed, location='bottom', pad="5%",ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.ax.set_xlabel('Probability (%)\n' +'\n' 'Initialized at ' + str(Date) + 
            '00Z\n NCAR Ensemble (10 members)')
plt.tight_layout(pad = 2)
ax.set_title('48-hr Precip > ' + str(PrecipProbMed) + '"')



######################### HIGH



ax = fig.add_subplot(236)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon, lat)                                          
levels = [-5,5,15,25,35,45,55,65,75,85,95,105]
cshigh = map.contourf(x,y,counthigh,levels, colors=('w', 'darkslateblue', 'darksage','mediumseagreen',
                                             'lightgoldenrodyellow', 'tan', 
                                            'gold', 'orange', 'orangered','sienna', 
                                            'maroon'))
map.drawcoastlines()
map.drawstates()
map.drawcountries()
cbar = map.colorbar(cshigh, location='bottom', pad="5%",ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.tick_params(bottom='off')  
cbar.ax.set_xlabel('Probability (%)')
plt.tight_layout(pad = 2)
ax.set_title('48-hr Precip > ' + str(PrecipProbHigh) + '"')
plt.tight_layout()

plt.savefig("EnsemblePrecipPlot_test.png")
plt.savefig("../public_html/ncar_plot_defence.png")






############################################################################################3






