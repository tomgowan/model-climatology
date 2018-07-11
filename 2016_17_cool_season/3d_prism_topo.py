
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
import pygrib, os, sys, glob
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta
from tempfile import TemporaryFile
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from matplotlib import colors, ticker, cm
from mpl_toolkits.mplot3d import Axes3D



wmont = [-117.0, 43.0, -108.5, 49.0]
utah = [-114.7, 36.7, -108.9, 42.5]
colorado = [-110.0, 36.0, -104.0, 41.9]
wasatch = [-111, 40.9, -110.9, 41]
cascades = [-125.3, 42.0, -116.5, 49.1]
west = [-125.3, 30.0, -102.5, 50.01]






    #map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
    
    #### use ncl colormaps
    #def readNCLcm(name):
    #    '''Read in NCL colormap for use in matplotlib'''
    #    import os
    #    rgb, appending = [], False
    #    ### directories to NCL colormaps on yellowstone/cheyenne
    #    
    #    rgb_dir_ch = '/uufs/chpc.utah.edu/sys/installdir/ncl/6.4.0/lib/ncarg/colormaps'
    #
    #    fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')
    #    print fh
    #
    #    for line in fh.read().splitlines():
    # 
    #
    #        #if appending: rgb.append(map(float,line.split()))
    #            
    #        if appending: rgb.append(float(x) for x in line.split())
    #        
    #        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    #        print appending
    #    maxrgb = max([ x for y in rgb for x in y ])
    #    #if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    #    print rgb
    #    
    #    return rgb
    #
    #cmap = readNCLcm('GMT_topo')
    
    
region = 'wasatch'



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


map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

###############################################################################
################ Get elevation data   #########################################
###############################################################################


### 1km data  

res = ['1', '2', '4', '6', '9', '12', '18', '27']

num = 8    

for q in range(len(res)):                
    el_file = '/uufs/chpc.utah.edu/common/home/u1013082/elevation_data/%skm_elevation.nc' % res[q]
    print el_file
    fh = Dataset(el_file, mode='r')
    
    
    elevation = fh.variables['Band1'][:]
    lat = fh.variables['lat'][:]
    lon = fh.variables['lon'][:]
    

    
    
    #%%
    
    
    
    ###############################################################################
    ################ Plot figure ############################### ##################
    ###############################################################################
    
    #%%

    
    latnew = zeros((len(lat),len(lon)))
    lonnew = zeros((len(lat),len(lon)))
    for i in range(len(lon)):
        latnew[:,i] = lat
    for j in range(len(lat)):
        lonnew[j,:] = lon
    fig = plt.figure(figsize = (12,7))#(12,7))
    

    #ax = Axes3D(fig)
    
    
    ax2 = fig.add_subplot(111, projection='3d')
    #ax2.add_collection3d(map.drawcoastlines(linewidth=.7))
    #ax2.add_collection3d(map.drawstates(linewidth=.7))
    #ax2.add_collection3d(map.drawcountries(linewidth=.7))
    plt.axis('off')  
    #xi2, yi2 = map(lonnew[1700:2000,1950:2430],latnew[1700:2000,1950:2430])
    x,y = map(lonnew,latnew)
    #levels = np.arange(-100,2000.1,100)
    csAVG2 = ax2.plot_surface(x,y,elevation, cmap = 'terrain', rstride=1, cstride=1, alpha = .9, linewidth = 0)
    ax2.set_zlim(0, 13000)
    ax2.set_ylim(-200000,100000)

    ax2.set_xlim(-200000,100000)
    
    #cbar = plt.colorbar(csAVG2, shrink=0.25, aspect=10, pad = .07)#,anchor = (-.5, 0.42) )
    #cbar.ax.get_xaxis().labelpad = 15
    #cbar.ax.set_xlabel('Elevation (m)', fontsize = 14)
    fig.text(0.32,0.07,'Grid Spacing: ~%s km' %res[q],fontsize = 28)

    
    ax2.view_init(elev=37., azim=-90)
    plt.tight_layout()
    subplots_adjust(left=-0.20, bottom=-0.09, right=1.2, top=1.86, wspace=.07, hspace=.02) 
    plt.savefig('../../../public_html/%d_3dtopo_%skm.png' %(num,res[q]))
    num = num - 1


    #%%







