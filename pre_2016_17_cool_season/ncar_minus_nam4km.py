
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
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from matplotlib import colors, ticker, cm






wmont = [-117.0, 43.0, -108.5, 49.0]
utah = [-114.7, 36.7, -108.9, 42.5]
colorado = [-110.0, 36.0, -104.0, 41.9]
wasatch = [-113.4, 39.5, -110.7, 41.9]
cascades = [-125.3, 42.0, -116.5, 49.1]
west = [-125.3, 31.0, -102.5, 49.2]


region = sys.argv[1]



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
    

nearest_hrrr = zeros((1033,842))
totalprecip = zeros((785,650))
ncar_minus_hrrr_file = zeros((785,650))
row_hrrr = zeros((4))
col_hrrr = zeros((4))
Date = zeros((183))











######################################      NAM4k
  
num_days = 0
num_hours = 1
grlatlon = pygrib.open('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/nam4km/regridded.2016012300F30namhires.grib2')

glatlon = grlatlon.select(name='Total Precipitation')[0]
hrrr_precip_test = glatlon.values
hrrr_lat,hrrr_lon = glatlon.latlons()



for nam4k_file in glob.glob('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/nam4km/regridded*'):
    print(nam4k_file)

    try:
        grbs = pygrib.open(nam4k_file)
        grb = grbs.select(name='Total Precipitation')[0]
        nam4k_precip = grb.values*0.0393689*25.4
        totalprecip = totalprecip + nam4k_precip
        num_hours = num_hours + 1
        print num_hours
    

    except:
        pass
    



num_days = num_hours/8  #8 files per day

nam4k_regrid = totalprecip/num_days







#################################################################  NCAR
Date2= '20151001'


for i in range(0,183):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date[i] = int(Date3)

x = 0

w = 706

hours = 0

totalprecip_ncar = zeros((985,1580))

for i in range(0,183):
    for j in range(13,37):
        NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/ncarens/%08d' % Date[i] + '00/ncar_3km_%08d' % Date[i] + '00_mem1_f0%02d.grb2' %  j
        
        if os.path.exists(NCARens_file):


            hours = hours + 1
            print hours 
            if i <= 11:
                grbs = pygrib.open(NCARens_file)
                tmpmsgs = grbs.select(name='Total Precipitation')
                msg = grbs[12]

            else:
                grbs = pygrib.open(NCARens_file)
                tmpmsgs = grbs.select(name='Total Precipitation')
                msg = grbs[16]

        precip_vals = msg.values
        NCAR_precip = precip_vals*0.0393689*25.4
        totalprecip_ncar = totalprecip_ncar + NCAR_precip

days = hours/24


ncar_precip = totalprecip_ncar/days

###### NCAR minus HRRR grid

ncar_minus_nam4k = ncar_precip[0:785,0:650]- nam4k_regrid

        
cmap = plt.cm.BrBG


        
        


fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
xi, yi = map(hrrr_lon, hrrr_lat)
#levels = [-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1,   0 ,0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
#levels = np.arange(-2.5,2.55,0.1)
levels = [-2.0,-1.6,-1.3,-1, -.8,-.6,-.4, -0.3, -.2,  -0.1,  0 ,0.1, .2,0.3, .4, .6,.8 ,1,1.3,1.6,2.0]
csAVG = map.contourf(xi,yi,ncar_minus_nam4k, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels, cmap.N),vmin = -1.6, vmax = 1.6,extend = 'both')

map.drawcoastlines()
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks=[-1.6,-1.3,-1, -.8,-.6,-.4,-.3, -.2, -.1  , 0 ,.1, .2,.3, .4, .6,.8 ,1,1.3,1.6])
cbar.ax.set_xticklabels(['-1.6','-1.3','-1', '-.8','-.6','-.4','-.3', '-.2','-.1',    '0 ','.1', '.2','.3', '.4', '.6','.8', '1','1.3','1.6'])
cbar.ax.set_xlabel('mm', fontsize  = 14)
ax.set_title('NCARens Control Minus NAM-4km Daily Precip', fontsize = 18)


plt.savefig("./plots/ncar_minus_nam4km_daily_precip_octtomar.pdf")
plt.show()























