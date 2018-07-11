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
from tempfile import TemporaryFile







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
  
'''
nearest_hrrr = zeros((1033,842))
totalprecip = zeros((785,650))
ncar_minus_hrrr = zeros((785,650))
ncar_minus_hrrr_file = zeros((785,650))
row_hrrr = zeros((4))
col_hrrr = zeros((4))
Date = zeros((183))


Date2= '20150930'


for i in range(0,183):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date[i] = int(Date3)


######################### NAM12k


num_days12k = 0
num_hours = 0


totalprecip = zeros((428,614))
for i in range(0,len(Date)): 

            for r in range(5,13):
                j = r*3
                nam12k_File = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/nam-12km/%08d' % Date[i] + '/nam_218_%08d' % Date[i] + '_0000_0%02d' % j + '.grb'
                try:
                    grbs = pygrib.open(nam12k_File)
                    try:                    
                        msg = grbs[4]
                        
                        num_hours = num_hours + 1
                        nam12k_precip = (msg.values)*0.0393689*25.4
                        totalprecip = totalprecip + nam12k_precip
                        print nam12k_File
                    except:
                        msg  = grbs[3]
                        num_hours = num_hours + 1
                        nam12k_precip = (msg.values)*0.0393689*25.4
                        totalprecip = totalprecip + nam12k_precip
                        print nam12k_File
                except: 
                    pass

num_days12k = num_hours/8.00

nam12k = totalprecip/num_days12k

NAM12k_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/nam-12km/20151003/nam_218_20151003_0000_018.grb'

grbs = pygrib.open(NAM12k_file)
grb = grbs.select(name='Total Precipitation')[0]
NAM12k_lat,NAM12k_lon = grb.latlons()



###############################   NAM4km

num_days4k = 0
num_hours = 0
totalprecip = zeros((1025,1473))


for i in range(0,len(Date)): 
            for r in range(5,13):
                j = r*3
                try:
                    nam4k_File = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/nam-4km/%08d' % Date[i] + '/%08d' % Date[i] + '00F%02d' % j + 'namhires.grib2'
                    grbs = pygrib.open(nam4k_File)
                    grb = grbs.select(name='Total Precipitation')[0]
                    nam4k_precip = grb.values*0.0393689*25.4
                    totalprecip = totalprecip + nam4k_precip
                    num_hours = num_hours + 1
                    print nam4k_File
                except:
                    pass


num_days4k = num_hours/8.00

nam4k = totalprecip/num_days4k

NAM4k_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/nam-4km/20151003/2015100300F21namhires.grib2'

grbs = pygrib.open(NAM4k_file)
grb = grbs.select(name='Total Precipitation')[0]
NAM4k_lat,NAM4k_lon = grb.latlons()



######################################      HRRR  
  
num_days = 0
num_hours = 0


totalprecip = zeros((1033,842))


for i in range(0,len(Date)): 
    for j in range(3,15):
            try:
                HRRR_File = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%08d' % Date[i] + '10F%02d' % j + 'hrrr.grib2'
                num_hours = num_hours + 1
                grbs = pygrib.open(HRRR_File)
                msg = grbs[6]
                hrrr_precip = (msg.values)*0.0393689*25.4
                totalprecip = totalprecip + hrrr_precip
                print HRRR_File
            except:
                print 'hi'
                

            

    for j in range(3,15):
            try:
                HRRR_File = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%08d' % Date[i] + '22F%02d' % j + 'hrrr.grib2'
                num_hours = num_hours + 1
                grbs = pygrib.open(HRRR_File)
                msg = grbs[6]
                hrrr_precip = (msg.values)*0.0393689*25.4
                totalprecip = totalprecip + hrrr_precip
                print HRRR_File
            except:
                pass



num_dayshrrr = num_hours/24.00

hrrr = totalprecip/num_dayshrrr

Hfile = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/2016020322F08hrrr.grib2'
grbs = pygrib.open(Hfile)
grb = grbs.select(name='Total Precipitation')[0]
HRRR_lat,HRRR_lon = grb.latlons()







#################################################################  NCAR

x = 0



hours = 0

totalprecip_ncar = zeros((985,1580))

for i in range(0,len(Date)):
    for j in range(13,37):
        NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/ncarens/%08d' % Date[i] + '00/ncar_3km_%08d' % Date[i] + '00_mem1_f0%02d.grb2' %  j
        
        if os.path.exists(NCARens_file):


            hours = hours + 1
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
        print NCARens_file

daysncar = hours/24.00


ncar = totalprecip_ncar/daysncar


NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/ncarens/2016010100/ncar_3km_2016010100_mem9_f047.grb2'

grbs = pygrib.open(NCARens_file)
grb = grbs.select(name='Total Precipitation')[0]
NCAR_lat,NCAR_lon = grb.latlons()

'''


print num_days12k
print num_days4k
print num_dayshrrr
print daysncar

'''
hrrr_climo = TemporaryFile()
NCAR_climo = TemporaryFile()
NAM4k_climo = TemporaryFile()
NAM12k_climo = TemporaryFile()

np.save(hrrr_climo, ncar)
np.save(NCAR_climo, hrrr)
np.save(NAM4k_climo, nam4k)
np.save(NAM12k_climo, nam12k)
'''
###############################################################################
###################    Plots    ###############################################
###############################################################################
sub = 220
precip_per_day = [ncar, nam4k, hrrr, nam12k]

title = ['NCARens Control', 'NAM4km', 'HRRR', 'NAM12km']

fig = plt.figure(figsize=(14,18))



for i  in range(4):
    sub = sub + 1
    ax = fig.add_subplot(sub)
    map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
    if i == 0:
        xi, yi = map(NCAR_lon, NCAR_lat)
    if i == 1:
        xi, yi = map(NAM4k_lon, NAM4k_lat)
    if i == 2:
        xi, yi = map(HRRR_lon, HRRR_lat)
    if i == 3:
        xi, yi = map(NAM12k_lon, NAM12k_lat)
        
        
    levels = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 14, 18, 24, 1000]
    csAVG = map.contourf(xi,yi,precip_per_day[i],levels, colors=('w', 'lightgrey', 'dimgray','palegreen',
                                             'limegreen', 'g','blue', 
                                            'royalblue', 'lightskyblue', 'cyan', 'navajowhite', 
                                            'darkorange', 'orangered', 'sienna', 'maroon'))
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()

  
        

    ax.set_title(title[i], fontsize = 22)
    plt.tight_layout()
    #if i == 2 or i == 3:
    #
    
    if i == 3 or i == 2:

        cbar = map.colorbar(csAVG, pad="5%", ticks=[0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 14, 18, 24], location = 'bottom')
        cbar.ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1', '1.5', '2', '3', '4', '6', '8', '10', '14', '18', '24+'], fontsize = 12)
        cbar.ax.set_xlabel('Mean Daily Precipitation (mm)', fontsize = 14)


plt.savefig("./plots/shortrange_climo.pdf")
plt.show()



'''

sub = 220
precip_per_day = [ncar, nam4k, hrrr, nam12k]



i = 0 
fig, axes = plt.subplots(nrows = 2, ncols = 2)

title = ['NCARens Control', 'NAM4km', 'HRRR', 'NAM12km']


for ax  in axes.flat:
    
    map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
    if i == 0:
        xi, yi = map(NCAR_lon, NCAR_lat)
    if i == 1:
        xi, yi = map(NAM4k_lon, NAM4k_lat)
    if i == 2:
        xi, yi = map(HRRR_lon, HRRR_lat)
    if i == 3:
        xi, yi = map(NAM12k_lon, NAM12k_lat)
        
        
    levels = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 14, 18, 24, 1000]
    csAVG = map.contourf(xi,yi,precip_per_day[i],levels, colors=('w', 'lightgrey', 'dimgray','palegreen',
                                             'limegreen', 'g','blue', 
                                            'royalblue', 'lightskyblue', 'cyan', 'navajowhite', 
                                            'darkorange', 'orangered', 'sienna', 'maroon'))
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()

  
        

    ax.set_title(title[i], fontsize = 22)
 
    i = i + 1
    #if i == 2 or i == 3:
    #




plt.savefig("./plots/shortrange_climo_test.pdf")
plt.show()

'''






