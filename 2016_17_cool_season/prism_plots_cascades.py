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



Cascades = [-123.96, 46.06, -120.38, 48.99]
WM = [-117.0, 43.0, -108.5, 49.0]
UT = [-114.7, 36.7, -108.9, 42.5]
CO = [-110.0, 36.0, -104.0, 41.9]
NU = [-112.4, 40.2, -111.2, 41.3]
#NU = [-113.4, 39.5, -110.7, 41.9]
NW = [-125.3, 42.0, -116.5, 49.1]
WE = [-125.3, 31.0, -102.5, 49.2]
US = [-125, 24.0, -66.5, 49.5]
SN = [-123.5, 33.5, -116.0, 41.0]

region = 'Cascades'

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

if region == 'Cascades':
    latlon = Cascades




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



###############################################################################
##############   Calc biases for western US  ##################################
###############################################################################

#############  NCAR   ######
avg1 = precip_ncar[17:453, 0:540]/precip_tot_ncar[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_ncar = np.average(avg)
bias_stdev_ncar = np.std(avg)





#####   HRRR   ############
avg1 = precip_hrrr[17:453, 0:540]/precip_tot_hrrr[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_hrrr = np.average(avg)
bias_stdev_hrrr = np.std(avg)


#####   gfs   ############
avg1 = precip_gfs[17:453, 0:540]/precip_tot_gfs[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_gfs = np.average(avg)
bias_stdev_gfs = np.std(avg)


#####   nam3km   ############
avg1 = precip_nam3km[17:453, 0:540]/precip_tot_nam3km[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_nam3km = np.average(avg)
bias_stdev_nam3km = np.std(avg)



#####   sref_arw   ############
avg1 = precip_sref_arw[17:453, 0:540]/precip_tot_sref_arw[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_sref_arw = np.average(avg)
bias_stdev_sref_arw = np.std(avg)



#####   sref_nmb   ############
avg1 = precip_sref_nmb[17:453, 0:540]/precip_tot_sref_nmb[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 10)]
bias_mean_sref_nmb = np.average(avg)
bias_stdev_sref_nmb = np.std(avg)








lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667


nws_precip_colors = [
    "#04e9e7",  # 0.01 - 0.10 inches
    "#019ff4",  # 0.10 - 0.25 inches
    "#0300f4",  # 0.25 - 0.50 inches
    "#02fd02",  # 0.50 - 0.75 inches
    "#01c501",  # 0.75 - 1.00 inches
    "#008e00",  # 1.00 - 1.50 inches
    "#D35400",  # 1.50 - 2.00 inches
    "#e5bc00",  # 2.00 - 2.50 inches
    "#fd9500",  # 2.50 - 3.00 inches
    "#fd0000",  # 3.00 - 4.00 inches
    "#d40000",  # 4.00 - 5.00 inches
    "#bc0000",  # 5.00 - 6.00 inches
    "#f800fd",  # 6.00 - 8.00 inches
    "#9854c6",  # 8.00 - 10.00 inches
    "#fdfdfd"   # 10.00+
]

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
#precip_colormap = matplotlib.colors.ListedColormap(colors)
'''

#############    PRISM  DAILY PRECIP   ########################################


fig1=plt.figure(num=None, figsize=(12,15), dpi=800, facecolor='w', edgecolor='k')
levels_el = np.arange(0,5000,100)
#levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7.5,  9,11, 13,  16, 22, 30,50]
levels_ticks = [0, 0.5,  1,  1.5, 2,  3,  4,  5,  7.5,  11,   16, 30]

#cmap = precip_colormap #matplotlib.cm.get_cmap('YlGnBu')

#cmap = matplotlib.cm.get_cmap('YlGnBu')
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

ax = fig1.add_subplot(111)
x, y = map(lons_prism, lats_prism)

csAVG = map.contourf(x,y,precip_tot_ncar, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  

#ax.set_cscale('log')
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()
ax.set_title("PRISM", fontsize = 28)               
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks = levels_ticks)
cbar.ax.set_xlabel('Mean-Daily Precipitation (mm)', fontsize = 24, labelpad = 15)
cbar.ax.set_xticklabels(levels_ticks, fontsize = 20) 

plt.tight_layout()
plt.savefig("../../../public_html/prism_precip_2016_17.png")



'''



#%%
#####   ELevation Data   #########

NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/random/wrfinput_d02'
fh = Dataset(NCARens_file, mode='r')

elevation = fh.variables['HGT'][:]
lat_netcdf = fh.variables['XLAT'][:]
long_netcdf = fh.variables['XLONG'][:] 

    
##Lbel Salt Lake City and Seattle
# Cities names and coordinates
cities = ['SLC', 'SEA']
lat = [40.774, 47.593]
lon = [-111.913, -122.306]


##############################################################################
##############################  Plot ##################################
##############################################################################

levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 10]

fig1=plt.figure(num=None, figsize=(11,17.5), dpi=800, facecolor='w', edgecolor='k')
levels_el = np.arange(-2500,2501,800)
thick = 1.1
cmap = plt.cm.BrBG

title = 19
dot = 110
info = 14
label = 16
axis_title = 18

##################################     NCAR     #############################
ax = fig1.add_subplot(321)
plt.tight_layout()
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i') 
#precip_tot_ncar = maskoceans(lons_prism, lats_prism, precip_tot_ncar)
x, y = map(lons_prism, lats_prism)
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
levels = levels
csAVG = map.contourf(x,y,precip_ncar/precip_tot_ncar, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)
map.drawcoastlines()  
map.drawstates()
map.drawcountries()


#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+160)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-3000, yc-30000, city, fontsize = label+1)
  
  
ax.set_title("NCAR ENS CTL", fontsize = title)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5]) 
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_ncar  +
#             'SD = %1.3f' % bias_stdev_ncar, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')



##################################     GFS     #############################

ax = fig1.add_subplot(322)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')


  
#precip_tot_gfs = maskoceans(lons_prism, lats_prism, precip_tot_gfs)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_gfs/precip_tot_gfs, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()


#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+160)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-3000, yc-30000, city, fontsize = label+1)
  
  
ax.set_title("GFS", fontsize = title)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_gfs  +
#             'SD = %1.3f' % bias_stdev_gfs, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')






##################################     HRRR     ###############################


ax = fig1.add_subplot(323)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

  
#precip_tot_hrrr = maskoceans(lons_prism, lats_prism, precip_tot_hrrr)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_hrrr/precip_tot_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N)) 
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1) 
cmap = plt.cm.BrBG 
map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()



#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+160)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-3000, yc-30000, city, fontsize = label+1)
  
  

ax.set_title("HRRR", fontsize = title)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_hrrr  +
#             'SD = %1.3f' % bias_stdev_hrrr, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')






##################################     NAM     #############################

ax = fig1.add_subplot(324)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')


#precip_tot_nam3km = maskoceans(lons_prism, lats_prism, precip_tot_nam3km)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_nam3km/precip_tot_nam3km, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG 

map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()



#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+160)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-3000, yc-30000, city, fontsize = label+1)
  
  
  
  
  
ax.set_title("NAM-3km", fontsize = title)           
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_nam3km  +
#             'SD = %1.3f' % bias_stdev_nam3km, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')
             
             
             
             
##################################     SREF ARW     #############################

ax = fig1.add_subplot(325)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')


#precip_tot_sref_arw = maskoceans(lons_prism, lats_prism, precip_tot_sref_arw)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_sref_arw/precip_tot_sref_arw, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)  
cmap = plt.cm.BrBG 


map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()



#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+160)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-3000, yc-30000, city, fontsize = label+1)
  
  
ax.set_title("SREF ARW CTL", fontsize = title)
             
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
  
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xlabel('Bias Ratio', fontsize  = axis_title) 
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_sref_arw  +
#             'SD = %1.3f' % bias_stdev_sref_arw, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')


             
             
##################################     SREF NMB    #############################

ax = fig1.add_subplot(326)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')



  
  
#precip_tot_sref_nmb = maskoceans(lons_prism, lats_prism, precip_tot_sref_nmb)
x, y = map(lons_prism, lats_prism)
levels = levels
csAVG = map.contourf(x,y,precip_sref_nmb/precip_tot_sref_nmb, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))  
csAVG2 = map.contour(x2,y2,elevation[0,:,:], levels_el,linewidths = thick, cmap = plt.cm.Greys, alpha = 1)
cmap = plt.cm.BrBG 


map.drawcoastlines()  ###RdYlBu
map.drawstates()
map.drawcountries()

#For cities
x, y = map(lon, lat)
map.scatter(x, y, marker = '*',c = 'r', s = dot+160)
# for each city,
for city, xc, yc in zip(cities, x, y):
# draw the city name in a yellow (shaded) box
  plt.text(xc-3000, yc-30000, city, fontsize = label+1)
  
  
  
ax.set_title("SREF NMB CTL", fontsize = title)
             
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1,  0.6,  0.8, 1,  1.4,  1.8,5])
cbar.ax.set_xlabel('Bias Ratio', fontsize  = axis_title)  
#cbar.ax.set_xlabel('mm', fontsize  = 14)
cbar.ax.set_xticklabels(['<0.5','0.6',  '0.8',  '1',  '1.4',  '1.8','>2'], fontsize = label) 
#plt.annotate('Mean = %1.3f\n' % bias_mean_sref_nmb  +
#             'SD = %1.3f' % bias_stdev_sref_nmb, xy=(0.015, .024),
#             xycoords='axes fraction', fontsize = info, backgroundcolor = 'w')
 

           
plt.tight_layout()
plt.savefig("../../../public_html/bias_prism_allmodels_interp_2016_17_cascades.png")














































