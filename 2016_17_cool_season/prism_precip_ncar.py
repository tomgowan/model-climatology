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





  
###############################################################################
##############   Read in ncar  and prism precip   #############################
###############################################################################

precip_ncar = zeros((621,1405))
precip_tot = zeros((621,1405))
num_days = 183

days = 183

#### Determine Dates
Date = zeros((days))
Date2= '20161001'
for i in range(0,days):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date[i] = int(Date3)
    
mem = 1    
y = 0
for i in range(0,days): 
    x = 0
    z = 0

    #### Make sure all ncar and prism files are present
    for j in range(13,37):
        NCARens_file = "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_prism_interp/2016_17_cool_season/ncarens/%08d" % Date[i] + "%02d" % j + "mem%02d" % mem + "ncarens_prism_interp.nc"
        if os.path.exists(NCARens_file):
            x = x + 1
    try:
        prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/2016_17_cool_season/PRISM_ppt_stable_4kmD2_%08d' % Date[i] + '_asc.asc'
        if os.path.exists(prism_file):
            z = 1
    except:
        pass
    
    try:
        prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/2016_17_cool_season/PRISM_ppt_provisional_4kmD2_%08d' % Date[i] + '_asc.asc'
        if os.path.exists(prism_file):
            z = 1
    except:
        pass


    print x
    if x == 24 and z == 1:
        y = y + 1
        for j in range(13,37):#32
            #############  NCAR   ############################
            NCARens_file = "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/model_prism_interp/2016_17_cool_season/ncarens/%08d" % Date[i] + "%02d" % j + "mem%02d" % mem + "ncarens_prism_interp.nc"
            print NCARens_file
            grbs = pygrib.open(NCARens_file)
            grb = grbs.select(name='Total Precipitation')[0]

            

            
            lat_ncar,lon_ncar = grb.latlons()
            tmpmsgs = grbs.select(name='Total Precipitation')
            msg = grbs[1]
            precip_vals = msg.values
            precip_vals = precip_vals*0.0393689*25.4
            precip_ncar = precip_ncar + precip_vals 

   

        ############### Prism #####################################
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/2016_17_cool_season/PRISM_ppt_stable_4kmD2_%08d" % Date[i] + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/2016_17_cool_season/PRISM_ppt_provisional_4kmD2_%08d" % Date[i] + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        
        precip_tot = precip_tot + precip
    
 

   
precip_tot = precip_tot/y
precip_ncar = precip_ncar/y
## Attempt to fix notation of lons so basemap understands it
lon_ncar = lon_ncar-360



###############################################################################
##############   Create lat lon grid for psirm    #############################
###############################################################################




lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667


##################   Saveprism and ncar array  ################################

np.savetxt('ncar_dailymean.txt', precip_ncar)
np.savetxt('prism_ncar_dailymean.txt', precip_tot)








#%%





###############################################################################
########################   Plot   #############################################
###############################################################################
west = [-125.3, 31.0, -102.5, 49.2]

latlon = west



cmap = matplotlib.cm.get_cmap('jet')
fig = plt.figure(figsize=(20,13))


levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
########################   NCAR   #############################################


ax = fig.add_subplot(231)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
precip_ncar = maskoceans(lons_prism, lats_prism, precip_ncar)
#map.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='deeppink', lakes=False)
csAVG = map.contourf(x,y,precip_ncar, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()

cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('NCAR Ensemble Control', fontsize = 18)
#cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)






########################   prism   #############################################
ax = fig.add_subplot(232)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

x, y = map(lons_prism, lats_prism)
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('PRISM', fontsize = 18)
#cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)












avg1 = precip_ncar[17:453, 0:540]/precip_tot[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 5)]
bias_mean = np.average(avg)


########################   bias   #############################################
ax = fig.add_subplot(233)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
#plt.text(1,1,'Mean bias = %1.3f' % bias_mean,rotation = 0, fontsize = 12)
#levels = np.arange(.45.000001,.1)
ax.set_title('NCAR/PRISM', fontsize = 18)
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_ncar/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
#set(cbar,'visible','off')

#cbar.ax.set_xlabel('Mean Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)
plt.annotate('Mean bias = %1.3f' % bias_mean, xy=(0.01, .01), xycoords='axes fraction', fontsize = 11)
plt.savefig("../../../public_html/ncar_prism_climo_2016_17.pdf")
plt.show()




#%%







'''


###############################################################################
############  plot hrr data also  #############################################
###############################################################################







precip_hrrr = np.loadtxt('hrrr_dailymean.txt')
precip_tot = np.loadtxt('prism_hrrr_dailymean.txt')



###############################################################################
########################   Plot   #############################################
###############################################################################




cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
levels = np.arange(.0001,37,.5)
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
########################   hrrr   #############################################


ax = fig.add_subplot(234)
#map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
precip_hrrr = maskoceans(lons_prism, lats_prism, precip_hrrr)
#map.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='deeppink', lakes=False)
csAVG = map.contourf(x,y,precip_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()

cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('HRRR', fontsize = 18)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)






########################   prism   #############################################
ax = fig.add_subplot(235)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')

x, y = map(lons_prism, lats_prism)
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('PRISM', fontsize = 18)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)












### Calcualte bias mean of whole array (only include data from the WESTERN US)
avg1 = precip_hrrr[17:453, 0:540]/precip_tot[17:453, 0:540]
avg = avg1[(avg1 > 0.1) & (avg1 < 5)]
bias_mean = np.average(avg)

########################   bias   #############################################
ax = fig.add_subplot(236)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]

#levels = np.arange(.45.000001,.1)
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_hrrr/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.title('HRRR/PRISM', fontsize = 18)
cbar.ax.set_xlabel('Mean Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)
#leg = ([], [], label='Mean bias = %1.3f' % bias_mean)
#plt.legend(handles = [leg],loc = "lower left")

#plt.text(.5,.5,'Mean bias = %1.3f' % bias_mean,rotation = 0, fontsize = 12)
plt.annotate('Mean bias = %1.3f' % bias_mean, xy=(0.01, .01), xycoords='axes fraction', fontsize = 11)
plt.savefig("./plots/hrrr_ncar_prism_climo_%s" % region + ".pdf")
plt.show()

'''














