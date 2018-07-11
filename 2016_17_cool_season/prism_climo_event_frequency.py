
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


WM = [-117.0, 43.0, -108.5, 49.0]
UT = [-114.7, 36.7, -108.9, 42.5]
CO = [-110.0, 36.0, -104.0, 41.9]
NU = [-113.4, 39.5, -110.7, 41.9]
NW = [-125.3, 42.0, -116.5, 49.1]
WE = [-125.3, 31.0, -102.5, 49.2]
US = [-125, 24.0, -66.5, 49.5]
SN = [-123.5, 33.5, -116.0, 41.0]

region = 'WE'

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



precip_freq = zeros((621,1405))

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
    

###############################################################################
############ Read in SNOTEL data   ############################################
###############################################################################
inhouse_data = zeros((649,186))

x = 0
q = 0
v = 0
i = 0   


links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/snotel_precip_2016_2017_qc.csv"]

#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        

data = zeros((len(links),798,185))

         
for c in range(1):
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

data[np.isnan(data)] = -9999

inhouse_data = data[0,:,:] 




#######   Calculate frequency of precip days    ###########
freq_snotel = zeros((798,3))
freq_snotel[:,0:2] = inhouse_data[:,1:3]

for j in range(798):
    x = 0
    for i in range(3,185):
        if inhouse_data[j,i] > -1:
            x = x + 1
            if inhouse_data[j,i] >= 2.54:
                freq_snotel[j,2] = freq_snotel[j,2] + 1
    freq_snotel[j,2] = freq_snotel[j,2]/x
    
                





###############################################################################
##############   Read in prism precip   #############################
###############################################################################
 


for i in range(0,num_days): 

    x = 0
    y = 0

    #### Make sure all prism files are present
    try:
        prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/2016_17_cool_season/PRISM_ppt_stable_4kmD2_%08d' % Date[i] + '_asc.asc'
        if os.path.exists(prism_file):
            y = 1
    except:
        pass
    
    try:
        prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/2016_17_cool_season/PRISM_ppt_provisional_4kmD2_%08d' % Date[i] + '_asc.asc'
        if os.path.exists(prism_file):
            y = 1
    except:
        pass

    print y
    if  y == 1:
        ############### Prism #####################################
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/2016_17_cool_season/PRISM_ppt_stable_4kmD2_%08d" % Date[i] + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/2016_17_cool_season/PRISM_ppt_provisional_4kmD2_%08d" % Date[i] + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        

        
    for i in range(621):
        for j in range(1405):
            if precip[i,j] >= 2.54:
                precip_freq[i,j] = precip_freq[i,j] + 1
#%%                
precip_freq = precip_freq/183
#%%
for i in range(621):
    for j in range(1405):
        if precip_freq[i,j] == 0:
            precip_freq[i,j] = -1
#%%
   



###############################################################################
##############   Create lat lon grid for psirm    #############################
###############################################################################




lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667


##################   Save prism= array  ################################
#np.savetxt('prism_dailymean.txt', precip_tot)


#%%


   
#cmap = make_cmap(colors, bit=True)


######  Elevation for map
NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/random/wrfinput_d02'
fh = Dataset(NCARens_file, mode='r')

elevation = fh.variables['HGT'][:]
lat_netcdf = fh.variables['XLAT'][:]
long_netcdf = fh.variables['XLONG'][:]


###############################################################################
########################   Plot   #############################################
###############################################################################
#%%

'''

cmap = matplotlib.cm.get_cmap('terrain')
fig = plt.figure(figsize=(17,12))

ax2 = fig.add_axes([0.0,0.19,1.2,0.62])

#levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
levels = np.arange(0.000001, 1.0001, 0.025)
levels_el = np.arange(0,5000,100)
#######################   PRISM event frequency  #############################################


ax = fig.add_subplot(122)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_freq, levels,  cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
plt.title('PRISM', fontsize = 20)




ax = fig.add_subplot(121)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
xi, yi = map(freq_snotel[:,1], freq_snotel[:,0])
csAVG = map.scatter(xi,yi, c = freq_snotel[:,2], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 120)#,norm=matplotlib.colors.LogNorm() )  
csAVG2 = map.contourf(x,y,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks = np.arange(0.000001, 1.0001, 0.1))
cbar.ax.tick_params(labelsize=12)

cbar.ax.set_xticklabels(np.arange(0,1.0001,0.1), fontsize = 17)
cbar.ax.set_xlabel('Frequnecy of > 2.54 mm/24-h', fontsize = 20, labelpad = 15)

plt.title('SNOTEL', fontsize = 20)

plt.tight_layout()
'''
#%%

fig = plt.figure(figsize=(17,13))
cmap = matplotlib.cm.get_cmap('terrain')
ax2 = fig.add_axes([.2,.09,.6,.04])


levels_el = np.arange(0,5000,100)
#######################   PRISM event frequency  #############################################


ax = fig.add_subplot(122)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_freq, levels,  cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
props = dict(boxstyle='square', facecolor='white', alpha=1)
ax.text(80000, 100000, '(b) PRISM', fontsize = 28, bbox = props)


levels = np.arange(0.00000, 1.0001, 0.025)


ax = fig.add_subplot(121)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
xi, yi = map(freq_snotel[:,1], freq_snotel[:,0])
csAVG = map.scatter(xi,yi, c = freq_snotel[:,2], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 120)#,norm=matplotlib.colors.LogNorm() )  
csAVG2 = map.contourf(x,y,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
#cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks = np.arange(0.000001, 1.0001, 0.1))
#cbar.ax.tick_params(labelsize=12)

#cbar.ax.set_xticklabels(np.arange(0,1.0001,0.1), fontsize = 17)
#cbar.ax.set_xlabel('Frequnecy of > 2.54 mm/24-h', fontsize = 20, labelpad = 15)

ax.text(80000, 100000, '(a) SNOTEL', fontsize = 28, bbox = props)

cbar = fig.colorbar(csAVG,cax=ax2, orientation='horizontal',ticks = np.arange(0.0, 1.0001, 0.1))
cbar.ax.tick_params(labelsize=12)

cbar.ax.set_xticklabels(np.arange(0.0, 1.0001, 0.1), fontsize = 20)
cbar.ax.set_xlabel(r'Frequency of $\geq$ 2.54 mm/24 hours', fontsize = 24, labelpad = 15)

plt.tight_layout()
plt.savefig("../../../public_html/snotel_prism_climo_event_freq_separate.png")








