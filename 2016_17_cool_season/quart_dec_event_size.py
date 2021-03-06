
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



#%%
#######   Calculate frequency of precip days    ###########



                
                
med_snotel = zeros((798,3))
quart_snotel = zeros((798,3))
dec_snotel = zeros((798,3))
med_snotel[:,0:2] = inhouse_data[:,1:3]
quart_snotel[:,0:2] = inhouse_data[:,1:3]
dec_snotel[:,0:2] = inhouse_data[:,1:3]

temp = []

x = 0
for j in range(798):
        med_snotel[j,2] = np.median(inhouse_data[j,3:][inhouse_data[j,3:] >= 2.54])
        
    
        temp = inhouse_data[j,3:]
        temp = np.delete(temp, np.where(temp < 2.54))
        
        if quart_snotel[j,0] > 0: # Make sure its a station
            quart_snotel[j,2] = np.percentile(temp, 75)
            dec_snotel[j,2] = np.percentile(temp, 90)

            
#%%    
                





###############################################################################
##############   Read in prism precip   #############################
###############################################################################
 

precip_med = zeros((183, 621,1405))
x = 0
for i in range(0,num_days): 


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
            precip_med[x,i,j] = precip[i,j]
    x = x + 1
    print x          
#%%
prism_med = zeros((621,1405))
prism_quart = zeros((621,1405))
prism_dec = zeros((621,1405))

prism_temp = []

for i in range(621):
    for j in range(1405):    
        prism_med[i,j] = np.median(precip_med[:,i,j][precip_med[:,i,j] >= 2.54])
        
        prism_temp = precip_med[:,i,j]
        prism_temp = np.delete(prism_temp, np.where(prism_temp < 2.54))
        print i
        if len(prism_temp) > 1: #Make sure entire array wasnt deleted
            prism_quart[i,j] = np.percentile(prism_temp, 75)
            prism_dec[i,j] = np.percentile(prism_temp, 90)

        
        

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
fig = plt.figure(figsize=(17,22))

ax2 = fig.add_axes([.2,.05,.6,.03])

#levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]
levels = [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7,  8, 9,10, 11, 12, 13, 14.5, 16, 17.5, 19,21, 24, 30,36, 42, 50, 60, 70, 80, 100,140]
levels_ticks = [3.5,  5,  6.5, 9, 12, 16, 21, 36,  60, 100]
levels_el = np.arange(0,5000,100)
#######################   PRISM event frequency  #############################################



ax = fig.add_subplot(221)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
xi, yi = map(quart_snotel[:,1], quart_snotel[:,0])
csAVG = map.scatter(xi,yi, c = quart_snotel[:,2], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 120)#,norm=matplotlib.colors.LogNorm() )  
csAVG2 = map.contourf(x,y,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
plt.title('SNOTEL', fontsize = 32)
plt.text(50000,50000,'Upper Quartile',fontsize = 27)


ax = fig.add_subplot(222)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,prism_quart, levels,  cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
plt.title('PRISM', fontsize = 32)
plt.text(50000,50000,'Upper Quartile',fontsize = 27)




ax = fig.add_subplot(223)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
xi, yi = map(dec_snotel[:,1], dec_snotel[:,0])
csAVG = map.scatter(xi,yi, c = dec_snotel[:,2], cmap=cmap, marker='o', norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), s = 120)#,norm=matplotlib.colors.LogNorm() )  
csAVG2 = map.contourf(x,y,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
plt.title('SNOTEL', fontsize = 32)
plt.text(50000,50000,'Upper Decile',fontsize = 27)


ax = fig.add_subplot(224)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,prism_dec, levels,  cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
plt.title('PRISM', fontsize = 32)
plt.text(50000,50000,'Upper Decile',fontsize = 27)



cbar = fig.colorbar(csAVG,cax=ax2, orientation='horizontal',ticks = levels_ticks)
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_xticklabels(levels_ticks, fontsize = 22)
cbar.ax.set_xlabel('Event Size (mm)', fontsize = 26, labelpad = 15)

plt.tight_layout(pad=2, w_pad=2, h_pad=-10)
plt.savefig("../../../public_html/snotel_prism_event_size.png")








