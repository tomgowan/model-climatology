import pygrib, os, sys
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta




import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
cast



Date = input("Enter start date (YYYYMMDD): ")
latloc = input('Enter the latitude of your location (degrees): ')
lonloc = input('Enter the longitude of your location (degrees): ')

#DateTotal = time.strftime("%x")
Date2= str(Date)

t=time.strptime(Date2,'%Y%m%d')

newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(1)
Date3 = newdate.strftime('%Y%m%d')


t=time.strptime(Date2,'%Y%m%d')
newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(2)
Date4 = newdate.strftime('%Y%m%d')

newdate=date(t.tm_year,t.tm_mon,t.tm_mday)-timedelta(1)
Date5 = newdate.strftime('%Y%m%d')
Date_minus1 = int(Date5)

tempplot = zeros((49,10))
precipplot = zeros((49,10))
preciponehour = zeros((49,10))
preciponehourflip = zeros((10,49))
nearest = zeros((785,650))
nearest_hrrr = zeros((1033,842))
onehourmed = zeros((49))
hrrr_precip = zeros((16,6))

N =50
ind = np.arange(N)  # the x locations for the groups
width = 0.35

x = range(0,49)





NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/ncarens/ncar_3km_westus_2016010500_mem9_f047.nc'
fhcoord = Dataset(NCARens_file, mode='r')

lon = fhcoord.variables['gridlon_0'][:]
lat = fhcoord.variables['gridlat_0'][:]

for i in range(0,785):
    for j in range(0,650):
        nearest[i,j] = abs(lon[i,j]-lonloc) + abs(lat[i,j]-latloc)
        
location =  np.where(nearest == nearest.min())
        
row = location[0]
col = location[1]


fhcoord.close()




####### Read in NCAR_Ens file ###################################################



for i in range(1,11):
    for j in range(0,49):
        NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/ncarens/ncar_3km_westus_%s' % Date + '00_mem%s' % i + '_f0%02d.nc' %  j
        fh = Dataset(NCARens_file, mode='r')

        if j == 1:
            precipI = fh.variables['APCP_P8_L1_GLC0_acc'][row, col]
            precip = precipI*0.0393689

        else:
            precipI = fh.variables['APCP_P8_L1_GLC0_acc1h'][row, col]
            precip = precipI*0.0393689


        

        precipplot[j,i-1] = precipplot[j-1,i-1]+precip
        preciponehour[j-1,i-1] = precip
    
        
fh.close()




####### Read in HRRR file ###################################################
grlatlon = pygrib.open('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/2016012400F15hrrr.grib2')
for g in grlatlon:
    print g

glatlon = grlatlon.select(name='Total Precipitation')[1]
hrrr_precip_test = glatlon.values
hrrr_lat,hrrr_lon = glatlon.latlons()

for i in range(0,1033):
    for j in range(0,842):
        nearest_hrrr[i,j] = abs(hrrr_lon[i,j]-lonloc) + abs(hrrr_lat[i,j]-latloc)
        
location_hrrr =  np.where(nearest_hrrr == nearest_hrrr.min())
        
row_hrrr = location_hrrr[0]
col_hrrr = location_hrrr[1]





for j in range(0,16):
    HRRR_File = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%s' % Date + '00F%02d' % j + 'hrrr.grib2'
    grbs = pygrib.open(HRRR_File)
    grb = grbs.select(name='Total Precipitation')[1]
    hrrr_precip[j,1] = grb.values[row_hrrr,col_hrrr]*0.0393689+hrrr_precip[j-1,1]

for j in range(0,16):
    HRRR_File2 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%s' % Date5 + '23F%02d' % j + 'hrrr.grib2'
    grbs2 = pygrib.open(HRRR_File2)
    grb2 = grbs2.select(name='Total Precipitation')[1]
    hrrr_precip[j,2] = grb2.values[row_hrrr,col_hrrr]*0.0393689+hrrr_precip[j-1,2]     
        
for j in range(0,16):
    HRRR_File3 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%s' % Date5 + '22F%02d' % j + 'hrrr.grib2'
    grbs3 = pygrib.open(HRRR_File3)
    grb3 = grbs3.select(name='Total Precipitation')[1]
    hrrr_precip[j,3] = grb3.values[row_hrrr,col_hrrr]*0.0393689+hrrr_precip[j-1,3]       
        
for j in range(0,16):
    HRRR_File4 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%s' % Date5 + '21F%02d' % j + 'hrrr.grib2'
    grbs4 = pygrib.open(HRRR_File4)
    grb4 = grbs4.select(name='Total Precipitation')[1]
    hrrr_precip[j,4] = grb4.values[row_hrrr,col_hrrr]*0.0393689+hrrr_precip[j-1,4]       
        
for j in range(0,16):
    HRRR_File5 = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/hrrr/%s' % Date5 + '20F%02d' % j + 'hrrr.grib2'
    grbs5 = pygrib.open(HRRR_File5)
    grb5 = grbs5.select(name='Total Precipitation')[1]
    hrrr_precip[j,5] = grb5.values[row_hrrr,col_hrrr]*0.0393689+hrrr_precip[j-1,5]       
        
        
        
        
        
        
        
        
        
        
        
preciponehourflip = np.swapaxes(preciponehour,0,1)




fig2 = plt.figure(figsize=(7,8))
fig2.subplots_adjust(hspace=0.5)
ax2 = fig2.add_subplot(211)
y = np.arange(0,49,4)
datess = [str(Date2[6:8]) + '/00Z', str(Date2[6:8]) + '/04Z', str(Date2[6:8]) + '/08Z', str(Date2[6:8]) + '/12Z', 
         str(Date2[6:8]) + '/16Z', str(Date2[6:8]) + '/20Z', str(Date3[6:8]) + '/00Z', str(Date3[6:8]) + '/04Z', 
         str(Date3[6:8]) + '/08Z', str(Date3[6:8]) + '/12Z', str(Date3[6:8]) + '/16Z', str(Date3[6:8]) + '/20Z', 
         str(Date4[6:8]) + '/00Z']
plt.xticks(y, datess)
plt.xticks(rotation=60)
plt.grid(True)
ax2.plot(precipplot[0:11, 0:10], 'b')
ax2.plot(hrrr_precip[0:11,1], 'r')
ax2.plot(hrrr_precip[1:12,2], 'r')
ax2.plot(hrrr_precip[2:13,3], 'r')
ax2.plot(hrrr_precip[3:14,4], 'r')
ax2.plot(hrrr_precip[4:15,5], 'r')
plt.title('Accumulated Precipitation (lat = ' + str(latloc) + ', lon = ' + str(lonloc) + ')\n NCAR Ensemble (10 members)')
plt.xlabel('Day/Hour')
plt.ylabel('Precipitation (Inches)')



ax2 = fig2.add_subplot(212)
bp = ax2.boxplot(preciponehourflip, patch_artist=True, whis=1000)
plt.setp(bp['whiskers'], color='b',  linestyle='-' )
y = np.arange(0,49,4)
datess = [str(Date2[6:8]) + '/00Z', str(Date2[6:8]) + '/04Z', str(Date2[6:8]) + '/08Z', str(Date2[6:8]) + '/12Z', 
         str(Date2[6:8]) + '/16Z', str(Date2[6:8]) + '/20Z', str(Date3[6:8]) + '/00Z', str(Date3[6:8]) + '/04Z', 
         str(Date3[6:8]) + '/08Z', str(Date3[6:8]) + '/12Z', str(Date3[6:8]) + '/16Z', str(Date3[6:8]) + '/20Z', 
         str(Date4[6:8]) + '/00Z']
plt.xticks(y, datess)
plt.xticks(rotation=60)
plt.xlim([0,24])
plt.grid(True)
plt.title('Hourly Precipitation')
plt.xlabel('Day/Hour')
plt.ylabel('Precipitation (Inches)')
plt.savefig("MeteogramPrecip.pdf")
plt.show()
