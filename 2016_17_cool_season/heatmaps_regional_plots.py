import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import pygrib, os, sys, glob
from netCDF4 import Dataset
from numpy import *
import numpy as np
from scipy import stats
from pylab import *
import time
from datetime import date, timedelta
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from matplotlib.mlab import bivariate_normal
from matplotlib import colors, ticker, cm
from datetime import date, timedelta
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colors, ticker, cm
import matplotlib.lines as mlines

array_pacific = np.load('heatmaps_pacific.npy')
array_interior = np.load('heatmaps_interior.npy')










#%%
med_pac = zeros((6,18,2))
med_int = zeros((6,18,2))

for mod in range(6):
    for k in range(18):
        test_med = []
        for i in range(18):
            for j in range(int(10000*array_pacific[mod,k,i])):
                test_med.append(i)
        med_pac[mod,k,0] = np.median(test_med)

for mod in range(6):
    for k in range(18):
        test_med = []
        for i in range(18):
            for j in range(int(10000*array_pacific[mod,i,k])):
                test_med.append(i)
        med_pac[mod,k,1] = np.median(test_med)
        

for mod in range(6):
    for k in range(18):
        test_med = []
        for i in range(18):
            for j in range(int(10000*array_interior[mod,k,i])):
                test_med.append(i)
        med_int[mod,k,0] = np.median(test_med)

for mod in range(6):
    for k in range(18):
        test_med = []
        for i in range(18):
            for j in range(int(10000*array_interior[mod,i,k])):
                test_med.append(i)
        med_int[mod,k,1] = np.median(test_med)



#%%
###############################################################################
#########################  PLots   ############################################
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


#colors = [
# (235, 246, 255),
# (214, 226, 255),
# (181, 201, 255),
# (142, 178, 255),
# (127, 150, 255),
# (114, 133, 248),
# (99, 112, 248),
# (0, 158,  30),
# (60, 188,  61),
# (179, 209, 110),
# (185, 249, 110),
# (255, 249,  19),
# (255, 163,   9),
# (229,   0,   0),
# (189,   0,   0),
# (129,   0,   0),
# (0,   0,   0)
# ]
colors = [
 (215, 227, 238),
 (181, 202, 255),
 (143, 179, 255),
 (127, 151, 255),
 (171, 207,  99),
 (232, 245, 158),
 (255, 250,  20),
 (255, 209,  33),
 (255, 163,  10),
 (255,  76,   0),
 ]
#colors = [
# (255, 255, 255),
# (237, 250, 194),
# (205, 255, 205),
# (153, 240, 178),
#  (83, 189, 159),
#  (50, 166, 150),
#  (50, 150, 180),
#   (5, 112, 176),
#   (5,  80, 140),
#  (10,  31, 15),
#  (44,   2,  70),
# (106,  44,  90),
#]
cmap = make_cmap(colors, bit=True)

###############################################################################
#####  Code found online to discritize any color map ##########################
###############################################################################

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


xlab = np.arange(1.27, 100, 2.54)
xlab = np.round(xlab, decimals = 1)

model = ['(a) NCAR ENS CTL', '(b) HRRR', '(g) NCAR ENS CTL', '(h) HRRR',
         '(c) NAM-3km', '(d) GFS', '(i) NAM-3km', '(j) GFS',
         '(e) SREF ARW CTL', '(f) SREF NMMB CTL','(k) SREF ARW CTL', '(l) SREF NMMB CTL'] 
labels = np.arange(2.54,60,2.54)
x = np.arange(0,20,1)
y = np.arange(0,20,1)


fig1=plt.figure(num=None, figsize=(18,14), dpi=500, facecolor='w', edgecolor='k')
plt.plot([0.5, 0.5],[0.41,0.69],color='k', linestyle='-', linewidth=4)
plt.axis('off')
ax2 = fig1.add_axes([0.12,-0.02, 0.8, 0.27], visible = False)
ax3 = fig1.add_axes([0.12,0.8, 0.8, 0.24])

#ax2 = fig1.add_axes([0,0.7,1,7])
plt.axis('off')
ax3.axis('off')

#array = [ncar_array, gfs_array, hrrr_array, nam3km_array]

for i in range(12):
    if i == 0:
        g = array_pacific[0]
    if i == 1:
        g = array_pacific[2]
    if i == 2:
        g = array_interior[0]
    if i == 3:
        g = array_interior[2]
    if i == 4:
        g = array_pacific[3]
    if i == 5:
        g = array_pacific[1]
    if i == 6:
        g = array_interior[3]
    if i == 7:
        g = array_interior[1]
    if i == 8:
        g = array_pacific[4]
    if i == 9:
        g = array_pacific[5]
    if i == 10:
        g = array_interior[4]
    if i == 11:
        g = array_interior[5]

    print i
    plot = i+1
    ax1 = fig1.add_subplot(3, 4, plot)
    ax1.set_xticks(x)
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    #plt.title(model[i], fontsize = 20)
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax1.text(0.7, 16.4, model[i], fontsize = 18, bbox = props)

    if i == 8 or i == 9 or i == 10 or i == 11:
        ax1.set_xlabel('Forecast Event Size (mm)', fontsize  = 17.5, labelpad = 12)
    if i == 0 or i == 4 or i == 8:
        ax1.set_ylabel('Observed Event Size (mm)', fontsize  = 17.5, labelpad = 12)



    if i == 0 or i == 4:
        ax1.set_yticklabels(xlab, fontsize = 16)
        for label in ax1.get_yticklabels()[1::2]:
            label.set_visible(False)
    else:
        ax1.set_yticklabels([])
        
    if i == 9 or i == 10 or i == 11:
        ax1.set_xticklabels(xlab, fontsize  = 16, rotation = 45)
        for label in ax1.get_xticklabels()[1::2]:
            label.set_visible(False)
    else:
        ax1.set_xticklabels([])


    if i == 8:
        ax1.set_xticklabels(xlab, fontsize  = 16, rotation = 45)
        for label in ax1.get_xticklabels()[1::2]:
            label.set_visible(False)
        ax1.set_yticklabels(xlab, fontsize = 16)
        for label in ax1.get_yticklabels()[1::2]:
            label.set_visible(False)
            
    cmap=discrete_cmap(13, cmap)
    cmap.set_bad(color='#%02x%02x%02x' % (215,227,238))
    heatmap = ax1.pcolor(g, vmin=0.00005, vmax=.1, cmap=cmap, norm=matplotlib.colors.LogNorm())

    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]

    # now plot both limits against eachother
    ax1.plot(lims, lims, 'k', alpha=1, zorder=1)
    
    
    med = zeros((18,2))
    for k in range(18):
            test_med = []
            for i in range(18):
                if 10000*g[i,k] > 20:
                    for j in range(int(10000*g[k,i])):
                        test_med.append(i)
                med[k,0] = np.median(test_med)


    for k in range(18):
            test_med = []
            for i in range(18):
                if 10000*g[i,k] > 20:
                    for j in range(int(10000*g[i,k])):
                        test_med.append(i)
                med[k,1] = np.median(test_med)
    
    ax1.scatter(np.arange(0.5,18,1), med[:,1]+0.5, alpha=1, zorder=1, color = 'blue')
    ax1.scatter(med[:,0]+0.5,np.arange(0.5,18,1), alpha=1, zorder=1, color = 'red')        
        
    
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    
    
    
    ### Create arrays for linear best fit
    x_val = []
    y_val = []
    for xx in range(18):
        for num in range(int(sum(g[:,xx])*100000)):
            x_val.append(xx)
            
    for yy in range(18):
        for num in range(int(sum(g[yy,:])*100000)):
            y_val.append(yy)
            
    x_val = np.asarray(x_val[:97000])
    y_val = np.asarray(y_val[:97000])
    
    # Generated linear fit
#    slope, intercept, r_value, p_value, std_err = stats.linregress(x_val,y_val)
#    line = slope*x_val+intercept
#    
#    ax1.plot(x_val, line)

    z = np.poly1d(polyfit(x_val, y_val, 1))
    line = ax1.plot(x_val, z(x_val), 'magenta', linewidth = 2, linestyle = '--')
    #plt.setp(line,'color', 'r-', 'linewidth', 1.5)
    
    
    
ax3.text(.06,0.77, 'Pacific Ranges', fontsize = 40)
ax3.text(.65,0.77, 'Interior Ranges', fontsize = 40)

formatter = LogFormatter(10)#, labelOnlyBase=False) 
#cbar = plt.colorbar(heatmap, ax = ax2, ticks=[1,1.77,3.16,10**.75, 10, 10**1.25, 10**1.5, 10**1.75, 10**2, 10**2.25, 10**2.5, 10**2.75, 10**3],format = formatter)
ticks = np.logspace(1, 0.232503, num = 14, base = 0.00005)
cbar = plt.colorbar(heatmap, ax = ax2,ticks = ticks,format = formatter, orientation='horizontal')
cbar.ax.set_xticklabels([' ', '<0.009    ', '0.02', '0.03','0.06','0.1','0.2','0.4','0.8','1.5','3','5','>10'], fontsize = 21)
plt.tight_layout()
cbar.ax.set_xlabel('\nFrequency of Event (%)', fontsize  = 22, labelpad = -6)
plt.savefig("../../../public_html/ms_thesis_plots/heatmaps_2016_17_both_regions_new.pdf", bbox_inches='tight')
plt.close(fig1)







