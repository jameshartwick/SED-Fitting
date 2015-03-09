# Colour-colour desity plots of models vs observations

import run_fits as rf
import pyfits
from difflib import Differ
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from pylab import *
import os
from scipy.stats import kde

scale = 0.187
c = 30.
x = '4.lbr'
d = np.loadtxt(x)
u_mod = d[:,25]
g_mod = d[:,26]
r_mod = d[:,27]
i_mod = d[:,28]
z_mod = d[:,29]
#k_mod = d[:,30]
#1866,2012,228,2442
ymin = 1766#1766
ymax = 2228#2228
xmin = 2228#2228
xmax = 2542#2542
sn_min = 20.

cdict = {
'red': (
(0.0, 1.0, 1.0),
(0.001, 0.0, 1.0),
(0.5, 0.8, 0.8),
(0.99, 0.0, 0.0),
(1.0, 0.0, 0.0)),
'green': (
(0.0, 1.0, 1.0),
(0.001, 0.0, 0.0),
(0.5, 1.0, 1.0),
(0.99, 0.0, 0.0),
(1.0, 0.0, 0.0)),
'blue': (
(0.0, 1.0, 1.0),
(0.001, 0.0, 0.0),
(0.5, 0.8, 0.8),
(0.99, 1.0, 1.0),
(1.0, 0.0, 0.0))}
cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

#setting up plotting

models = '../magphys/'

file_names = rf.get_file_names('list.txt')
d = rf.get_fits_data(file_names,ymin,ymax,xmin,xmax)

u = np.absolute(d[0])
g = np.absolute(d[2])
r = np.absolute(d[4])
i = np.absolute(d[6])
z = np.absolute(d[8])
#k = np.absolute(d[10])
print np.shape(u)
u_sig = np.absolute(d[1])
g_sig = np.absolute(d[3])
r_sig = np.absolute(d[5])
i_sig = np.absolute(d[7])
z_sig = np.absolute(d[9])
#k_sig = 1./np.sqrt(np.absolute(d[11]))
x = np.ones(np.shape(u))
u_sig[np.where(u_sig == 0.)] = 0.000001
g_sig[np.where(g_sig == 0.)] = 0.000001
r_sig[np.where(r_sig == 0.)] = 0.000001
i_sig[np.where(i_sig == 0.)] = 0.000001
z_sig[np.where(z_sig == 0.)] = 0.000001
#k_sig[np.where(k_sig == 0.)] = 0.000001
u_sn = np.absolute(u/u_sig)
g_sn = np.absolute(g/g_sig)
r_sn = np.absolute(r/r_sig)
i_sn = np.absolute(i/i_sig)
z_sn = np.absolute(z/z_sig)
#k_sn = np.absolute(k/k_sig)
x[np.where(u_sn <= sn_min)] = 0.
x[np.where(g_sn <= sn_min)] = 0.
x[np.where(r_sn <= sn_min)] = 0.
x[np.where(i_sn <= sn_min)] = 0.
x[np.where(z_sn <= sn_min)] = 0.
#x[np.where(k_sn <= sn_min)] = 0.
w = np.where(x != 0.)
ww = np.where(x == 0.)



band1_mod = i_mod
band2_mod = z_mod
band3_mod = g_mod
band4_mod = r_mod


band1 = rf.get_mag(i)
band2 = rf.get_mag(z)#+1.85
band3 = rf.get_mag(g)
band4 = rf.get_mag(r)#+1.85

#band3-4 is x, 1-2 is y

#b = np.where(((band3-band4)<-.5))# & ((band1-band2)>1.))

g = -2.5*np.log10(g/(scale**2))+c
#g[b] = 10.
g[ww] = 30.
#band4[b] = 100000



band1 = band1[w]
band2 = band2[w]
band3 = band3[w]
band4 = band4[w]

axis_y = (band1 - band2)
axis_x = (band3 - band4)

axis_y_mod = (band1_mod - band2_mod).reshape(-1)
axis_x_mod = (band3_mod - band4_mod).reshape(-1)

ypmin = np.min(axis_y)
ypmax = np.max(axis_y)
xpmin = np.min(axis_x)
xpmax = np.max(axis_x)

fig, axes = plt.subplots()

nbins=30
axes.set_title('(New Models) Data(Density Map), Models(Contours)')

#clip all values outside of data range.
#lims = np.where((axis_x_mod>=np.min(axis_x)) & (axis_x_mod<=np.max(axis_x)) & (axis_y_mod>=np.min(axis_y)) & (axis_y_mod<=np.max(axis_y)))
#axis_x_mod = axis_x_mod[lims]
#axis_y_mod = axis_y_mod[lims]


axis_x = np.clip(axis_x,np.min(axis_x_mod), np.max(axis_x_mod))
axis_y = np.clip(axis_y,np.min(axis_y_mod), np.max(axis_y_mod))
#axes.hexbin(axis_x_mod, axis_y_mod ,gridsize=nbins)
plt.hexbin(axis_x, axis_y, cmap=cmap, bins='log', gridsize=nbins) #cmap=plt.cm.gist_heat, 
axes.set_ylim(np.min(axis_y_mod),np.max(axis_y_mod))
axes.set_xlim(np.min(axis_x_mod), np.max(axis_x_mod))# np.min(axis_x), np.max(axis_x))
cb = plt.colorbar()
cb.set_label('log(Data points per bin)')

# Convert to 2d histogram.
Bins = 30
hist2D, xedges, yedges = np.histogram2d(axis_x_mod, axis_y_mod, bins=[Bins,Bins], normed=False)
hist2D = hist2D.transpose()
# Overplot with error contours 1,2,3 sigma.
maximum = np.max(hist2D)


[L1,L2,L3] = [0.5*maximum,0.25*maximum,0.01*maximum]  # Replace with a proper code!
# Use bin edges to restore extent.
#L1 = np.sdt(hist2D)

extent = [xedges[0],xedges[-1], yedges[0],yedges[-1]]
cs = plt.contour(hist2D, extent=extent, levels=[L1,L2,L3],colors=['black','black','black'], linewidths=1)
# use dictionary in order to assign your own labels to the contours.
fmtdict = {L1:r'$50\%$',L2:r'$25\%$',L3:r'$1\%$'}
plt.clabel(cs, fmt=fmtdict, inline=True, fontsize=20)

axes.set_ylabel('Colour (i-z)')
axes.set_xlabel('Colour (g-r)')
axes.set_xlim(0,1.2)
#figure(2)
#plt.imshow(g,cmap=cmap,origin='lower')
#cb = plt.colorbar()
#cb.set_label('')
plt.show()