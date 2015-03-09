# This Script is used to overplot our models and observations in colour-colour space.
# It must be used to ensure that the model set fully covers our parameter space.

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
#1866,2012,228,2442
ymin = 1766#1766
ymax = 2228#2228
xmin = 2228#2228
xmax = 2542#2542
sn_min = 20.


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

u = np.log10(u[w])
g = np.log10(g[w])
r = np.log10(r[w])
i = np.log10(i[w])
z = np.log10(z[w])

axis_y = [3740., 4870., 6360., 7700.,8900.]
#u = np.reshape(len(u),2)
#u[1,:] = axis_y[0]

#u = np.append(u,np.zeros([len(u)]))
bands = np.resize(u,(5,len(u)))
bands[1] = g
bands[2] = r
bands[3] = i
bands[4] = z
bands = rf.get_mag(bands)
norm = 0#bands[4,0]

bands =  (bands+(norm-bands[4]))
u = np.resize(bands[0],(2,len(u)))
u[1,:] = 3740.
g = np.resize(bands[1],(2,len(g)))
g[1,:] = 4870.
r = np.resize(bands[2],(2,len(r)))
r[1,:] = 6360.
i = np.resize(bands[3],(2,len(i)))
i[1,:] = 7700.
z = np.resize(bands[4],(2,len(z)))
z[1,:] = 8900.
print np.max(bands[4]), np.min(bands[4])
axis_y = np.concatenate((u[0,:],g[0,:],r[0,:],i[0,:],z[0,:]))
axis_x = np.concatenate((u[1,:],g[1,:],r[1,:],i[1,:],z[1,:]))
fig, axes = plt.subplots()

nbins=20
axes.set_title('Data(Density Map), Models(Contours)')

axes.hexbin(axis_x, axis_y ,gridsize=nbins)
plt.hexbin(axis_x, axis_y, cmap=plt.cm.YlOrRd, bins='log', gridsize=nbins) #cmap=plt.cm.gist_heat, 
axes.set_ylim(np.min(axis_y),np.max(axis_y))
axes.set_xlim(np.min(axis_x), np.max(axis_x))# np.min(axis_x), np.max(axis_x))
cb = plt.colorbar()
cb.set_label('log(Data points per bin)')

axes.set_ylabel('AB Magnitude Difference')
axes.set_xlabel(r'$\lambda$')
plt.gca().invert_yaxis()
axes.set_xlim(3500,9500)
plt.show()
lambda_x = [3740., 4870., 6360., 7700.,8900.]


