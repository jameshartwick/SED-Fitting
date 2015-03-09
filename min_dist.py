# Spacially resolved min-chi-square-distance from pixel to model.

import run_fits as rf
import pyfits
from difflib import Differ
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from pylab import *
import os
from scipy.stats import kde
import scipy
import cPickle as pickle

x = '4.lbr'
d = np.loadtxt(x)
u_mod = d[:,25]
g_mod = d[:,26]
r_mod = d[:,27]
i_mod = d[:,28]
z_mod = d[:,29]
#y = 1942 x = 2384
ymin = 0#1766
ymax = 4999#2228
xmin = 0#2228
xmax = 4999#2542
sn_min = 5.

cmap = rf.get_cmap()

#setting up plotting

models = '../magphys/'

file_names = rf.get_file_names('list.txt')
d = rf.get_fits_data(file_names,ymin,ymax,xmin,xmax)
h = pyfits.getheader('../data/VCC1043_g.fits')
hdr = h.copy()
u = np.absolute(d[0])
g = np.absolute(d[2])
r = np.absolute(d[4])
i = np.absolute(d[6])
z = np.absolute(d[8])
#k = np.absolute(d[10])

u_sig = 1./np.sqrt(np.absolute(d[1]))
g_sig = 1./np.sqrt(np.absolute(d[3]))
r_sig = 1./np.sqrt(np.absolute(d[5]))
i_sig = 1./np.sqrt(np.absolute(d[7]))
z_sig = 1./np.sqrt(np.absolute(d[9]))
#k_sig = 1./np.sqrt(np.absolute(d[11]))

#FInding pixels with SN issues
x = np.ones(np.shape(u))
u_sig[np.where(u_sig == 0.)] = 0.0001
g_sig[np.where(g_sig == 0.)] = 0.0001
r_sig[np.where(r_sig == 0.)] = 0.0001
i_sig[np.where(i_sig == 0.)] = 0.0001
z_sig[np.where(z_sig == 0.)] = 0.0001	
#k_sig[np.where(k_sig == 0.)] = 0.0001
u_sn = np.absolute(u/u_sig)
g_sn = np.absolute(g/g_sig)
r_sn = np.absolute(r/r_sig)
i_sn = np.absolute(i/i_sig)
z_sn = np.absolute(z/z_sig)
#k_sn = np.absolute(k/k_sig)
#x[np.where(u_sn <= sn_min)] = 0.
x[np.where(g_sn <= sn_min)] = 0.
x[np.where(r_sn <= sn_min)] = 0.
x[np.where(i_sn <= sn_min)] = 0.
x[np.where(z_sn <= sn_min)] = 0.
#x[np.where(k_sn <= sn_min)] = 0.
w = np.where(x != 0.)
print len(g_mod)

band1_mod = u_mod
band2_mod = g_mod
band3_mod = i_mod#+1.85

colour1_mod = band1_mod-band2_mod
colour2_mod = band2_mod-band3_mod

band1 = rf.get_mag(u)
band2 = rf.get_mag(g)
band3 = rf.get_mag(i)

colour1 = band1 - band2
colour2 = band2 - band3

dist = np.zeros(np.shape(colour1))-1

#print np.min(np.sqrt((colour1[y,x]-colour1_mod[:])**2+(colour2[y,x]-colour2_mod[:])**2))

for x, y in zip(w[1][:],w[0][:]):
	min_dist = np.min(np.sqrt((colour1[y,x]-colour1_mod[:])**2+(colour2[y,x]-colour2_mod[:])**2))
	dist[y,x] = min_dist
	#print min_dist, x, y

#numbins = 100
##plt.hist(dist[np.where(dist >= 0)], numbins, color='black', log=True)
#pickle.dump(w, open("sn.p","wb"))
#pickle.dump(dist, open("dist.p","wb"))
#filename = 'min_dist_resamp_20.fits'
#pyfits.writeto(filename,dist,hdr)

dist = np.clip(dist,-50,.99)
dist[np.where(dist < 0)] = .1

plt.imshow(dist,cmap=cmap,origin='lower')
cb = plt.colorbar()
cb.set_label('Min Distance to a Model (Colour difference)')
#dist = dist.reshape(-1)
#figure(2)
#plt.hist(dist[np.where(dist >= 0)], log=True)
plt.show()