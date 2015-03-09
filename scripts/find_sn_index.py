import pyfits 
import numpy as np
import pylab as pyl
import img_scale
import matplotlib.pyplot as plt
from pylab import *

fig, ax = plt.subplots()
cdict = {'red': (
(0.0, 1.0, 1.0),
(0.001, 0.0, 1.0),
(0.5, 0.8, 0.8),
(1.0, 0.0, 1.0)),
'green': (
(0.0, 1.0, 1.0),
(0.001, 0.0, 0.0),
(0.5, 1.0, 1.0),
(1.0, 0.0, 0.0)),
'blue': (
(0.0, 1.0, 1.0),
(0.001, 0.0, 0.0),
(0.5, 0.8, 0.8),
(1.0, 1.0, 0.0))}
cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

#setting up plotting
fig, ax = plt.subplots()
#cmap = plt.cm.YlOrRd

#Image boundaries
ymin = 5068 #2910 4610
ymax = 5078 #7910 5610
xmin = 4959 #2790 4590
xmax = 4969 #7790 5590

c = 30.

#Grabbing image data
z = pyfits.getdata('../data/VCC1043_Z_swarp_5000.fits')
z_sig = pyfits.getdata('../data/VCC1043_Z_sigma_swarp_5000.fits')
i = pyfits.getdata('../data/VCC1043_I_swarp_5000.fits')
i_sig = pyfits.getdata('../data/VCC1043_I_sigma_swarp_5000.fits')
g = pyfits.getdata('../data/VCC1043_G_swarp_5000.fits')
g_sig = pyfits.getdata('../data/VCC1043_G_sigma_swarp_5000.fits')
u = pyfits.getdata('../data/VCC1043_U_swarp_5000.fits')
u_sig = pyfits.getdata('../data/VCC1043_U_sigma_swarp_5000.fits')

w = np.where(u_sig == 0.)
u_sig[w] = 0.0001
w = np.where(g_sig == 0.)
g_sig[w] = 0.0001
w = np.where(i_sig == 0.)
i_sig[w] = 0.0001
w = np.where(z_sig == 0.)
z_sig[w] = 0.0001

x = np.ones(np.shape(u))

u_img = np.absolute(u/u_sig)
g_img = np.absolute(g/g_sig)
i_img = np.absolute(i/i_sig)
z_img = np.absolute(z/z_sig)
print np.max(u_img)
w = np.where(u_img <= 5.)
x[w] = 0.
w = np.where(g_img <= 5.)
x[w] = 0.
w = np.where(i_img <= 5.)
x[w] = 0.
w = np.where(z_img <= 5.)
x[w] = 0.

w = np.where(x != 0.)																																																																																																																																																																															
print x[w]

print('Done!')

#print('Scaling done!')
##plt.imshow(x,cmap=cmap,origin='lower')
#cb = plt.colorbar()
#cb.set_label('log(S/N)')
#print('Colorbar done!')
#plt.show()
#print('Done!')