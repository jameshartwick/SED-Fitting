import pyfits 
import numpy as np
import pylab as pyl
import img_scale
import matplotlib.pyplot as plt
from pylab import *

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

min_sn = 5.
c = 30
scale = 0.187
#Grabbing image data
i = pyfits.getdata('../data/VCC1043_k.fits')
i_sig = 1./np.sqrt(pyfits.getdata('../data/VCC1043_k_weight.fits'))

w = np.where(i_sig == 0.)
i_sig[w] = 0.000001
print('where done!')
i_sn = i/i_sig

w = np.where(i_sn <= min_sn)
i[w] = np.max(i)

mu = -2.5*np.log10(i/(scale*scale))+c
print('first image made') 
#u_img = u_img[ymin:ymax,xmin:xmax].astype(np.float)
#print('Image cropped!')
plt.xticks([602,1205,1807,2410,3012,3615,4217,4820])
plt.yticks([602,1205,1807,2410,3012,3615,4217,4820])
ax.set_yticklabels(['',20,'',40,'',60,'',80])
ax.set_xticklabels(['',20,'',40,'',60,'',80])
ax.xaxis.set_tick_params(width=1, length=5, labelsize=8)
ax.yaxis.set_tick_params(width=1, length=5, labelsize=8)
ax.set_xlabel(r'$Physical\ size\ (kpc)$')
ax.set_ylabel(r'$Physical\ size\ (kpc)$')
fig.suptitle(r'$Surface\ Brightness\ Map\ (k)$')

print('Scaling done!')
plt.imshow(mu,cmap=cmap,origin='lower')
cb = plt.colorbar()
cb.set_label(r'$Surface\ Brightness\ (Mag/ArcSec^2)$')
print('Colorbar done!')
plt.show()
print('Done!')
