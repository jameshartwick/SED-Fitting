import pyfits 
import numpy as np
import pylab as pyl
import img_scale
import matplotlib.pyplot as plt
from pylab import *

cdict = {
'red': (
(0.0, 0.0, 1.0),
(0.5, 0.8, 0.8),
(0.999, 0.0, 1.0),
(1.0, 1.0, 1.0)),
'green': (
(0.0, 0.0, 0.0),
(0.5, 1.0, 1.0),
(0.999, 0.0, 0.0),
(1.0, 1.0, 1.0)),
'blue': (
(0.0, 1.0, 0.0),
(0.5, 0.8, 0.8),
(0.999, 1.0, 1.0),
(1.0, 1.0, 1.0))}
cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

#setting up plotting
fig, ax = plt.subplots()
#cmap = plt.cm.YlOrRd

#Image boundaries
ymin = 5068 #2910 4610
ymax = 5078 #7910 5610
xmin = 4959 #2790 4590
xmax = 4969 #7790 5590

c = 30

u_img = pyfits.getdata('../data/VCC1043_Z_swarp_5000.fits')
print('first image made') 
#u_img = u_img[ymin:ymax,xmin:xmax].astype(np.float)
#print('Image cropped!')


u_img = -2.5*np.log10(np.absolute(u_img))+30

u_img = u_img.clip(min=10, max=29)

plt.xticks([602,1205,1807,2410,3012,3615,4217,4820])
plt.yticks([602,1205,1807,2410,3012,3615,4217,4820])
ax.set_yticklabels(['',20,'',40,'',60,'',80])
ax.set_xticklabels(['',20,'',40,'',60,'',80])
ax.xaxis.set_tick_params(width=1, length=5)
ax.yaxis.set_tick_params(width=1, length=5)
ax.set_xlabel('P (kpc)')
ax.set_ylabel('Distance (kpc)')
print('Scaling done!')
plt.imshow(u_img,cmap=cmap,origin='lower')
cb = plt.colorbar()
cb.set_label('Magnitude')
print('Colorbar done!')
plt.show()
print('Done!')
