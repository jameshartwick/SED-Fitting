import pyfits 
import numpy as np
import pylab as pyl
import img_scale
import matplotlib.pyplot as plt
from pylab import *
import run_fits as rf

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
ymin = 0 #2910 4610
ymax = 4999 #7910 5610
xmin = 0 #2790 4590
xmax = 4999 #7790 5590
sn_min = 5.

file_names = rf.get_file_names('list.txt')
d = rf.get_fits_data(file_names)

u = np.absolute(d[0][ymin:ymax,xmin:xmax])
g = np.absolute(d[2][ymin:ymax,xmin:xmax])
r = np.absolute(d[4][ymin:ymax,xmin:xmax])
i = np.absolute(d[6][ymin:ymax,xmin:xmax])
z = np.absolute(d[8][ymin:ymax,xmin:xmax])
k = np.absolute(d[10][ymin:ymax,xmin:xmax])

u_sig = np.absolute(d[1][ymin:ymax,xmin:xmax])
g_sig = np.absolute(d[3][ymin:ymax,xmin:xmax])
r_sig = np.absolute(d[5][ymin:ymax,xmin:xmax])
i_sig = np.absolute(d[7][ymin:ymax,xmin:xmax])
z_sig = np.absolute(d[9][ymin:ymax,xmin:xmax])
k_sig = 1./np.sqrt(np.absolute(d[11][ymin:ymax,xmin:xmax]))
x = np.ones(np.shape(u))
u_sig[np.where(u_sig == 0.)] = 0.000001
g_sig[np.where(g_sig == 0.)] = 0.000001
r_sig[np.where(r_sig == 0.)] = 0.000001
i_sig[np.where(i_sig == 0.)] = 0.000001
z_sig[np.where(z_sig == 0.)] = 0.000001
k_sig[np.where(k_sig == 0.)] = 0.000001
u_sn = np.absolute(u/u_sig)
g_sn = np.absolute(g/g_sig)
r_sn = np.absolute(r/r_sig)
i_sn = np.absolute(i/i_sig)
z_sn = np.absolute(z/z_sig)
k_sn = np.absolute(k/k_sig)
x[np.where(u_sn <= sn_min)] = 0.
x[np.where(g_sn <= sn_min)] = 0.
x[np.where(r_sn <= sn_min)] = 0.
x[np.where(i_sn <= sn_min)] = 0.
x[np.where(z_sn <= sn_min)] = 0.
x[np.where(k_sn <= sn_min)] = 0.
w = np.where(x == 0.)



c = 30.

scale = 0.187

g[w] = np.max(g)

mu = -2.5*np.log10(g/(scale*scale))+c
print('first image made') 
#u_img = u_img[ymin:ymax,xmin:xmax].astype(np.float)
#print('Image cropped!')

plt.imshow(mu,cmap=cmap,origin='lower')
cb = plt.colorbar()

print('Colorbar done!')
plt.show()
print('Done!')
