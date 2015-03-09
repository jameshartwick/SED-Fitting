import pyfits 
import numpy as np
import pylab as pyl
import img_scale
import matplotlib.pyplot as plt

def get_m_error(s, n):
    u = get_sb(s)-((-2.5*np.log10(s)+30.)-(2.5*np.log10(1.+n/s)))
    l = ((-2.5*np.log10(s)+30.)-(2.5*np.log10(1.-n/s)))-get_sb(s)
    return [u,l]

#setting up plotting
fig, ax = plt.subplots()
cmap = plt.cm.YlOrRd

u_img = pyfits.getdata('../data/VCC1043_U_swarp_fullres.fits')
#g_img = pyfits.getdata('../data/VCC1043_G_swarp_fullres.fits')
#i_img = pyfits.getdata('../data/VCC1043_I_swarp_fullres.fits')
#z_img = pyfits.getdata('../data/VCC1043_Z_swarp_fullres.fits')

u_sig = pyfits.getdata('../data/VCC1043_U_sigma_swarp_fullres.fits')
#g_sig = pyfits.getdata('../data/VCC1043_G_sigma_swarp_fullres.fits')
#i_sig = pyfits.getdata('../data/VCC1043_I_sigma_swarp_fullres.fits')
#z_sig = pyfits.getdata('../data/VCC1043_Z_sigma_swarp_fullres.fits')

ymin = 2910 #2910 4610
ymax = 7910 #7910 5610
xmin = 2790 #2790 4590
xmax = 7790 #7790 5590


c = 30
u_img = u_img[ymin:ymax,xmin:xmax].astype(np.float)
u_sig = u_sig[ymin:ymax,xmin:xmax].astype(np.float)

u_sig[np.where(u_sig == 0)] = 1
sn = u_img/u_sig
sn[np.where(u_img <= 0)] = 1
sn[np.where(u_sig <= 0)] = 1

sn = np.log10(sn)

plt.imshow(sn,cmap=cmap,origin='lower')
cb = plt.colorbar()
cb.set_label('log(S/N')
print('Colorbar done!')
plt.show()
print('Done!')
