import pyfits 
import numpy as np
import pylab as pyl
import img_scale
import matplotlib.pyplot as plt

def get_sb(n):
    n = -2.5*np.log10(np.absolute(n))+30
    n = n.clip(min=15, max=29)
    return n

def get_m_error(s, n):
    u = get_sb(s)-((-2.5*np.log10(s)+30.)-(2.5*np.log10(1.+n/s)))
    l = ((-2.5*np.log10(s)+30.)-(2.5*np.log10(1.-n/s)))-get_sb(s)
    return [u,l]

#setting up plotting
fig, ax = plt.subplots()
cmap = plt.cm.YlOrRd
#Image boundaries
ymin = 5000 #2910 4610
ymax = 5001 #7910 5610
xmin = 5000 #2790 4590
xmax = 5001 #7790 5590


u_img = pyfits.getdata('../data/VCC1043_U_swarp_fullres.fits')
g_img = pyfits.getdata('../data/VCC1043_G_swarp_fullres.fits')
i_img = pyfits.getdata('../data/VCC1043_I_swarp_fullres.fits')
z_img = pyfits.getdata('../data/VCC1043_Z_swarp_fullres.fits')

u_sig = pyfits.getdata('../data/VCC1043_U_sigma_swarp_fullres.fits')
g_sig = pyfits.getdata('../data/VCC1043_G_sigma_swarp_fullres.fits')
i_sig = pyfits.getdata('../data/VCC1043_I_sigma_swarp_fullres.fits')
z_sig = pyfits.getdata('../data/VCC1043_Z_sigma_swarp_fullres.fits')


print('first images made') 
u_img = u_img[ymin:ymax,xmin:xmax].astype(np.float)
g_img = g_img[ymin:ymax,xmin:xmax].astype(np.float)
i_img = i_img[ymin:ymax,xmin:xmax].astype(np.float)
z_img = z_img[ymin:ymax,xmin:xmax].astype(np.float)

u_sig = u_sig[ymin:ymax,xmin:xmax].astype(np.float)
g_sig = g_sig[ymin:ymax,xmin:xmax].astype(np.float)
i_sig = i_sig[ymin:ymax,xmin:xmax].astype(np.float)
z_sig = z_sig[ymin:ymax,xmin:xmax].astype(np.float)

print('Images cropped!')
u_img = get_sb(u_img)
g_img = get_sb(g_img)
i_img = get_sb(i_img)
z_img = get_sb(z_img)

u_e = get_m_error(u_img, u_sig)
g_e = get_m_error(g_img, g_sig)
i_e = get_m_error(i_img, i_sig)
z_e = get_m_error(z_img, z_sig)

print("u* upper/lower errors:")
print u_e[0]
print u_e[1]

#bands = [u_img, g_img, i_img, z_img]

bands = [3740,4870,7700,8900]
sed = [u_img[0],g_img[0],i_img[0],z_img[0]]
for i in range(0,3):
    ax.errorbar(bands[i], sed[i], yerr=u_e[0][i])
    
    
#for x in range(0,3):
#    plt.plot([u_img[x], g_img[x], i_img[x], z_img[x]])


#plt.imshow(u_img,cmap=cmap,origin='lower')
#cb = plt.colorbar()
#cb.set_label('Surface brightness')
#print('Colorbar done!')
plt.show()
print('Done!')
