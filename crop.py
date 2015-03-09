import pyfits
import numpy as np
imgname = '../data/new_images/gwyn/NGVS-1+1.K.MH001.v6.fits'
img = pyfits.getdata(imgname)
h = pyfits.getheader(imgname)
img2 = img[9999:14999,10299:15299]
img2 = 1./np.sqrt(np.absolute(img2))
print len(img2)
hdr = h.copy()
filename = '../data/new_images/cropped/VCC1043_k_sig.fits'
pyfits.writeto(filename, img2, hdr)
pyfits.append(imgname, img2, hdr)
#pyfits.update(filename, img2, hdr, ext)
