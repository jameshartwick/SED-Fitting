import pyfits
import numpy as np
imgname = '../data/VCC1043_k_sig.fits'
img = pyfits.getdata(imgname)
h = pyfits.getheader(imgname)
img2 = (1./img)**2
print len(img2)
hdr = h.copy()
filename = '../data/VCC1043_k_weight.fits'
pyfits.writeto(filename, img2, hdr)
pyfits.append(imgname, img2, hdr)
#pyfits.update(filename, img2, hdr, ext)
