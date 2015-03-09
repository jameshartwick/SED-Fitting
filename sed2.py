# Used for exploratory SED analysis. This will later be 
# implemented into our stellar parameter density plots
# Arrow keys may be used to change current pixel
import matplotlib.pyplot as plt
import numpy as np
import pyfits 
import pylab as pyl
import img_scale
import random

global x_pix

def main():
 #   pyl.ion()
    x_pix = 5001
    x = 5000
    y = 5000
    img_full = load_fits()
    img = get_pixel(img_full, x, y)
    sb_e = img
    img = get_sb_e(img,sb_e)

    y_errors = [sb_e[1], sb_e[3], sb_e[5], sb_e[7]]
    x_errors = [[380,730,869,660],[360,710,701,300]] #FWHM
    bands = [3740,4870,7700,8900]
    sed = [sb_e[0],sb_e[2],sb_e[4],sb_e[6]]
    #need transpose for yerr to work correctly
    y_errors = np.transpose(y_errors)

    axes = AxesSequence()
    for i, ax in zip(range(13), axes):
        get_pixel(img_full, x_pix, y)
        sb_e = img
        img = get_sb_e(img,sb_e)

        y_errors = np.transpose(y_errors)
        y_errors = [sb_e[1], sb_e[3], sb_e[5], sb_e[7]]
        sed = [sb_e[0],sb_e[2],sb_e[4],sb_e[6]]
        #need transpose for yerr to work correctly
        y_errors = np.transpose(y_errors)

        #ax.errorbar(bands, sed, yerr=y_errors, xerr=x_errors)
        ax.plot(bands, sed)
        ax.set_title('test!')
        ax.set_ylabel('Surface Brightness')
        ax.set_xlabel('Wavelength (A)')
        ax.invert_yaxis()
        print x
        print x_pix
        axes.show()
    axes.show()
#    x = np.linspace(0, 10, 100)
#    for i, ax in zip(range(3), axes):
#        ax.plot(x, np.sin(i * x))
#        ax.set_title('Line {}'.format(i))
#    for i, ax in zip(range(5), axes):
#        ax.imshow(np.random.random((10,10)))
#       ax.set_title('Image {}'.format(i))
#    axes.show()

def get_sb_e(img, sb_e):
    sb_e[0] = get_sb(img[0]) #u* sb
    sb_e[2] = get_sb(img[2]) #g sb
    sb_e[4] = get_sb(img[4]) #i sb
    sb_e[6] = get_sb(img[6]) #z sb

    sb_e[1] = get_m_error(img[0], img[1]) #u* error mag tuples
    sb_e[3] = get_m_error(img[2], img[3]) #g error
    sb_e[5] = get_m_error(img[4], img[5]) #i error
    sb_e[7] = get_m_error(img[6], img[7]) # z error

    return sb_e

def get_pixel(img, x ,y):
    print('grabbing pixel')
    print x
    print y
    print('ffffffffffff')
    pixel_img = []  
    pixel_img.append(img[0][y][x].astype(np.float))  #u*
    pixel_img.append(img[2][y][x].astype(np.float))  #g
    pixel_img.append(img[4][y][x].astype(np.float))  #i
    pixel_img.append(img[6][y][x].astype(np.float))  #z
    pixel_img.append(img[1][y][x].astype(np.float))  #u* sig
    pixel_img.append(img[3][y][x].astype(np.float))  #g sig
    pixel_img.append(img[5][y][x].astype(np.float))  #i sig
    pixel_img.append(img[7][y][x].astype(np.float))  #z sig
    print pixel_img
    return pixel_img

def load_fits():
    u_img = pyfits.getdata('../data/VCC1043_U_swarp_fullres.fits')
    g_img = pyfits.getdata('../data/VCC1043_G_swarp_fullres.fits')
    i_img = pyfits.getdata('../data/VCC1043_I_swarp_fullres.fits')
    z_img = pyfits.getdata('../data/VCC1043_Z_swarp_fullres.fits')
    u_sig = pyfits.getdata('../data/VCC1043_U_sigma_swarp_fullres.fits')
    g_sig = pyfits.getdata('../data/VCC1043_G_sigma_swarp_fullres.fits')
    i_sig = pyfits.getdata('../data/VCC1043_I_sigma_swarp_fullres.fits')
    z_sig = pyfits.getdata('../data/VCC1043_Z_sigma_swarp_fullres.fits')
    return [u_img, u_sig, g_img, g_sig, i_img, i_sig, z_img, z_sig]

def get_sb(s):
    s = -2.5*np.log10(np.absolute(s))+30
    s = s.clip(min=15, max=29)
    return s

def get_m_error(s, n):
    u = ((-2.5*np.log10(s)+30.)-(2.5*np.log10(1.+n/s)))-get_sb(s)
    l = get_sb(s)-((-2.5*np.log10(s)+30.)-(2.5*np.log10(1.-n/s)))
    return u, l

class AxesSequence(object):
    global x_pix
    x_pix = 5000
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""
    def __init__(self):
        self.fig = plt.figure()
        self.axes = []
        self._i = 0 # Currently displayed axes index
        self._n = 0 # Last created axes index
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8], 
                               visible=False, label=self._n)
        self._n += 1
        self.axes.append(ax)
        return ax

    def on_keypress(self, event):
        import __main__
        if event.key == 'right':
            __main__.x_pix += 1
            print('test!!')
            print __main__.x_pix
            self.next_plot()
        elif event.key == 'left':
            __main__.x_pix -= 1
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i < len(self.axes):
            self.axes[self._i].set_visible(False)
            self.axes[self._i+1].set_visible(True)
            self._i += 1

    def prev_plot(self):
        if self._i > 0:
            self.axes[self._i].set_visible(False)
            self.axes[self._i-1].set_visible(True)
            self._i -= 1

    def show(self):
        self.axes[0].set_visible(True)
        plt.show()

if __name__ == '__main__':
    main()