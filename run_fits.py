import pyfits
from difflib import Differ
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from pylab import *
import os

	
""" Usage ex:
file_names = np.array('../data/example.fits','../data/example_sigma.fits')
#or use get_file_names('../data/list.txt')
d = get_fits_data(file_names)
#d = crop_images(d)
build_obs(d,ymin,ymax,xmin,xmax)
run_fits()
f = os.popen("ls *.sed *.fit")
d = f.read()
while 1:
	d = search(d)
	sleep(10)

"""

def get_cmap(top = 0, bot = 0):
################################################################################################
# To do:
# - Setup if/else so that the top/bottom 1% can be saturated black or white.
# 0 = none, > 1 means white, < 1 means black
################################################################################################
	cdict = {
	'red': (
	(0.0, 1.0, 1.0),
	(0.001, 0.0, 1.0),
	(0.5, 0.8, 0.8),
	(0.99, 0.0, 0.0),
	(1.0, 0.0, 0.0)),
	'green': (
	(0.0, 1.0, 1.0),
	(0.001, 0.0, 0.0),
	(0.5, 1.0, 1.0),
	(0.99, 0.0, 0.0),
	(1.0, 0.0, 0.0)),
	'blue': (
	(0.0, 1.0, 1.0),
	(0.001, 0.0, 0.0),
	(0.5, 0.8, 0.8),
	(0.99, 1.0, 1.0),
	(1.0, 0.0, 0.0))}
	return matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

def get_file_names(x):
################################################################################################
# Input: Location of a file that lists all locations of .fits files. This file must alternate
# count/flux and sigma images starting with count/flux.
# Returns: np array of the .fits locations
################################################################################################
	f = open(x, 'r')
	d = []
	for line in f.readlines():
		d.append(line.strip())
	d = np.asarray(d)
	return d

def get_fits_data(s,ymin,ymax,xmin,xmax):
################################################################################################
# Loads .fits files into an array
# Input: the results of get_file_names() and min/max values (subtract 1 from ds9)
# Output: data cube [counts/flux/sigma(alternating),y,x] even = counts/flux, odd = sigma
################################################################################################
	d = []
	for i in range(0,len(s)):
		d.append(pyfits.getdata(s[i])[ymin:ymax,xmin:xmax])
	return d

def get_mag(f, c=30.): #flux, optional constant - defaults to 30
################################################################################################
# 
################################################################################################
	x = -2.5*np.log10(f)+c
	return x #returns mag

def crop_images(a,ymin,ymax,xmin,xmax): #a = array from get_fits data
################################################################################################
# No longer needed... I dont think
################################################################################################
	for i in range(0,len(a)):
		a[i] = a[i][ymin:ymax][xmin:xmax]
	return a #cropped version of a

def get_ab(s, c=30.): #flux, optional constant - defaults to 30
################################################################################################
# 
################################################################################################
    n = -2.5*np.log10(np.absolute(s))+c
    return n

def get_jansky(s): 
################################################################################################
# Needed to build the oberservation.dat files for Magphys
################################################################################################
	j = 10.**((8.9-get_ab(s))/2.5)
	return j


def build_obs(a,ymin,ymax,xmin,xmax): #a is the fits data from get_fits_data(). Use min/max values from ds9
################################################################################################
# To do:
# - Update it so ymin/max vals aren't used for getting counts anymore.
################################################################################################
	redshift = 0.004 #real value is 0.000237

	#File IO stuff
	f = open('../magphys/eg_user_files/observations.dat', 'wb')
	f.write("#Header text\n")

	for i in range(0,(ymax-ymin)):
		for j in range(0,(xmax-xmin)):
			f.write(str(i+ymin)+str(j+xmin)+"         "+str(redshift)+"      ")
			for k in range(1,len(a),2):
				f.write(str(get_jansky(a[k-1][i,j].astype(np.float)))+"   "+str(get_jansky(a[(k)][i,j].astype(np.float)))+"   ")
			f.write("\n")
	f.close()

def get_sed_header(f):
################################################################################################
# 
################################################################################################
	f = open(f, 'r')
	h = np.array([[],[]])	
	f.readline()
	f.readline()
	f.readline()
	h[0] = f.readline().split()
	f.readline()
	f.readline()
	h[1] = f.readline().split()
	f.readline()
	f.readline()
	f.readline()
	f.close()
	return h.astype(np.float)

def get_sed(f):
################################################################################################
# 
################################################################################################
	f = open(f, 'r')
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	d = []
	for line in f:
		d.append(line.strip().split())
	d = np.asarray(d)
	f.close()
	return d.astype(np.float)

def get_fit_header(f):
################################################################################################
# 
################################################################################################
	f = open(f, 'r')
	#h = np.array([[],[],[],[],[]])
	h=[[],[],[],[],[],[]]
	f.readline()
	f.readline()
	h[0] = f.readline().split()
	h[1] = f.readline().split()
	f.readline()
	f.readline()
	h[2] = f.readline().split()
	f.readline()
	h[3] = f.readline().split()
	f.readline()
	h[4] = f.readline().split()
	f.readline()
	h[5] = f.readline().split()
	f.close()
	return h

def get_fit(f):
################################################################################################
# get_fit()
# Stores probability distrubutions from the i_gal.fit file. Skips header info (see get_fit_header)
# Input: f is a string containing the location of the .fit file eg('../data/name.fit')
# Returns: a 3d np array of the data set in the form [header,datapoints,0-1(axis)]
################################################################################################
	f = open(f, 'r')
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	d = []
	test = True
	l = 0
	spots = []
	for line in f:
		if test and line[0] != '#':
			spots.append(l)
		if line[0] != '#':
			d.append(line.strip().split())
			test = False
			l += 1
		else:
			test = True
	s = []
	prev = 0
	for x in spots:
		s.append(d[prev:x])
		prev = x 
	s.append(d[x])
	s = np.delete(s,0,axis=0)
	for i in range(0,len(s)):
		s[i] = np.asarray(s[i][:][:])
	f.close()
	return s 


def run_fits():
################################################################################################
# Runs magphys fortran code
# Currently not working. the command "source" and env-var $magphys are not defined in the shell
# python opens
################################################################################################
	os.system("cd $magphys")
	os.system("source fit_sample")


def search(old): 
################################################################################################
# Waits for files to be created by magphys and builds fit objects as they appear.
# Input: The results of "ls *.sed *.fit" as a string from the previous iteration
# Returns: The updated results for ls (if changed at all)
# To do:
# - Get env variables working
# - Delete files and backup fit objects as they are made
################################################################################################
	os.system("cd $magphys") #currently not working, $magphys is not defined in python shell
	f = os.popen("ls *.s *.f")
	new = f.read()
	x = list(Differ().compare(new, old))
	diff = "".join([i[2:] for i in x if not i[:1] in '-?' ]) #0x5f3759df - ( i >> 1 )
	diff = diff.split()
	print new
	catch = np.array(['',''])
	if len(diff) == 2:
		catch[0] = diff[0]
		catch[1] = diff[1]
		x = fit(catch)
		x.save()
		return new
	else:
		return old


def plot_map(img,sn_map,f=['','',''],sn=False):
################################################################################################
# img is a matrix of the pixels to be plotted. f is an array of strings [cb_label,x_label,y_label]
# If sn is set to true sn_map must by a boolean matrix of the same shape as img containing zeros
# where we wish to mask a pixel to to S/N issues.
################################################################################################
	cmap = get_cmap()

	if(sn):
		img[np.where(sn_map == 0)] = 0

	fig, ax = plt.subplots()
	ax.xaxis.set_tick_params(width=1, length=5)
	ax.yaxis.set_tick_params(width=1, length=5)

	plt.imshow(img,cmap=cmap,origin='lower')
	cb = plt.colorbar()
	cb.set_label(f[0]) #f definitions here
	ax.set_xlabel(f[1])
	ax.set_ylabel(f[2])
	plt.show()


def build_sn_map(a,sn_min=5):
################################################################################################
# Input: Alternating counts/flux map and sigma maps. ex: the results of get_fits_data()
# Returns: A boolean (1 or 0) array on the same shape as a. 0 means the S/N is too low.
################################################################################################
	sn = np.array(len(a))
	x = np.ones(np.shape(a[0]))
	for i in range(0,(len(a)/2)):	
		a[(2*i)+1][np.where(a[(2*i)+1] == 0.)] = 0.000001
		sn = np.absolute(a[2*i]/a[(2*i)+1])
		x[np.where(sn <= sn_min)] = 0.
	w = np.where(x == 0.)
	return w

class fit(object):
################################################################################################
# Stores everything in the .fit and .sed files
################################################################################################
	def __init__(self, catch):
		self.sed = get_sed(catch[0])
		self.fit = get_fit(catch[1])
		self.sed_h = get_sed_header(catch[0])
		self.fit_h = get_fit_header(catch[1])

		#sed header-------------------------------------------------------
		#self. = self.sed_h[]
		self.bf_t_form_yr = self.sed_h[0:2]
		self.bf_gamma = self.sed_h[0:3]
		self.bf_z_z0 = self.sed_h[0:4]
		self.bf_m_m_sun = self.sed_h[0:7]
		self.bf_l_d_l_sun = self.sed_h[0:9]

		#sed--------------------------------------------------------------
		#self. = self.sed[]
		#fit header-------------------------------------------------------
		#self. = self.fit_h[]
		#self.chi_2 = self.fit_h[2:]
		self.bf_f_mu_sfh = self.fit_h[3:0]
		self.bf_f_mu_ir = self.fit_h[3:1]
		self.bf_mu = self.fit_h[3:2]
		self.bf_tau_v = self.fit_h[3:3]
		self.bf_s_sfr = self.fit_h[3:4]
		self.bf_m = self.fit_h[3:5]
		self.bf_l_dust = self.fit_h[3:6]
		self.bf_t_w_bc = self.fit_h[3:7]
		self.bf_t_c_ism = self.fit_h[3:8]
		self.bf_xi_c_tot = self.fit_h[3:9]
		self.bf_xi_pah_tot = self.fit_h[3:10]
		self.bf_xi_mir_tot = self.fit_h[3:11]
		self.bf_xi_w_tot = self.fit_h[3:12]
		self.bf_tau_v_ism = self.fit_h[3:13]
		self.bf_m_dust = self.fit_h[3:14]
		self.bf_sfr = self.fit_h[3:15]

		#fit--------------------------------------------------------------
		#. = self.fit[]
		self.f_mu_sfh = self.fit[0]
		self.f_mu_sfh_pdf = self.fit[1]
		self.f_mu_ir = self.fit[2]
		self.f_mu_ir_pdf = self.fit[3]
		self.mu_param = self.fit[4] #sb
		self.mu_param_pdf = self.fit[5]
		self.tau_v = self.fit[6] #optical depth
		self.tau_v_pdf = self.fit[7]
		self.ssfr0_1gyr = self.fit[8]
		self.ssfr0_1gyr_pdf = self.fit[9]
		self.m_stars = self.fit[10]
		self.m_stars_pdf = self.fit[11]
		self.l_dust = self.fit[12]
		self.l_dust_pdf = self.fit[13]
		self.t_c_ism = self.fit[14]
		self.t_c_ism_pdf = self.fit[15]
		self.t_w_bc = self.fit[16]
		self.t_w_bc_pdf = self.fit[17]
		self.xi_c_tot = self.fit[18]
		self.xi_c_tot_pdf = self.fit[19]
		self.xi_pah_tot = self.fit[20]
		self.xi_pah_tot_pdf = self.fit[21]
		self.xi_mir_tot = self.fit[22]
		self.xi_mir_tot_pdf = self.fit[23]
		self.xi_w_tot = self.fit[24]
		self.xi_w_tot_pdf = self.fit[25]
		self.tau_v_ism = self.fit[26]
		self.tau_v_ism_pdf = self.fit[27]
		self.m_dust = self.fit[28]
		self.m_dust_pdf = self.fit[29]
		self.sfr0_1gyr = self.fit[30]
		self.sfr0_1gyr_pdf = self.fit[31]
		#do stuffs
	#def save(self):
		#saves as a python pickle - look up how to do this
	#def plot(self):
		#do stuffs
