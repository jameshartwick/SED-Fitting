# Work in progress - script watches for new files from Magphys and collects data
# as they appear.

import pyfits 
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from pylab import *
import os
""" Usage ex:
file_names = np.array('../data/example.fits','../data/example_sigma.fits')
#or use get_file_names('../data/list.txt')
d = get_files_data(file_names)
#d = crop_images(d)
build_obs(d,ymin,ymax,xmin,xmax)
run_fits()
while 1:
	search()
	sleep(10)

"""

def get_file_names(x):
	f = open(x, 'r')
	d = f.readline().split()
	d = np.asarray(d)
	return d

def get_fits_data(s): #input must be np arrays: s = alternating signal/sigma fits image locations
	d = np.empty((len(s)+len(n)))
	for i, image in range(0,len(s)),s:
		d[i] = pyfits.getdata(s[i])
	return d #np data cube [band/sigma,y,x]

def get_mag(f, c=30): #flux, optional constant - defaults to 30
	x = -2.5*np.log10(f)+c
	return x #returns mag

#crop images not needed
def crop_images(a,ymin,ymax,xmin,xmax): #a = array from get_fits data
	for i in range(0,len(a)):
		a[i] = a[i,ymin:ymax,xmin:xmax]
	return a #cropped version of a

def get_ab(s):
    n = -2.5*np.log10(np.absolute(s))+30.
    return n

def get_jansky(s):
	j = 10**((8.9-get_ab(s))/2.5)
	return j

def build_obs(a,ymin,ymax,xmin,xmax): #a is the fits data from get_fits_data(). Use min/max values from ds9
	redshift = 0.000237

	#File IO stuff
	f = open('observations.dat', 'wb')
	f.write("#Header text\n")

	#printing
	for i in range(ymin-1,ymax):
		for j in range(xmin-1,xmax):
			u_e = get_jansky(a[1,i,j].astype(np.float))
			g_e = get_jansky(a[3,i,j].astype(np.float))
			i_e = get_jansky(a[5,i,j].astype(np.float))
			z_e = get_jansky(a[7,i,j].astype(np.float))
			u_img = get_jansky(a[0,i,j].astype(np.float))
			g_img = get_jansky(a[2,i,j].astype(np.float))
			i_img = get_jansky(a[4,i,j].astype(np.float))
			z_img = get_jansky(a[6,i,j].astype(np.float))
			f.write(str(j+1)+str(i+1)+"         "+str(redshift)+"      "+str(u_img)+"   "+str(u_e)+"   "+str(g_img)+"   "+str(g_e)+"   "+str(i_img)+"   "+str(i_e)+"   "+str(z_img)+"   "+str(z_e)+"\n")
	f.close()
	print(time.time() - start_time, "seconds")

def get_sed_header(f):
	f = open(f, 'r')
	h = np.array(['',''])
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
	return h

def get_sed(f):
	f = open(f, 'r')
	get_sed_header(f) #bandaid fix to get file reader into the correct position
	d = []
	for line in f:
		d.append(line.strip().split())
	d = np.asarray(d)
	return d

def get_fit_header(f):
	f = open(f, 'r')
	h = np.array([[],[],[],[],[]])
	f.readline()
	f.readline()
	h[0] = f.readline().split()
	h[1] = f.readline().split()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	f.readline()
	h[2] = f.readline().split()
	f.readline()
	h[3] = f.readline().split()
	f.readline()
	h[4] = f.readline().split()
	return h

def get_fit(f): #input is a string containing the location of the .fit file eg('../data/name.fit')
	f = open(f, 'r')
	get_fit_header(f) #band aid fix to get the file reader in the correct position
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
			l+=1
		else:
			test = True
	s = []
	prev = 0
	for x in spots:
		s.append(d[prev:x])
		prev = x 
	s.append(d[x])
	s = np.delete(s,0,axis=0)
	s = np.asarray(s)
	for i in range(0,len(s)): #band aid fix to convert to np arrays
		s[i] = np.asarray(s[i][:][:])
	return s #returns a 3d np array of the data set in the form [header,datapoints,0-1(axis)]

def run_fits():
	os.system("cd $magphys")
	os.system("source ./.magphys_tcshrc")
	os.system("./make_zgrid")
	os.system("y")
	os.system("source get_libs")
	os.system("source fit_sample")
	#run magphys fortran code

def search(): # #wait for files to be created by magphys - currently lots of pseudo code
	new = old
	os.system("cd $magphys")
	f = os.popen("ls *.sed *.fit")
	new = f.read()
	catch = new-old #array containing matching .sed and .fit files
	if len(catch) == 2:
		x = fit(catch)
		x.save() 

class fit(object)
	def __init__(self, catch):
		self.__sed = catch[0]
		self.__fit = catch[1]
		#sed
		self. = self.__sed[]
		#fit
		self. = self.__fit[]
		#do stuffs
	def get_(self):
		#do stuffs
		return self.__
	def get_(self):
		#do stuffs
		return self.__
	def get_(self):
		#do stuffs
		return self.__

file_names = np.array('../data/example.fits','../data/example_sigma.fits')
#or use get_file_names('../data/list.txt')
d = get_fits_data(file_names)
#d = crop_images(d)
build_obs(d)
run_fits()
while 1:
	search()
	sleep(10)