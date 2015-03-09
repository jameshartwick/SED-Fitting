import pyfits 
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from pylab import *

def get_header(f):
	h = ['','']
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
	d = []
	for line in f:
		d.append(line.strip().split())
	d = np.array(d)
	return d

def plot_sed(h, d):
	length = len(d[:,0])-1
	plt.plot(d[:,2])
	return None

def format():
	return None


fig, ax = plt.subplots()
f = open('4170.sed', 'r')
#data = f.read()

header = get_header(f)
data = get_sed(f)
data = np.array(data)
plot_sed(header, data)
format()
plt.xscale('log')
plt.show()