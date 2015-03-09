import fortranfile
import numpy as np
f = fortranfile.FortranFile("../magphys/OptiLIB_cb07.bin")
x = f.readReals()
for i in range(0,24,3):
	params = f.readReals()
	nage = f.readReals()
	sed = f.readReals()
print nage[0:14]


#The 33 values are these, in this order.
#tform,gamma,zmet,tauv0,mu,nburst,mstr1,mstr0,mstry,tlastburst,
#(fburst(i),i=1,5),(ftot(i),i=1,5),age_wm,age_wr,aux,aux,aux,aux,
#lha,lhb,aux,aux,ldtot,fmu,fbc

#Next is the 1200-2000 line. It is:
#nage,(age(i),sfr(i),i=1,nage),(sfrav(i),i=1,5)

#Then the 13000 is:
#(opt_sed(i),opt_sed0(i),i=1,niw_opt)
#Both of the opt_sed's are 6918 long (num of wavelength points)
#both passed into INTERP
#First one is an array of data points for dependent variables
#Second is the interplolated value of y