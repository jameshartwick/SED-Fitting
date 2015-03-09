import run_fits as rf
import pyfits
from difflib import Differ
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from pylab import *
import os


z = 0
min_ld = 0.01
ymin = 4.2
ymax = 5.5
xmin = 3500
xmax = 9000
ylim_u = 0.40
yticks_u = [0.0, 0.1, 0.2, 0.3, 0.4]
ylim_l = 0.25
yticks_l = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
catch = ['19582337.s','19582337.f']

#name = "(x,y) = (" + catch[0][0:4] + ", " + catch[0][4:8] + ")"



x = rf.fit(catch)


lambda_x = [3740., 4870., 6360., 7700.,8900.]#,21460]
lambda_eff = [0.374,0.487,0.636,0.77,0.89]#,2.146]
lambda_err = np.asarray([float(i) for i in x.fit_h[:][1]])
lambda_val = np.log10((1.+z)*np.asarray([float(i) for i in x.fit_h[:][0]])*3.0e+14/lambda_eff)
lambda_err_l = np.log10(np.absolute((1.+z)*lambda_val*3.0e+14/lambda_eff))-np.log10(np.absolute(((1.+z)*lambda_val*3.0e+14/lambda_eff)-(lambda_err*(1.+z)*3.0e+14/lambda_eff)))
lambda_err_u = -np.log10(np.absolute((1.+z)*lambda_val*3.0e+14/lambda_eff))+np.log10(np.absolute((1.+z)*lambda_val*3.0e+14/lambda_eff+lambda_err*(1.+z)*3.0e+14/lambda_eff))

print lambda_err_l

fig = plt.figure(figsize=(14, 9))
fig.suptitle('Best Fit Model and Probability Distributions for Pixel: (' + catch[0][0:4]+', '+ catch[0][4:8]+')',size=15)
ax1 = fig.add_axes([0.1, 0.51, 0.8, 0.44])
ax1.set_xlabel(r'$\lambda \ (\AA)$')
ax1.set_ylabel(r'$log(\lambda L_{\lambda}/L_0)$')
ax1.plot(10**x.sed[:,0], np.log10((np.power(10,x.sed[:,0]))*(np.power(10,x.sed[:,2]))))
ax1.plot(10**x.sed[:,0], np.log10((np.power(10,x.sed[:,0]))*(np.power(10,x.sed[:,1]))))
ax1.plot(lambda_x, lambda_val, 'ro', markersize=7)



ax1.errorbar(lambda_x, lambda_val, xerr=[lambda_err_l,lambda_err_u],zorder=3,capthick=2,fmt = None)
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])

currentvar = x.f_mu_sfh
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax2 = fig.add_axes([0.1, 0.3, (0.8/6.), 0.15])
ax2.set_yticks(yticks_u)
ax2.tick_params(axis='both', which='major', labelsize=6)
ax2.set_ylim([0,ylim_u])
ax2.set_xlabel(r'$f_\mu$')
ax2.set_ylabel(r'$Likelhood \ Distr.$')
ax2.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)

currentvar = x.tau_v
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax3 = fig.add_axes([(.8/6.+.1), 0.3, (.8/6.), 0.15])
ax3.set_yticks(yticks_u)
ax3.tick_params(axis='both', which='major', labelsize=6)
ax3.set_ylim([0,ylim_u])
ax3.set_xlabel(r'$\tau_V$')
ax3.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax3.set_yticklabels('')

currentvar = x.tau_v_ism
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax4 = fig.add_axes([(1.6/6.+.1), 0.3, (.8/6.), 0.15])
ax4.set_yticks(yticks_u)
ax4.tick_params(axis='both', which='major', labelsize=6)
ax4.set_ylim([0,ylim_u])
ax4.set_xlabel(r'$\mu\tau_V$')
ax4.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax4.set_yticklabels('')

currentvar = x.m_stars
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax5 = fig.add_axes([(2.4/6.+.1), 0.3, (.8/6.), 0.15])
ax5.set_yticks(yticks_u)
ax5.tick_params(axis='both', which='major', labelsize=6)
ax5.set_ylim([0,ylim_u])
ax5.set_xlabel(r'$log(M_{stars}/M_0)$')
ax5.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax5.set_yticklabels('')

currentvar = x.ssfr0_1gyr
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax6 = fig.add_axes([(3.2/6.+.1), 0.3, (.8/6.), 0.15])
ax6.set_yticks(yticks_u)
ax6.tick_params(axis='both', which='major', labelsize=6)
ax6.set_ylim([0,ylim_u])
ax6.set_xlabel(r'$log(sSFR)yr^{-1}$')
ax6.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax6.set_yticklabels('')

currentvar = x.sfr0_1gyr
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax7 = fig.add_axes([(4./6.+.1), 0.3, (.8/6.), 0.15])
ax7.set_yticks(yticks_u)
ax7.tick_params(axis='both', which='major', labelsize=6)
ax7.set_ylim([0,ylim_u])
ax7.set_xlabel(r'$log(SFR/M_0yr^{-1})$')
ax7.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax7.set_yticklabels('')

currentvar = x.l_dust
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax8 = fig.add_axes([0.1, 0.1, (0.8/6.), 0.15])
ax8.set_yticks(yticks_l)
ax8.tick_params(axis='both', which='major', labelsize=6)
ax8.set_ylim([0,ylim_l])
ax8.set_xlabel(r'$log(L_{dust}/L_0)$')
ax8.set_ylabel(r'$Likelhood\ Distr.$')
ax8.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)

currentvar = x.m_dust
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax9 = fig.add_axes([(.8/6.+.1), 0.1, (.8/6.), 0.15])
ax9.set_yticks(yticks_l)
ax9.tick_params(axis='both', which='major', labelsize=6)
ax9.set_ylim([0,ylim_l])
ax9.set_xlabel(r'$log(M_{dust}/M_0)$')
ax9.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax9.set_yticklabels('')

currentvar = x.t_c_ism
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax10 = fig.add_axes([(1.6/6.+.1), 0.1, (.8/6.), 0.15])
ax10.set_yticks(yticks_l)
ax10.tick_params(axis='both', which='major', labelsize=6)
ax10.set_ylim([0,ylim_l])
ax10.set_xlabel(r'$T^{ISM}_C/K$')
ax10.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax10.set_yticklabels('')

currentvar = x.t_w_bc
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax11 = fig.add_axes([(2.4/6.+.1), 0.1, (.8/6.), 0.15])
ax11.set_yticks(yticks_l)
ax11.tick_params(axis='both', which='major', labelsize=6)
ax11.set_ylim([0,ylim_l])
ax11.set_xlabel(r'$T^{BC}_W/K$')
ax11.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax11.set_yticklabels('')

currentvar = x.xi_c_tot
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax12 = fig.add_axes([(3.2/6.+.1), 0.1, (.8/6.), 0.15])
ax12.set_yticklabels('')
ax12.set_yticks(yticks_l)
ax12.tick_params(axis='both', which='major', labelsize=6)
ax12.set_ylim([0,ylim_l])
ax12.set_xlabel(r'$\xi^{tot}_C$')
ax12.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)

currentvar = x.xi_w_tot
minplot = np.min(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
maxplot = np.max(np.where(currentvar[:,1][:].astype(np.float) > min_ld))
ax13 = fig.add_axes([(4./6.+.1), 0.1, (.8/6.), 0.15])
ax13.set_yticks(yticks_l)
ax13.tick_params(axis='both', which='major', labelsize=6)
ax13.set_ylim([0,ylim_l])
ax13.set_xlabel(r'$\xi^{tot}_w$')
ax13.bar(currentvar[minplot:maxplot,0].astype(np.float),currentvar[minplot:maxplot,1].astype(np.float), width = .05)
ax13.set_yticklabels('')


plt.show()


"""
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylim([0,0.40])
ax.set_xlabel('')
ax.bar(x.[:,0].astype(np.float),x.[:,1].astype(np.float), width = .05)
"""



"""
def f(t):
    val = np.exp(-t) * np.cos(2*pi*t)
    return val

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

#x.f_mu_sfh[0],x.f_mu_sfh[1]

plt.figure(1)             # Make the first figure
plt.clf()
plt.subplot(2,6,7)  # 2 rows, 1 column, plot 1
#plt.plot(x.f_mu_sfh[0],x.f_mu_sfh[1])
plt.title('FIGURE 1')
plt.text(2, 0.8, 'AXES 211')

plt.subplot(2,6,9)  # 2 rows, 1 column, plot 2
plt.plot(t2, np.cos(2*pi*t2), 'r--')
plt.text(2, 0.8, 'AXES 212')

plt.subplot(2,6,10)  # 2 rows, 1 column, plot 2
plt.plot(t2, np.cos(2*pi*t2), 'r--')
plt.text(2, 0.8, 'AXES 212')

plt.figure(2)             # Make a second figure
plt.xlim([2.5,7])
plt.ylim([0,10])
plt.plot(x.sed[:,0], x.sed[:,2])
plt.plot(x.sed[:,0], x.sed[:,1])
plt.show()

plt.figure(1)             # Select the existing first figure
plt.subplot(2,6,2)          # Select the existing subplot 212
plt.plot(t2, np.cos(pi*t2), 'g--')   # Add a plot to the axes
plt.text(2, -0.8, 'Back to AXES 212')"""