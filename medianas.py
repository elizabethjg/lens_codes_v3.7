import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pylab as pylab
import time
from scipy.optimize import curve_fit
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})

def separate_medianas(x,y,plot = True,label_x ='r',label_y = 'y',out_plot = 'plot.eps'):

	x_bins = np.percentile(x,np.arange(0,110,10))

	digit = np.digitize(x,x_bins)

	x_mean = np.zeros(len(x_bins)-1)
	y_mean = np.zeros(len(x_bins)-1)
	
	for j in range(max(digit)-1):
		x_mean[j] = 0.5*(x_bins[j]+x_bins[j+1])
		y_mean[j]= np.median(y[digit==(j+1)])
	
	
	popt, pcov = curve_fit(lambda x,m,n: x*m+n, x_mean, y_mean)
	m,n = popt
	
	above = np.zeros(len(x))

	
	mabove = y > (x*m+n)

	above[mabove] = 1

	
	if plot:
		
	
		fig = plt.figure(figsize=(6,6))
		fig.subplots_adjust(hspace=0,wspace=0)
		
		gs = GridSpec(4,4)
		
		ax_joint = fig.add_subplot(gs[1:4,0:3])
		ax_marg_x = fig.add_subplot(gs[0,0:3])
		ax_marg_y = fig.add_subplot(gs[1:4,3])
		
		ax_joint.plot(x[mabove],y[mabove],'.',color='palevioletred',alpha = 0.7)
		ax_joint.plot(x[~mabove],y[~mabove],'C9.', alpha = 0.7)
		# ax_joint.plot(x,y,'C5,', alpha = 0.8)
		#ax_joint.plot(x_mean,y_mean,'k.', alpha = 0.8)
		ax_joint.plot(x,m*x+n,'k')
		ax_joint.set_xlabel(label_x,fontsize=14)
		ax_joint.set_ylabel(label_y,fontsize=14)
		
		
		ax_marg_x.hist(x[mabove],50,histtype='step',stacked=True,fill=False,color='palevioletred',lw=2)
		ax_marg_x.hist(x[~mabove],50,histtype='step',stacked=True,fill=False,color='C9',lw=2)
		ax_marg_x.set_ylabel(r'$N$',fontsize=14)
		
		ax_marg_y.hist(y[mabove],50,orientation="horizontal",histtype='step',stacked=True,fill=False,color='palevioletred',lw=2)
		ax_marg_y.hist(y[~mabove],50,orientation="horizontal",histtype='step',stacked=True,fill=False,color='C9',lw=2)
		ax_marg_y.set_xlabel(r'$N$',fontsize=14)
		
		
		plt.setp(ax_marg_x.get_xticklabels(), visible=False)
		plt.setp(ax_marg_y.get_yticklabels(), visible=False)
		plt.savefig(out_plot, format='pdf',bbox_inches='tight')
		plt.show()
	
	return x_mean,y_mean,m,n,mabove

def select_medianas_disp(x,y,plot = False,label_x ='x',label_y = 'y',out_plot = 'plot.eps'):

	x_bins = np.percentile(x,np.arange(0,110,10))
	x_bins[0] = x.min() - 1
	x_bins[-1] = x.max() + 1

	digit = np.digitize(x,x_bins)

	x_bins[0] = x.min() 
	x_bins[-1] = x.max() 


	y_mean = np.zeros(len(x_bins)-1)
	y_std  = np.zeros(len(x_bins)-1)
	mask = np.zeros(len(x)).astype(bool)

	x_mean = 0.5*(x_bins[1:]+x_bins[:-1])
	
	for j in range(max(digit)):
		y_mean[j]= np.median(y[digit==(j+1)])
		y_std[j]= np.std(y[digit==(j+1)])
		m = np.abs(y[digit==(j+1)] - y_mean[j]) < 0.5*y_std[j]
		mask[digit==(j+1)] = m
		
	
	
	
	if plot:
		
		rcParams['figure.figsize'] =10.,10.
		
		fig = plt.figure(figsize=(6,6))
		
		
		plt.plot(x_mean,y_mean,'k')
		plt.fill_between(x_mean, y_mean-y_std, y_mean+y_std,facecolor = '0.8')
		plt.xlabel(label_x,fontsize=14)
		plt.ylabel(label_y,fontsize=14)
		
		
		plt.savefig(out_plot, format='eps',bbox_inches='tight')
		plt.show()
	
	return mask
