import sys, os
import numpy as np
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
from models_profiles import *
from astropy import units as u
from astropy.constants import G,c,M_sun, pc

cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

class Delta_Sigma_fit:
	# R en Mpc, D_Sigma M_Sun/pc2
	#Ecuacion 15 (g(x)/2)

    def __init__(self,R,D_Sigma,err,z, cosmo,fitc = False):

        roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value

        xplot   = np.arange(0.001,R.max()+1.,0.001)
        if fitc:
            def NFW_profile(R,R200,c200):
                M200 = M200_NFW(R200,z,cosmo)
                return Delta_Sigma_NFW(R,z,M200,c200,cosmo=cosmo)
            
            NFW_out = curve_fit(NFW_profile,R,D_Sigma,sigma=err,absolute_sigma=True)
            pcov    = NFW_out[1]
            perr    = np.sqrt(np.diag(pcov))
            e_R200  = perr[0]
            e_c200  = perr[1]
            R200    = NFW_out[0][0]
            c200    = NFW_out[0][1]
            M200    = M200_NFW(R200,z,cosmo)
            
            ajuste  = NFW_profile(R,R200,c200)
            chired  = chi_red(ajuste,D_Sigma,err,2)	
            
            yplot   = NFW_profile(xplot,R200,c200)
    
        else:
            
            def NFW_profile(R,R200):
                M200 = M200_NFW(R200,z,cosmo)
                return Delta_Sigma_NFW(R,z=z,M200=M200,cosmo=cosmo)
            
            NFW_out = curve_fit(NFW_profile,R,D_Sigma,sigma=err,absolute_sigma=True)
            e_R200  = np.sqrt(NFW_out[1][0][0])
            R200    = NFW_out[0][0]
            
            ajuste  = NFW_profile(R,R200)
            
            chired  = chi_red(ajuste,D_Sigma,err,1)	
            
            yplot   = NFW_profile(xplot,R200)
            
            #calculo de c usando la relacion de Duffy et al 2008
            M200   = M200_NFW(R200,z,cosmo)
            
            c200   = c200_duffy(M200*cosmo.h,z)
            e_c200 = 0.
        
        e_M200 =((800.0*np.pi*roc_mpc*(R200**2))/(Msun))*e_R200

        self.xplot = xplot
        self.yplot = yplot
        self.chi2  = chired
        self.R200 = R200
        self.error_R200 = e_R200
        self.M200 = M200
        self.error_M200 = e_M200
        self.c200 = c200
        self.error_c200 = e_c200

