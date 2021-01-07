import sys, os
import numpy as np
from pylab import *
from astropy.cosmology import LambdaCDM
from scipy.misc import derivative
from scipy import integrate
from multiprocessing import Pool
from multiprocessing import Process
import time
from astropy.constants import G,c,M_sun, pc
from astropy import units as u
cosmo = LambdaCDM(H0=100, Om0=0.3, Ode0=0.7)

cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

def Gamma(Roff,s_off):
    return (Roff/s_off**2)*np.exp(-1.*(Roff/s_off))

def Rayleigh(Roff,s_off):
    return (Roff/s_off**2)*np.exp(-0.5*(Roff/s_off)**2)


def chi_red(ajuste,data,err,gl):
	'''
	Reduced chi**2
	------------------------------------------------------------------
	INPUT:
	ajuste       (float or array of floats) fitted value/s
	data         (float or array of floats) data used for fitting
	err          (float or array of floats) error in data
	gl           (float) grade of freedom (number of fitted variables)
	------------------------------------------------------------------
	OUTPUT:
	chi          (float) Reduced chi**2 	
	'''
		
	BIN=len(data)
	chi=((((ajuste-data)**2)/(err**2)).sum())/float(BIN-1-gl)
	return chi



def c200_duffy(M,z):
    #calculo de c usando la relacion de Duffy et al 2008
    return 5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)
    
def R200_NFW(M200,z,cosmo=cosmo):	
    '''
    
    Returns the R_200
    ------------------------------------------------------------------
    INPUT:
    M200         (float or array of floats) M_200 mass in solar masses
    roc_mpc      (float or array of floats) Critical density at the z 
                of the halo in units of kg/Mpc**3
    ------------------------------------------------------------------
    OUTPUT:
    R_200         (float or array of floats) 
    
    '''

    roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value
    
    return ((M200*(3.0*Msun))/(800.0*np.pi*roc_mpc))**(1./3.)
    

def M200_NFW(R200,z,cosmo=cosmo):	
    
    '''
    
    Returns the R_200
    ------------------------------------------------------------------
    INPUT:
    M200         (float or array of floats) M_200 mass in solar masses
    roc_mpc      (float or array of floats) Critical density at the z 
                of the halo in units of kg/Mpc**3
    ------------------------------------------------------------------
    OUTPUT:
    R_200         (float or array of floats) 
    
    '''

    roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value
    
    return (800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun)



def Delta_Sigma_NFW(R,z,M200,c200 = None,cosmo=cosmo):	
    '''
    Density contraste for NFW
    
    '''
  
    R200 = R200_NFW(M200,z,cosmo)
    
    roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value
    
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)
    
    ####################################################
    
    deltac=(200./3.)*( (c200**3) / ( np.log(1.+c200)- (c200/(1+c200)) ))
    x=np.round((R*c200)/R200,12)
    m1= x< 1.0
    m2= x> 1.0 
    m3= (x == 1.0)
    
    try: 
        jota=np.zeros(len(x))
        atanh=np.arctanh(((1.0-x[m1])/(1.0+x[m1]))**0.5)
        jota[m1]=(4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5)) \
            + (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0) \
            + (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5))    
        atan=np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
        jota[m2]=(4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5)) \
            + (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0) \
            + (2.0*atan)/((x[m2]**2.0-1.0)**1.5)
        jota[m3]=2.0*np.log(0.5)+5.0/3.0
    except:
        if m1:
            atanh=np.arctanh(((1.0-x[m1])/(1.0+x[m1]))**0.5)
            jota = (4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5)) \
                + (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0) \
                + (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5))   
        if m2:		 
            atan=np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
            jota = (4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5)) \
                + (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0) \
                + (2.0*atan)/((x[m2]**2.0-1.0)**1.5)
        if m3:
            jota = 2.0*np.log(0.5)+5.0/3.0
    
    
        
    rs_m=(R200*1.e6*pc)/c200
    kapak=((2.*rs_m*deltac*roc_mpc)*(pc**2/Msun))/((pc*1.0e6)**3.0)
    return kapak*jota



def Sigma_NFW(R,z,M200,c200 = None,cosmo=cosmo):			
    '''
    Projected density for NFW
    
    '''		
  
		
    
    if not isinstance(R, (np.ndarray)):
        R = np.array([R])


    m = R == 0.
    R[m] = 1.e-8  

    
    R200 = R200_NFW(M200,z,cosmo)
    
    roc_mpc = cosmo.critical_density(z).to(u.kg/(u.Mpc)**3).value
    
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)
    
    ####################################################
    
    deltac=(200./3.)*( (c200**3) / ( np.log(1.+c200)- (c200/(1+c200)) ))
    
    x=(R*c200)/R200
    m1 = x <= (1.0-1.e-12)
    m2 = x >= (1.0+1.e-12)
    m3 = (x == 1.0)
    m4 = (~m1)*(~m2)*(~m3)
    
    jota  = np.zeros(len(x))
    atanh = np.arctanh(np.sqrt((1.0-x[m1])/(1.0+x[m1])))
    jota[m1] = (1./(x[m1]**2-1.))*(1.-(2./np.sqrt(1.-x[m1]**2))*atanh) 
    
    atan = np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
    jota[m2] = (1./(x[m2]**2-1.))*(1.-(2./np.sqrt(x[m2]**2 - 1.))*atan) 
    
    jota[m3] = 1./3.
    
    x1 = 1.-1.e-4
    atanh1 = np.arctanh(np.sqrt((1.0-x1)/(1.0+x1)))
    j1 = (1./(x1**2-1.))*(1.-(2./np.sqrt(1.-x1**2))*atanh1) 
    
    x2 = 1.+1.e-4
    atan2 = np.arctan(((x2-1.0)/(1.0+x2))**0.5)
    j2 = (1./(x2**2-1.))*(1.-(2./np.sqrt(x2**2 - 1.))*atan2) 
    
    jota[m4] = np.interp(x[m4].astype(float64),[x1,x2],[j1,j2])
                
    rs_m=(R200*1.e6*pc)/c200
    kapak=((2.*rs_m*deltac*roc_mpc)*(pc**2/Msun))/((pc*1.0e6)**3.0)
    
    return kapak*jota


def GAMMA_components(R,z,ellip,M200,c200 = None,cosmo=cosmo):
   
    '''
    Quadrupole term defined as (d(Sigma)/dr)*r
    
    '''
    
    def monopole(R):
        return Sigma_NFW(R,z,M200,c200,cosmo=cosmo)
    
    m0p = derivative(monopole,R,dx=1e-5)
    q   =  m0p*R
    
    '''
    vU_Eq10
    
    '''
    
    integral = []
    for r in R:
        argumento = lambda x: (x**3)*monopole(x)
        integral += [integrate.quad(argumento, 0, r, epsabs=1.e-01, epsrel=1.e-01)[0]]
        
    p2 = integral*(-2./(R**2))
    m  = monopole(R)
    
    gt2 = ellip*((-6*p2/R**2) - 2.*m + q)
    gx2 = ellip*((-6*p2/R**2) - 4.*m)
    
    return [gt2,gx2]


def Sigma_NFW_miss(R,z,M200,s_off = None, tau = 0.2,
                   c200 = None, P_Roff = Gamma, cosmo=cosmo):
                       
    
    '''
    Misscentred density NFW profile
    F_Eq12
    '''
    

    
    R200 = R200_NFW(M200,z,cosmo)
    
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)

    if not s_off:
        s_off = tau*R200


    def DS_RRs(Rs,R):
        # F_Eq13
        #argumento = lambda x: monopole(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x)))
        #integral  = integrate.quad(argumento, 0, 2.*np.pi, epsabs=1.e-01, epsrel=1.e-01)[0]
        x = np.linspace(0.,2.*np.pi,500)
        integral  = integrate.simps(Sigma_NFW((np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x))),z,M200,c200,cosmo),x,even='first')
        return integral/(2.*np.pi)

    
    
    integral = []
    for r in R:
        argumento = lambda x: DS_RRs(x,r)*P_Roff(x,s_off)
        integral  += [integrate.quad(argumento, 0, np.inf, epsabs=1.e-02, epsrel=1.e-02)[0]]
        
    return integral


def Delta_Sigma_NFW_miss(R,z,M200,s_off = None, tau = 0.2,
                         c200 = None, P_Roff = Gamma, cosmo=cosmo):	
    
    '''
    Misscentred density contraste for NFW
    
    '''
  
    R200 = R200_NFW(M200,z,cosmo)
    
        
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)

    if not s_off:
        s_off = tau*R200


    integral = []
    for r in R:
        argumento = lambda x: Sigma_NFW_miss([x],z,M200,s_off,tau,c200,P_Roff,cosmo)[0]*x
        integral  += [integrate.quad(argumento, 0, r, epsabs=1.e-02, epsrel=1.e-02)[0]]

    DS_off    = (2./R**2)*integral - Sigma_NFW_miss(R,z,M200,s_off,tau,c200,P_Roff,cosmo)

    return DS_off

    
