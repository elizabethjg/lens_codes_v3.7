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


    def S_RRs(Rs,R):
        # F_Eq13
        #argumento = lambda x: monopole(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x)))
        #integral  = integrate.quad(argumento, 0, 2.*np.pi, epsabs=1.e-01, epsrel=1.e-01)[0]
        x = np.linspace(0.,2.*np.pi,500)
        integral  = integrate.simps(Sigma_NFW((np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x))),z,M200,c200,cosmo),x,even='first')
        return integral/(2.*np.pi)

    
    
    integral = []
    for r in R:
        argumento = lambda x: S_RRs(x,r)*P_Roff(x,s_off)
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

    


def GAMMA_components_miss(R,z,M200,ellip,s_off = None, tau = 0.2,
                         c200 = None, P_Roff = Gamma, cosmo=cosmo, return_S2 = False):	
    
    '''
    Misscentred quadrupole components for NFW
    
    '''
  
    R200 = R200_NFW(M200,z,cosmo)
    
        
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)

    if not s_off:
        s_off = tau*R200

    def monopole(R):
        return Sigma_NFW(R,z,M200,c200,cosmo=cosmo)
    
    def monopole_off(R):
        return Sigma_NFW_miss([R],z,M200,s_off,tau,c200,P_Roff,cosmo)[0]
    
    def quadrupole(R):
        m0p = derivative(monopole,R,dx=1e-5)
        return m0p*R
        
    def S2_RRs(Rs,R):
        # F_Eq13
        #argumento = lambda x: monopole(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x)))
        #integral  = integrate.quad(argumento, 0, 2.*np.pi, epsabs=1.e-01, epsrel=1.e-01)[0]
        x = np.linspace(0.,2.*np.pi,500)
        integral  = integrate.simps(quadrupole(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x))),x,even='first')
        return integral/(2.*np.pi)

    def psi2_off(R):
        def arg(x):
            return (x**3)*monopole_off(x)
        argumento = lambda x: arg(x)
        integral = integrate.quad(argumento, 0, R, epsabs=1.e-01, epsrel=1.e-01)[0]
        return integral*(-2./(R**2))        

    S2off = []
    p2off = []
    S0off = Sigma_NFW_miss(R,z,M200,s_off,tau,c200,P_Roff,cosmo)

    for r in R:
        argumento = lambda x: S2_RRs(x,r)*P_Roff(x,s_off)
        S2off  += [integrate.quad(argumento, 0, np.inf, epsabs=1.e-02, epsrel=1.e-02)[0]]
        p2off  += [psi2_off(r)]

    
    '''
    vU_Eq10
    
    '''
    
    p2off = np.array(p2off)
    S2off = np.array(S2off)
    S0off = np.array(S0off)
    
    gt2off = ellip*((-6.*p2off/R**2) - 2.*S0off + S2off)
    gx2off = ellip*((-6.*p2off/R**2) - 4.*S0off)
    
    if return_S2:
        return [gt2off,gx2off,S2off]
    else:
        return [gt2off,gx2off]

def GAMMA_components_miss_unpack(minput):
	return GAMMA_components_miss(*minput)

def GAMMA_components_miss_parallel(r,z,M200,ellip,
                                   s_off = None, tau = 0.2,
                                   c200 = None, P_Roff = Gamma, 
                                   cosmo=cosmo,return_S2 = False,
                                   ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    
    slicer = int(round(len(r)/float(ncores), 0))
    slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
    slices = slices[(slices <= len(r))]
    r_splitted = np.split(r,slices)
    
    ncores = len(r_splitted)
    
    z      = [z]*ncores
    M200   = [M200]*ncores
    ellip  = [ellip]*ncores
    s_off  = [s_off]*ncores
    tau    = [tau]*ncores
    c200   = [c200]*ncores
    P_Roff = [P_Roff]*ncores
    cosmo  = [cosmo]*ncores
    rS2    = [return_S2]*ncores
        
    entrada = np.array([r_splitted,z,M200,ellip,s_off,tau,c200,P_Roff,cosmo,rS2]).T
    
    pool = Pool(processes=(ncores))
    salida=np.array(pool.map(GAMMA_components_miss_unpack, entrada))
    pool.terminate()

    GT_miss = np.array([])
    GX_miss = np.array([])
    S2_miss = np.array([])
    
    for s in salida:
        if return_S2:
            gt,gx,s2 = s
            GT_miss = np.append(GT_miss,gt)
            GX_miss = np.append(GX_miss,gx)
            S2_miss = np.append(S2_miss,s2)
        else:
            gt,gx = s
            GT_miss = np.append(GT_miss,gt)
            GX_miss = np.append(GX_miss,gx)

    if return_S2:
        return [GT_miss,GX_miss,S2_miss]
    else:
        return [GT_miss,GX_miss]
    
    
def Delta_Sigma_NFW_miss_unpack(minput):
	return Delta_Sigma_NFW_miss(*minput)

def Delta_Sigma_NFW_miss_parallel(r,z,M200,s_off = None, tau = 0.2,
                         c200 = None, P_Roff = Gamma, cosmo=cosmo,ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    
    slicer = int(round(len(r)/float(ncores), 0))
    slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
    slices = slices[(slices <= len(r))]
    r_splitted = np.split(r,slices)
    
    ncores = len(r_splitted)
    
    z      = [z]*ncores
    M200   = [M200]*ncores
    s_off  = [s_off]*ncores
    tau    = [tau]*ncores
    c200   = [c200]*ncores
    P_Roff = [P_Roff]*ncores
    cosmo  = [cosmo]*ncores
        
    entrada = np.array([r_splitted,z,M200,s_off,tau,c200,P_Roff,cosmo]).T
    
    pool = Pool(processes=(ncores))
    salida=np.array(pool.map(Delta_Sigma_NFW_miss_unpack, entrada))
    pool.terminate()

    DS_miss = np.array([])
    
    for s in salida:
        DS_miss = np.append(DS_miss,s)
            
    return DS_miss


def Sigma_NFW_miss_unpack(minput):
	return Sigma_NFW_miss(*minput)


def Sigma_NFW_miss_parallel(r,z,M200,s_off = None, tau = 0.2,
                         c200 = None, P_Roff = Gamma, cosmo=cosmo,ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    
    slicer = int(round(len(r)/float(ncores), 0))
    slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
    slices = slices[(slices <= len(r))]
    r_splitted = np.split(r,slices)
    
    ncores = len(r_splitted)
    
    z      = [z]*ncores
    M200   = [M200]*ncores
    s_off  = [s_off]*ncores
    tau    = [tau]*ncores
    c200   = [c200]*ncores
    P_Roff = [P_Roff]*ncores
    cosmo  = [cosmo]*ncores
        
    entrada = np.array([r_splitted,z,M200,s_off,tau,c200,P_Roff,cosmo]).T
    
    pool = Pool(processes=(ncores))
    salida=np.array(pool.map(Sigma_NFW_miss_unpack, entrada))
    pool.terminate()

    S_miss = np.array([])
    
    for s in salida:
        S_miss = np.append(S_miss,s)
            
    return S_miss

##############################################

def Rayleigh_elip(roff,theta,s_off,q):
    Roff = roff*np.sqrt(q*(np.cos(theta))**2 + (np.sin(theta))**2 / q)
    return (Roff/s_off**2)*np.exp(-0.5*(Roff/s_off)**2)

def Gauss_elip_xy(x,y,soffx,soffy):
    g = (1./(2*np.pi*soffx*soffy))*np.exp(-0.5*((x/soffx)**2 + (y/soffy)**2))
    return g

def Gauss_elip(r,t,soffx,soffy):
    g = Gauss_elip_xy(r*np.cos(t),r*np.sin(t),soffx,soffy)
    return g



def Sigma_NFW_miss_elip(R,z,M200,soffx = 0.1, soffy = 0.05,
                   c200 = None, P_Roff = Gauss_elip, 
                   qmiss = 0.8, cosmo=cosmo):
                       
    
    '''
    Misscentred density NFW profile
    F_Eq12
    '''
    

    
    R200 = R200_NFW(M200,z,cosmo)
    
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)



    def Smiss(Rs,theta,R):
        # F_Eq13
        Sm = Sigma_NFW((np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(theta))),z,M200,c200,cosmo)
        return Sm

    
    
    integral = []
    for r in R:
        argumento = lambda t,x: x*Smiss(x,t,r)*P_Roff(x,t,soffx,soffy)
        integral  += [(1./(2.*np.pi))*(integrate.dblquad(argumento, 0, np.inf, lambda x: 0, lambda x: 2.*np.pi, epsabs=1.e-02, epsrel=1.e-02)[0])]
        
    return integral


def Delta_Sigma_NFW_miss_elip(R,z,M200,soffx = 0.1, soffy = 0.05,
                         c200 = None, P_Roff = Gauss_elip, 
                         qmiss = 0.8, cosmo=cosmo):	
    
    '''
    Misscentred density contraste for NFW
    
    '''
  
    R200 = R200_NFW(M200,z,cosmo)
    
        
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)


    integral = []
    for r in R:
        argumento = lambda x: Sigma_NFW_miss_elip([x],z,M200,soffx,soffy,c200,P_Roff,qmiss,cosmo)[0]*x
        integral  += [integrate.quad(argumento, 0, r, epsabs=1.e-02, epsrel=1.e-02)[0]]

    DS_off    = (2./R**2)*integral - Sigma_NFW_miss_elip(R,z,M200,soffx,soffy,c200,P_Roff,qmiss,cosmo)

    return DS_off

    


def GAMMA_components_miss_elip(R,z,M200,ellip,soffx = 0.1, soffy = 0.05,
                         c200 = None, P_Roff = Gauss_elip, 
                         qmiss = 0.8, cosmo=cosmo, return_S2 = False):	
    
    '''
    Misscentred quadrupole components for NFW
    
    '''
  
    R200 = R200_NFW(M200,z,cosmo)
    
        
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)


    def monopole(R):
        return Sigma_NFW(R,z,M200,c200,cosmo=cosmo)
    
    def monopole_off(R):
        return Sigma_NFW_miss_elip([R],z,M200,soffx,soffy,c200,P_Roff,qmiss,cosmo)[0]
    
    def quadrupole(R):
        m0p = derivative(monopole,R,dx=1e-5)
        return m0p*R
        
    def S2miss(Rs,theta,R):
        # F_Eq13
        S2m = quadrupole(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(theta)))
        return S2m

    def psi2_off(R):
        def arg(x):
            return (x**3)*monopole_off(x)
        argumento = lambda x: arg(x)
        integral = integrate.quad(argumento, 0, R, epsabs=1.e-01, epsrel=1.e-01)[0]
        return integral*(-2./(R**2))        

    S2off = []
    p2off = []
    S0off = Sigma_NFW_miss_elip(R,z,M200,soffx,soffy,c200,P_Roff,qmiss,cosmo)

    for r in R:
        argumento = lambda t,x: x*S2miss(x,t,r)*P_Roff(x,t,soffx,soffy)
        S2off  += [(1./(2.*np.pi))*(integrate.dblquad(argumento, 0, np.inf, lambda x: 0, lambda x: 2.*np.pi, epsabs=1.e-02, epsrel=1.e-02)[0])]
        p2off  += [psi2_off(r)]

    
    '''
    vU_Eq10
    
    '''
    
    p2off = np.array(p2off)
    S2off = np.array(S2off)
    S0off = np.array(S0off)
    
    gt2off = ellip*((-6.*p2off/R**2) - 2.*S0off + S2off)
    gx2off = ellip*((-6.*p2off/R**2) - 4.*S0off)
    
    if return_S2:
        return [gt2off,gx2off,S2off]
    else:
        return [gt2off,gx2off]

def GAMMA_components_miss_elip_unpack(minput):
	return GAMMA_components_miss_elip(*minput)

def GAMMA_components_miss_elip_parallel(r,z,M200,ellip,
                                   soffx = 0.1, soffy = 0.05,
                                   c200 = None, P_Roff = Gauss_elip, 
                                   qmiss = 0.8, cosmo=cosmo,
                                   return_S2 = False,
                                   ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    
    slicer = int(round(len(r)/float(ncores), 0))
    slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
    slices = slices[(slices <= len(r))]
    r_splitted = np.split(r,slices)
    
    ncores = len(r_splitted)
    
    z      = [z]*ncores
    M200   = [M200]*ncores
    ellip  = [ellip]*ncores
    soffx  = [soffx]*ncores
    soffy  = [soffy]*ncores
    c200   = [c200]*ncores
    P_Roff = [P_Roff]*ncores
    cosmo  = [cosmo]*ncores
    rS2    = [return_S2]*ncores
    qmiss    = [qmiss]*ncores
        
    entrada = np.array([r_splitted,z,M200,ellip,soffx,soffy,c200,P_Roff,qmiss,cosmo,rS2]).T
    
    pool = Pool(processes=(ncores))
    salida=np.array(pool.map(GAMMA_components_miss_elip_unpack, entrada))
    pool.terminate()

    GT_miss = np.array([])
    GX_miss = np.array([])
    S2_miss = np.array([])
    
    for s in salida:
        if return_S2:
            gt,gx,s2 = s
            GT_miss = np.append(GT_miss,gt)
            GX_miss = np.append(GX_miss,gx)
            S2_miss = np.append(S2_miss,s2)
        else:
            gt,gx = s
            GT_miss = np.append(GT_miss,gt)
            GX_miss = np.append(GX_miss,gx)

    if return_S2:
        return [GT_miss,GX_miss,S2_miss]
    else:
        return [GT_miss,GX_miss]
    
    
def Delta_Sigma_NFW_miss_elip_unpack(minput):
	return Delta_Sigma_NFW_miss_elip(*minput)

def Delta_Sigma_NFW_miss_elip_parallel(r,z,M200,soffx = 0.1, soffy = 0.05,
                         c200 = None, P_Roff = Gauss_elip, 
                         qmiss = 0.8,cosmo=cosmo,ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    
    slicer = int(round(len(r)/float(ncores), 0))
    slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
    slices = slices[(slices <= len(r))]
    r_splitted = np.split(r,slices)
    
    ncores = len(r_splitted)
    
    z      = [z]*ncores
    M200   = [M200]*ncores
    soffx  = [soffx]*ncores
    soffy  = [soffy]*ncores
    c200   = [c200]*ncores
    P_Roff = [P_Roff]*ncores
    cosmo  = [cosmo]*ncores
    qmiss  = [qmiss]*ncores
        
    entrada = np.array([r_splitted,z,M200,soffx,soffy,c200,P_Roff,qmiss,cosmo]).T
    
    pool = Pool(processes=(ncores))
    salida=np.array(pool.map(Delta_Sigma_NFW_miss_elip_unpack, entrada))
    pool.terminate()

    DS_miss = np.array([])
    
    for s in salida:
        DS_miss = np.append(DS_miss,s)
            
    return DS_miss


def Sigma_NFW_miss_elip_unpack(minput):
	return Sigma_NFW_miss_elip(*minput)


def Sigma_NFW_miss_elip_parallel(r,z,M200,soffx = 0.1, soffy = 0.05,
                         c200 = None, P_Roff = Gauss_elip, 
                         qmiss = 0.8,cosmo=cosmo,ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    
    slicer = int(round(len(r)/float(ncores), 0))
    slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
    slices = slices[(slices <= len(r))]
    r_splitted = np.split(r,slices)
    
    ncores = len(r_splitted)
    
    z      = [z]*ncores
    M200   = [M200]*ncores
    soffx  = [soffx]*ncores
    soffy  = [soffy]*ncores
    c200   = [c200]*ncores
    P_Roff = [P_Roff]*ncores
    cosmo  = [cosmo]*ncores
    qmiss  = [qmiss]*ncores
        
    entrada = np.array([r_splitted,z,M200,soffx,soffy,c200,P_Roff,qmiss,cosmo]).T
    
    pool = Pool(processes=(ncores))
    salida=np.array(pool.map(Sigma_NFW_miss_elip_unpack, entrada))
    pool.terminate()

    S_miss = np.array([])
    
    for s in salida:
        S_miss = np.append(S_miss,s)
            
    return S_miss

