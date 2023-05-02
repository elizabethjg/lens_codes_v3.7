import sys, os
import numpy as np
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
params = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}

def g(x,disp): 
    return (1./(np.sqrt(2*np.pi)*disp))*np.exp(-0.5*(x/disp)**2)

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
    
def R200_NFW(M200,z,cosmo):	
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
    # from colossus.cosmology import cosmology  
    # cosmology.addCosmology('MyCosmo', cosmo_params)
    # cosmo = cosmology.setCosmology('MyCosmo')
    # roc_mpc = cosmo.rho_c(z)*(Msun*(1.e3**3))
    
    return ((M200*(3.0*Msun))/(800.0*np.pi*roc_mpc))**(1./3.)
    

def M200_NFW(R200,z,cosmo):	
    
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
    
    # from colossus.cosmology import cosmology  
    # cosmology.addCosmology('MyCosmo', cosmo_params)
    # cosmo = cosmology.setCosmology('MyCosmo')
    # roc_mpc = cosmo.rho_c(z)/(1.e3**3)
    
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


    # from colossus.cosmology import cosmology  
    # cosmology.addCosmology('MyCosmo', cosmo_params)
    # cosmo = cosmology.setCosmology('MyCosmo')
    # roc_mpc = cosmo.rho_c(z)*(1.e3**3)

    R200 = R200_NFW(M200,z,cosmo)
    
    roc_mpc = cosmo.critical_density(z).to(u.Msun/(u.Mpc)**3).value
    
    if not c200:
        c200 = c200_duffy(M200*cosmo.H0/100.,z)
    
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
    
    jota[m4] = np.interp(x[m4],[x1,x2],[j1,j2])
                
    rs_m = R200/c200
    kapak = (2.*rs_m*deltac*roc_mpc)
    # Units M_sun/Mpc2
    return kapak*jota
    
def rho_NFW(R,z,M200,c200 = None,cosmo=cosmo):			
    '''
    Projected density for NFW
    
    '''		
    
    if not isinstance(R, (np.ndarray)):
        R = np.array([R])


    m = R == 0.
    R[m] = 1.e-8  

    
    R200 = R200_NFW(M200,z,cosmo)
    
    roc_mpc = cosmo.critical_density(z).to(u.Msun/(u.Mpc)**3).value
    
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)
    
    ####################################################
    
    deltac=(200./3.)*( (c200**3) / ( np.log(1.+c200)- (c200/(1+c200)) ))
    
    x=(R*c200)/R200

    ro = (deltac * roc_mpc)/(x * ((1 + x)**2))
    # Units M_sun/Mpc3
    return ro

def quadrupole(R,z,M200,c200 = None,cosmo=cosmo):
   
    '''
    Quadrupole term defined as (d(Sigma)/dr)*r
    
    '''
    
    def monopole(R):
        return Sigma_NFW(R,z,M200,c200,cosmo=cosmo)/1.e6**2
    
    m0p = derivative(monopole,R,dx=1e-5)
    q   =  -1.*m0p*R

    return q

def rho_NFW_2h(R,z,M200,c200,
               cosmo_params=params,
               terms='1h',limint=500e3):
    
    '''
    R - radii [Mpc]
    z - redshift 
    M200 - M200c [M_sun]
    
    3D NFW density from colossus
    units Msun/pc3
    '''
    
    from colossus.lss import bias
    from colossus.halo import profile_nfw
    from colossus.halo import profile_outer
    from colossus.cosmology import cosmology  
    cosmology.addCosmology('MyCosmo', cosmo_params)
    cosmo = cosmology.setCosmology('MyCosmo')

    b = bias.haloBias(M200, model = 'tinker10', z = z, mdef = '200c')
    
    outer_term = profile_outer.OuterTermCorrelationFunction(z = z, bias = b)
    pNFW = profile_nfw.NFWProfile(M = M200, mdef = '200c', z = z, c = c200, outer_terms = [outer_term])
    
    # Outer term integrated up to 100Mpc (Luo et al. 2017, Niemic et al 2017)
    
    if terms == '1h':
        rho_in  = pNFW.density(R*1.e3)
        rho = rho_in
    elif terms == '2h':
        rho_out = pNFW.densityOuter(R*1.e3, interpolate=False, accuracy=0.01, max_r_integrate=limint)
        rho = rho_out
    elif terms == '1h+2h':
        rho_in  = pNFW.density(R*1.e3)
        rho_out = pNFW.densityOuter(R*1.e3, interpolate=False, accuracy=0.01, max_r_integrate=limint)
        rho = rho_in + rho_out
    
    return rho/(1.e3**3)


def Sigma_NFW_2h(R,z,M200,c200,
                 cosmo_params=params,
                 terms='1h',limint=500e3):

    '''
    projected NFW density from colossus
    units Msun/pc2
    '''
    
    from colossus.lss import bias
    from colossus.halo import profile_nfw
    from colossus.halo import profile_outer
    from colossus.cosmology import cosmology  
    cosmology.addCosmology('MyCosmo', cosmo_params)
    cosmo = cosmology.setCosmology('MyCosmo')

    b = bias.haloBias(M200, model = 'tinker10', z = z, mdef = '200c')
    
    outer_term = profile_outer.OuterTermCorrelationFunction(z = z, bias = b)
    pNFW = profile_nfw.NFWProfile(M = M200, mdef = '200c', z = z, c = c200, outer_terms = [outer_term])
    
    # Outer term integrated up to 100Mpc (Luo et al. 2017, Niemic et al 2017)
    
    if terms == '1h':
        s_in  = pNFW.surfaceDensityInner(R*1.e3)
        s = s_in
    elif terms == '2h':
        s_out = pNFW.surfaceDensityOuter(R*1.e3, interpolate=False, accuracy=0.01, max_r_integrate=limint)
        s = s_out
    elif terms == '1h+2h':
        s_in  = pNFW.surfaceDensityInner(R*1.e3)
        s_out = pNFW.surfaceDensityOuter(R*1.e3, interpolate=False, accuracy=0.01, max_r_integrate=limint)
        s = s_in + s_out
    
    return s/(1.e3**2)

def Sigma_Ein_2h(R,z,M200,c200,
                 alpha,cosmo_params=params,
                 terms='1h',limint=500e3):
    
    '''
    projected Ein density from colossus
    units Msun/pc2
    '''
    
    
    from colossus.lss import bias
    from colossus.halo import profile_einasto
    from colossus.halo import profile_outer
    from colossus.cosmology import cosmology  
    cosmology.addCosmology('MyCosmo', cosmo_params)
    cosmo = cosmology.setCosmology('MyCosmo')

    b = bias.haloBias(M200, model = 'tinker10', z = z, mdef = '200c')
    
    outer_term = profile_outer.OuterTermCorrelationFunction(z = z, bias = b)
    p = profile_einasto.EinastoProfile(M = M200, mdef = '200c', z = z, c = c200, alpha = alpha, outer_terms = [outer_term])
    
    # Outer term integrated up to 100Mpc (Luo et al. 2017, Niemic et al 2017)
    
    if terms == '1h':
        s_in  = p.surfaceDensityInner(R*1.e3)
        s = s_in
    elif terms == '2h':
        s_out = p.surfaceDensityOuter(R*1.e3, interpolate=False, accuracy=0.01, max_r_integrate=limint)
        s = s_out
    elif terms == '1h+2h':
        s_in  = p.surfaceDensityInner(R*1.e3)
        s_out = p.surfaceDensityOuter(R*1.e3, interpolate=False, accuracy=0.01, max_r_integrate=limint)
        s = s_in + s_out
    
    return s/(1.e3**2)

def Delta_Sigma_NFW_2h(R,z,M200,c200,
                    cosmo_params=params,
                    terms='1h',limint=100e3):
    
    '''
    NFW contrast density from colossus
    units Msun/pc2
    '''
    
    
    from colossus.lss import bias
    from colossus.halo import profile_nfw
    from colossus.halo import profile_outer
    from colossus.cosmology import cosmology  
    cosmology.addCosmology('MyCosmo', cosmo_params)
    cosmo = cosmology.setCosmology('MyCosmo')

    b = bias.haloBias(M200, model = 'tinker10', z = z, mdef = '200c')
    
    outer_term = profile_outer.OuterTermCorrelationFunction(z = z, bias = b)
    pNFW = profile_nfw.NFWProfile(M = M200, mdef = '200c', z = z, c = c200, outer_terms = [outer_term])    
    
    # Outer term integrated up to 50Mpc (Luo et al. 2017, Niemic et al 2017)
    if terms == '1h':
        ds_in  = pNFW.deltaSigmaInner(R*1.e3)
        ds = ds_in
    elif terms == '2h':
        ds_out = pNFW.deltaSigmaOuter(R*1.e3, interpolate=False, interpolate_surface_density=False, accuracy=0.01, max_r_integrate=limint)
        ds = ds_out
    elif terms == '1h+2h':
        ds_in  = pNFW.deltaSigmaInner(R*1.e3)
        ds_out = pNFW.deltaSigmaOuter(R*1.e3, interpolate=False, interpolate_surface_density=False, accuracy=0.01, max_r_integrate=limint)
        ds = ds_in + ds_out
    
    return ds/(1.e3**2)
    
def Delta_Sigma_NFW_2h_unpack(minput):
	return Delta_Sigma_NFW_2h(*minput)

def Delta_Sigma_NFW_2h_parallel(r,z,M200,c200,
                    cosmo_params=params,
                    terms='1h',limint=100e3,ncores=10):
    
    if ncores > len(r):
        ncores = len(r)
            
    r_splitted = np.array_split(r,ncores)
    
    entrada = []
    for j in range(ncores):
        entrada += [[r_splitted[j],z,M200,c200,cosmo_params,terms,limint]]
        
    pool   = Pool(processes=(ncores))
    salida = pool.map(Delta_Sigma_NFW_2h_unpack, entrada)
    pool.terminate()

    DS_2h = np.array([])
    
    for s in salida:
        DS_2h = np.append(DS_2h,s)
            
    return DS_2h


def Delta_Sigma_Ein_2h(R,z,M200,c200,
                       alpha,cosmo_params=params,
                       terms='1h',limint=100e3):
    
    '''
    Einasto contrast density from colossus
    units Msun/pc2
    '''    
    
    from colossus.lss import bias
    from colossus.halo import profile_einasto
    from colossus.halo import profile_outer
    from colossus.cosmology import cosmology  
    cosmology.addCosmology('MyCosmo', cosmo_params)
    cosmo = cosmology.setCosmology('MyCosmo')

    b = bias.haloBias(M200, model = 'tinker10', z = z, mdef = '200c')
    
    outer_term = profile_outer.OuterTermCorrelationFunction(z = z, bias = b)
    p = profile_einasto.EinastoProfile(M = M200, mdef = '200c', z = z, c = c200, alpha = alpha, outer_terms = [outer_term])    
    
    # Outer term integrated up to 50Mpc (Luo et al. 2017, Niemic et al 2017)
    if terms == '1h':
        ds_in  = p.deltaSigmaInner(R*1.e3)
        ds = ds_in
    elif terms == '2h':
        ds_out = p.deltaSigmaOuter(R*1.e3, interpolate=False, interpolate_surface_density=False, accuracy=0.01, max_r_integrate=limint)
        ds = ds_out
    elif terms == '1h+2h':
        ds_in  = p.deltaSigmaInner(R*1.e3)
        ds_out = p.deltaSigmaOuter(R*1.e3, interpolate=False, interpolate_surface_density=False, accuracy=0.01, max_r_integrate=limint)
        ds = ds_in + ds_out
    
    return ds/(1.e3**2)

def Delta_Sigma_Ein_2h_unpack(minput):
	return Delta_Sigma_Ein_2h(*minput)

def Delta_Sigma_Ein_2h_parallel(r,z,M200,c200,
                       alpha,cosmo_params=params,
                       terms='1h',limint=100e3,ncores=10):
    
    if ncores > len(r):
        ncores = len(r)
    
    r_splitted = np.array_split(r,ncores)
    
    entrada = []
    for j in range(ncores):
        entrada += [[r_splitted[j],z,M200,c200,alpha,cosmo_params,terms,limint]]
            
    pool = Pool(processes=(ncores))
    salida = pool.map(Delta_Sigma_Ein_2h_unpack, entrada)
    pool.terminate()

    DS_2h = np.array([])
    
    for s in salida:
        DS_2h = np.append(DS_2h,s)
            
    return DS_2h


def S2_quadrupole(R,z,M200,c200 = None,
                  terms='1h',cosmo_params=params,
                  pname='NFW',alpha=0.3,limint=500e3):

    '''
    Quadrupole term defined as -1.*(d(Sigma)/dr)*r
    
    '''
    
    def monopole(R):
        if pname == 'NFW':
            return Sigma_NFW_2h(R,z,M200,c200,terms=terms,cosmo_params=cosmo_params,limint=limint)
        elif pname == 'Einasto':
            return Sigma_Ein_2h(R,z,M200,c200,alpha,terms=terms,cosmo_params=cosmo_params,limint=limint)
    
    m0p = derivative(monopole,R,dx=1e-4)
    q   =  m0p*R

    return -1.*q

def GAMMA_components(R,z,ellip,M200,c200 = None,terms='1h',cosmo_params=params,pname='NFW',alpha=0.3):
    
    def monopole(R):
        if pname == 'NFW':
            return Sigma_NFW_2h(R,z,M200,c200,terms=terms,cosmo_params=cosmo_params)
        elif pname == 'Einasto':
            return Sigma_Ein_2h(R,z,M200,c200,alpha,terms=terms,cosmo_params=cosmo_params)

    '''
    Quadrupole term defined as (d(Sigma)/dr)*r
    
    '''
    
    m0p = derivative(monopole,R,dx=1e-4)
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
    
    gt = ellip*((-6*p2/R**2) - 2.*m + q)
    gx = ellip*((-6*p2/R**2) - 4.*m)
    
    return [gt,gx]
    
def GAMMA_components_unpack(minput):
	return GAMMA_components(*minput)

def GAMMA_components_parallel(r,z,ellip,M200,
                              c200 = None,terms='1h',
                              cosmo_params=params,pname='NFW',
                              alpha=0.3,ncores=4):	

    if ncores > len(r):
        ncores = len(r)
    
    r_splitted = np.array_split(r,ncores)
    
    entrada = []
    for j in range(ncores):
        entrada += [[r_splitted[j],z,ellip,M200,c200,terms,cosmo_params,pname,alpha]]
    
    pool   = Pool(processes=(ncores))
    salida = pool.map(GAMMA_components_unpack, entrada)
    pool.terminate()

    gt = np.array([])
    gx = np.array([])
    
    for s in salida:
        GT, GX = s
        gt = np.append(gt,GT)
        gx = np.append(gx,GT)
            
    return [gt,gx]

### MISCENTRED

def Sigma_NFW_miss(R,z,M200,s_off = None, tau = 0.2,
                   c200 = None, P_Roff = Rayleigh, cosmo_params=params):
                       
    
    '''
    Misscentred density NFW profile
    F_Eq12
    '''
    


    if not s_off:
        s_off = tau*R200


    def SNFW(r):
        # return Sigma_NFW_2h(r,z,M200,c200,cosmo_params=params,terms='1h')
        return Sigma_NFW(r,z,M200,c200)/1.e12

    def S_RRs(Rs,R):
        # F_Eq13
        #argumento = lambda x: monopole(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x)))
        #integral  = integrate.quad(argumento, 0, 2.*np.pi, epsabs=1.e-01, epsrel=1.e-01)[0]
        x = np.linspace(0.,2.*np.pi,500)
        integral  = integrate.simps(SNFW(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x))),x,even='first')
        return integral/(2.*np.pi)

    
    
    integral = []
    for r in R:
        argumento = lambda x: S_RRs(x,r)*P_Roff(x,s_off)
        integral  += [integrate.quad(argumento, 0, np.inf, epsabs=1.e-02, epsrel=1.e-02)[0]]
        
    return integral


def Delta_Sigma_NFW_miss(R,z,M200,s_off = None, tau = 0.2,
                         c200 = None, P_Roff = Rayleigh, cosmo_params=params):	
    
    '''
    Misscentred density contraste for NFW
    
    '''
        
    if not c200:
        c200 = c200_duffy(M200*cosmo.h,z)

    if not s_off:
        R200 = R200_NFW(M200,z,params)
        s_off = tau*R200


    integral = []
    for r in R:
        argumento = lambda x: Sigma_NFW_miss([x],z,M200,s_off,tau,c200,P_Roff,cosmo_params)[0]*x
        integral  += [integrate.quad(argumento, 0, r, epsabs=1.e-02, epsrel=1.e-02)[0]]

    DS_off    = (2./R**2)*integral - Sigma_NFW_miss(R,z,M200,s_off,tau,c200,P_Roff,cosmo_params)

    return DS_off

    
def Delta_Sigma_NFW_miss_unpack(minput):
	return Delta_Sigma_NFW_miss(*minput)

def Delta_Sigma_NFW_miss_parallel(r,z,M200,s_off = None, tau = 0.2,
                         c200 = None, P_Roff = Rayleigh, cosmo_params=params,ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    r_splitted = np.array_split(r,ncores)
    
    entrada = []
    for j in range(ncores):
        entrada += [[r_splitted[j],z,M200,s_off,tau,c200,P_Roff,cosmo_params]]
    
    pool = Pool(processes=(ncores))
    salida = pool.map(Delta_Sigma_NFW_miss_unpack, entrada)
    pool.terminate()

    DS_miss = np.array([])
    
    for s in salida:
        DS_miss = np.append(DS_miss,s)
            
    return DS_miss


def Sigma_NFW_miss_unpack(minput):
	return Sigma_NFW_miss(*minput)


def Sigma_NFW_miss_parallel(r,z,M200,s_off = None, tau = 0.2,
                         c200 = None, P_Roff = Rayleigh, cosmo_params=params,ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    
    r_splitted = np.array_split(r,ncores)
    
    entrada = []
    for j in range(ncores):
        entrada += [[r_splitted[j],z,M200,s_off,tau,c200,P_Roff,cosmo_params]]
    
    pool = Pool(processes=(ncores))
    salida = pool.map(Sigma_NFW_miss_unpack, entrada)
    pool.terminate()

    S_miss = np.array([])
    
    for s in salida:
        S_miss = np.append(S_miss,s)
            
    return S_miss
    
    
def DELTA_SIGMA_full(R,z,M200,c200,
                     s_off = None, tau = 0.2,
                     pcc = 0.7, P_Roff = Rayleigh, 
                     cosmo_params=params):
    
    
    DS_miss = Delta_Sigma_NFW_miss(R,z,M200,s_off,tau,c200,P_Roff)
    DS_1h   = Delta_Sigma_NFW_2h(R,z,M200,c200,params,'1h')
    DS_2h   = Delta_Sigma_NFW_2h(R,z,M200,c200,params,'2h')
        
    return pcc*(DS_1h) + (1.-pcc)*DS_miss + DS_2h


def DELTA_SIGMA_full_unpack(minput):
	return DELTA_SIGMA_full(*minput)

def DELTA_SIGMA_full_parallel(r,z,M200,c200,
                                  s_off = None, tau = 0.2,
                                  pcc = 0.7, P_Roff = Rayleigh, 
                                  cosmo_params=params,ncores=4):	
	
    if ncores > len(r):
        ncores = len(r)
    
    r_splitted = np.array_split(r,ncores)
    
    entrada = []
    for j in range(ncores):
        entrada += [[r_splitted[j],z,M200,c200,s_off,tau,pcc,P_Roff,cosmo_params]]
    
    pool = Pool(processes=(ncores))
    salida = pool.map(DELTA_SIGMA_full_unpack, entrada)
    pool.terminate()

    DS_full = np.array([])
    
    for s in salida:
        DS_full = np.append(DS_full,s)
            
    return DS_full

# t1 = time.time()
# ds   = DELTA_SIGMA_full_parallel(R,0.2,1.e14,3.5,s_off = 0.15, pcc = 0.7, P_Roff = Gamma, cosmo_params=params,ncores=20)
# print(time.time()-t1)
