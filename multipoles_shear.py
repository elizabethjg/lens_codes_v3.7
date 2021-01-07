import sys, os
import numpy as np
from pylab import *
from astropy.cosmology import LambdaCDM
from scipy.misc import derivative
from scipy import integrate
from profiles_fit import *
from multiprocessing import Pool
from multiprocessing import Process
import time


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
#parameters

cvel = 299792458;   # Speed of light (m.s-1)
G    = 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16; # 1 pc (m)
Msun = 1.989e30 # Solar mass (kg)

def multipole_shear(r,M200=1.e14,ellip=0.25,z=0.2,h=0.7,
					misscentred=False,s_off=0.4,
					components = ['t0','t','tcos','xsin'],
					verbose=True,Yanmiss = False):

	'''
	Equations from van Uitert (vU, arXiv:1610.04226) for the 
	multipole expansion and from Ford et al. (F,2015) for
	the misscentring

	INPUTS:
	r               [Mpc] - Float or numpy_array
	                Distance to the centre of 
	                the gravitational potential
	
	OPTIONAL:
	M200            [M_sun] - Float/ Cluster Mass
	ellip           Float / Halo ellipticity defined as
	                (1-q)/(1+q) where q is the semi-axis
	                ratio (q < 1)
	z               Float/ Lens redshift
	zs              Float/ Source redshift
	h               Float/ Cosmological quantity
	misscentred     Float/ If True computes de misscentred quantities
					The misscentred is considered only in the 
					x - axis
	s_off           [Mpc h-1] Float/ sigma_offset width of the distribution 
	                of cluster offsets (F_Eq11)
	components      List of misscentred components that are going
	                to be computed:
	                't0'   -> Gt0_off
	                't'    -> Gt_off
	                'tcos' -> Gt_off_cos
	                'xsin' -> Gx_off_sin
	verbose         Print time computing
	Yanmiss         if True use Eq 4 to model the miss of Yan et al. 2020
	
	OUTPUT:
	output          Dictionary (the shear is scales with the
	                critical density Gamma_i = Sigma_cr*gamma_i
	                ---------- vU_Eq18 ---------
	                Gt0: Shear monopole
	                Gt2: Tangential component for the quadrupole
	                Gx2: Cross component for the quadrupole
	                --------- Misscentred terms --------
	                Gt0_off: (ellip = 0) F_Eq14
	                Gt_off: Integrated tangential component (vU_Eq17)
	                for considering kappa_offset
	                Gt_off_cos: Integrated tangential component times
	                cos(2theta) (vU_Eq17) for considering kappa_offset
	                Gx_off_sin: Integrated cross component times
	                sin(2theta) (vU_Eq17) for considering kappa_offset
	'''
	
	if not verbose:
		blockPrint()

	# Check if r is float or numpy array
	if not isinstance(r, (np.ndarray)):
		r = np.array([r])

	# Compute cosmological parameters
	cosmo = LambdaCDM(H0=h*100, Om0=0.3, Ode0=0.7)
	H        = cosmo.H(z).value/(1.0e3*pc) #H at z_pair s-1 
	roc      = (3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
	roc_mpc  = roc*((pc*1.0e6)**3.0)
	
	# Compute R_200
	R200 = r200_nfw(M200,roc_mpc)

	# Scaling sigma_off
	s_off = s_off/h	
	
	#print '##################'
	#print '      CENTRED     '
	#print '##################'
	
	def Delta_Sigma(R):
		'''
		Density contraste for NFW
		
		'''
		
		#calculo de c usando la relacion de Duffy et al 2008
		
		M=((800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun))*h
		c=5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)

		####################################################
		
		deltac=(200./3.)*( (c**3) / ( np.log(1.+c)- (c/(1+c)) ))
		x=np.round((R*c)/R200,12)
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
	
	
			
		rs_m=(R200*1.e6*pc)/c
		kapak=((2.*rs_m*deltac*roc_mpc)*(pc**2/Msun))/((pc*1.0e6)**3.0)
		return kapak*jota
	
	
	def monopole(R):		
		'''
		Projected density for NFW
		
		'''		
		if not isinstance(R, (np.ndarray)):
			R = np.array([R])
		
		m = R == 0.
		R[m] = 1.e-8
		
		#calculo de c usando la relacion de Duffy et al 2008
		
		M=((800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun))*h
		c=5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)

		####################################################
		
		deltac=(200./3.)*( (c**3) / ( np.log(1.+c)- (c/(1+c)) ))

		x=(R*c)/R200
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
					
		rs_m=(R200*1.e6*pc)/c
		kapak=((2.*rs_m*deltac*roc_mpc)*(pc**2/Msun))/((pc*1.0e6)**3.0)
		return kapak*jota

	def quadrupole(R):
		'''
		Quadrupole term defined as (d(Sigma)/dr)*r
		
		'''		
		m0p = derivative(monopole,R,dx=1e-5)
		return m0p*R

	def psi2(R):
		'''
		vU_Eq10
		
		'''
		argumento = lambda x: (x**3)*monopole(x)
		integral = integrate.quad(argumento, 0, R, epsabs=1.e-01, epsrel=1.e-01)[0]
		return integral*(-2./(R**2))
	
	vecpsi2 = np.vectorize(psi2)
		
	#print '##################'
	#print '    MISCENTRED    '
	#print '##################'

	def P_Roff(Roff):
		if Yanmiss:
			Poff = (Roff/s_off**2)*np.exp(-1.*(Roff/s_off))
		else:
			# F_Eq11
			Poff = abs((Roff/s_off**2)*np.exp(-0.5*(Roff/s_off)**2))
		return Poff
	
	def monopole_off(R,theta):
		'''
		F_Eq12
		'''
		
		def moff(x):
			return monopole(np.sqrt(R**2+x**2-2.*x*R*np.cos(theta)))*P_Roff(x)*0.5
		argumento = lambda x: moff(x)
		# integral1  = integrate.quad(argumento, -1.*np.inf, 0, epsabs=1.e-01, epsrel=1.e-01)[0]
		integral1  = integrate.quad(argumento, -1.*np.inf, -100., epsabs=1.e-01, epsrel=1.e-01)[0]
		# integral2  = integrate.quad(argumento, 0., R, epsabs=1.e-01, epsrel=1.e-01)[0]
		x = np.linspace(-100.,100.,5000)
		integral2  = integrate.simps(moff(x),x,even='first')
		# integral3  = integrate.quad(argumento, R, np.inf, epsabs=1.e-01, epsrel=1.e-01)[0]
		integral3  = integrate.quad(argumento, 100., np.inf, epsabs=1.e-01, epsrel=1.e-01)[0]
		return integral1 + integral2 + integral3
	vec_moff = np.vectorize(monopole_off)
	
	
	
	def Delta_Sigma_off(R,theta):
		'''
		F_Eq14
		'''
		
		argumento = lambda x: monopole_off(x,theta)*x
		integral  = integrate.quad(argumento, 0, R, epsabs=1.e-01, epsrel=1.e-01)[0]
		DS_off    = (2./R**2)*integral - monopole_off(R,theta)
		return DS_off


	def monopole_off0(R):
		'''
		F_Eq12
		
		'''						
		def DS_RRs(Rs,R):
			# F_Eq13
			#argumento = lambda x: monopole(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x)))
			#integral  = integrate.quad(argumento, 0, 2.*np.pi, epsabs=1.e-01, epsrel=1.e-01)[0]
			x = np.linspace(0.,2.*np.pi,500)
			integral  = integrate.simps(monopole(np.sqrt(R**2+Rs**2-2.*Rs*R*np.cos(x))),x,even='first')
			return integral/(2.*np.pi)

		argumento = lambda x: DS_RRs(x,R)*P_Roff(x)
		integral  = integrate.quad(argumento, 0, np.inf, epsabs=1.e-02, epsrel=1.e-02)[0]
		return integral

	def Delta_Sigma_off0(R):
		'''
		F_Eq14
		
		'''						
		argumento = lambda x: monopole_off0(x)*x
		integral  = integrate.quad(argumento, 0, R, epsabs=1.e-02, epsrel=1.e-02)[0]
		DS_off    = (2./R**2)*integral - monopole_off0(R)
		return DS_off
	
	vec_DSoff0 = np.vectorize(Delta_Sigma_off0)
	
	
	def quadrupole_off(R,theta):
		def q_off(roff):
			rp = np.sqrt(R**2+roff**2-2*roff*R*np.cos(theta))
			return quadrupole(rp)*P_Roff(roff)*0.5
		argumento = lambda x: q_off(x)
		# integral10  = integrate.quad(argumento, -1.*np.inf, 0, epsabs=1.e-01, epsrel=1.e-01)[0]
		integral10  = integrate.quad(argumento, -1.*np.inf, -100., epsabs=1.e-01, epsrel=1.e-01)[0]
		# integral20  = integrate.quad(argumento, 0., R, epsabs=1.e-01, epsrel=1.e-01)[0]
		x = np.linspace(-100.,100.,5000)
		integral20  = integrate.simps(q_off(x),x,even='first')
		# integral30  = integrate.quad(argumento, R, np.inf, epsabs=1.e-01, epsrel=1.e-01)[0]
		integral30  = integrate.quad(argumento, 100., np.inf, epsabs=1.e-01, epsrel=1.e-01)[0]

		return integral10 + integral20 + integral30	
	vec_qoff = np.vectorize(quadrupole_off)
	
	def psi2_off(R,theta):
		def arg(x):
			return (x**3)*monopole_off(x,theta)
		argumento = lambda x: arg(x)
		integral = integrate.quad(argumento, 0, R, epsabs=1.e-01, epsrel=1.e-01)[0]
		return integral*(-2./(R**2))
		
	vecpsi2_off = np.vectorize(psi2_off)	

	def quantities_centred(r):
		
		gt0 = Delta_Sigma(r)
		monopole_r = monopole(r)
		quadrupole_r = quadrupole(r)
		print('computing psi2 centred')
		psi2_r = vecpsi2(r)
				
		return gt0,monopole_r,quadrupole_r,psi2_r
		
	def quantities_misscentred(r):
		print('computing misscentred profile')
		gamma_t0_off = []
		gamma_t_off0 = []
		gamma_t_off = []
		gamma_x_off = []
		for R in r:
			
			if 't0' in components:
				print('computing DS_t0_off')
				t1 = time.time()
				DSoff = Delta_Sigma_off0(R)
				gamma_t0_off = np.append(gamma_t0_off,DSoff)
				t2 = time.time()
				print((t2-t1)/60.)

			def DS_t_off(theta):
				gamma_t0 = []
				gamma_t2 = []
				for t in theta:
					gamma_t0 = np.append(gamma_t0,Delta_Sigma_off(R,t))
					gamma_t2 = np.append(gamma_t2,((-6*psi2_off(R,t)/R**2) 
								- 2.*monopole_off(R,t) 
								+ quadrupole_off(R,t))*np.cos(2.*t))
				return gamma_t0 + ellip*gamma_t2


			if 't' in components:
				print('computing DS_t_off')
				t1 = time.time()
				if ellip == 0.:
					DSoff = Delta_Sigma_off0(R)
					gamma_t_off0 = np.append(gamma_t_off0,DSoff)
				else:
					x = np.linspace(0.,2.*np.pi,100)
					integral  = integrate.simps(DS_t_off(x),x,even='first')
					gamma_t_off0 = np.append(gamma_t_off0,integral/(2.*np.pi))
				t2 = time.time()
				print(R,(t2-t1)/60.)

			if 'tcos' in components:	
				print('computing DS_t_off_cos')
				t1 = time.time()
				x = np.linspace(0.,2.*np.pi,100)
				integral  = integrate.simps(DS_t_off(x)*np.cos(2.*x),x,even='first')
				gamma_t_off = np.append(gamma_t_off,integral/np.pi)
				t2 = time.time()
				print('tcos',R,(t2-t1)/60.)

			if 'xsin' in components:
				print('computing DS_x_off_sin')
				t1 = time.time()
				def DS_x_off(theta):
					gamma_x2 = ((-6*psi2_off(R,theta)/R**2) 
								- 4.*monopole_off(R,theta))
					return ellip*gamma_x2*np.sin(2.*theta)
				argumento = lambda x: DS_x_off(x)*np.sin(2.*x)
				integral  = integrate.quad(argumento, 0, 2.*np.pi,points=[np.pi], epsabs=1.e-01, epsrel=1.e-01)[0]
				gamma_x_off = np.append(gamma_x_off,integral/np.pi)	
				t2 = time.time()
				print('xsin',R,(t2-t1)/60.)
					
		return gamma_t0_off, gamma_t_off0, gamma_t_off, gamma_x_off
		
	
	gt0,m,q,p2 = quantities_centred(r)
	
	'''
	vU_Eq18
	
	'''
	gt2 = ellip*((-6*p2/r**2) - 2.*m + q)
	gx2 = ellip*((-6*p2/r**2) - 4.*m)
	
	output = {'Gt0':gt0,'Gt2':gt2,'Gx2':gx2}
	
	if misscentred:
		gt0_off, gt_off0, gt_off, gx_off = quantities_misscentred(r)	
		output.update({'Gt0_off':gt0_off,'Gt_off':gt_off0,'Gt_off_cos':gt_off,'Gx_off_sin':gx_off})

	enablePrint()
		
	return output

# '''

def multipole_shear_unpack(minput):
	return multipole_shear(*minput)
	
def multipole_shear_parallel(r,M200=1.e14,ellip=0.25,z=0.2,
							 h=0.7,misscentred=False,
							 s_off=0.4,components = ['t0','t','tcos','xsin'],
							 verbose = True, Yanmiss = False, ncores=2):
	
	if ncores > len(r):
		ncores = len(r)
	
	
	slicer = int(round(len(r)/float(ncores), 0))
	slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
	slices = slices[(slices <= len(r))]
	r_splitted = np.split(r,slices)
	
	ncores = len(r_splitted)
	
	M200  = np.ones(ncores)*M200
	ellip = np.ones(ncores)*ellip
	z     = np.ones(ncores)*z
	h     = np.ones(ncores)*h
	miss  = np.ones(ncores,dtype=bool)*misscentred
	s_off = np.ones(ncores)*s_off
	comp  = [components]*ncores
	v  = np.ones(ncores,dtype=bool)*verbose
	y  = np.ones(ncores,dtype=bool)*Yanmiss
	
	entrada = np.array([r_splitted,M200,ellip,z,h,miss,s_off,comp,v,y]).T
	
	pool = Pool(processes=(ncores))
	salida=np.array(pool.map(multipole_shear_unpack, entrada))
	pool.terminate()

	gt0 = []
	gt2 = []
	gx2 = []
	
	if misscentred:
		gt0_off = []
		gt_off0 = []
		gt_off  = []
		gx_off  = []

	for s in salida:
		gt0 = np.append(gt0,s['Gt0'])
		gt2 = np.append(gt2,s['Gt2'])
		gx2 = np.append(gx2,s['Gx2'])
		if misscentred:
			gt0_off = np.append(gt0_off,s['Gt0_off'])
			gt_off0 = np.append(gt_off0,s['Gt_off'])
			gt_off  = np.append(gt_off,s['Gt_off_cos'])
			gx_off  = np.append(gx_off,s['Gx_off_sin'])
			
	output = {'Gt0':gt0,'Gt2':gt2,'Gx2':gx2}
	
	if misscentred:
		output.update({'Gt0_off':gt0_off,'Gt_off':gt_off0,'Gt_off_cos':gt_off,'Gx_off_sin':gx_off})

	return output


def model_Gamma(multipole_out,component = 't', misscentred = False, pcc = 0.8):
	
	if misscentred:
		if component == 't':
			G = pcc*multipole_out['Gt0'] + (1-pcc)*multipole_out['Gt_off']
		if component == 'tcos':
			G = pcc*multipole_out['Gt2'] + (1-pcc)*multipole_out['Gt_off_cos']
		if component == 'xsin':
			G = pcc*multipole_out['Gx2'] + (1-pcc)*multipole_out['Gx_off_sin']
	else:
		if component == 't':
			G = multipole_out['Gt0'] 
		if component == 'tcos':
			G = multipole_out['Gt2'] 
		if component == 'xsin':
			G = multipole_out['Gx2'] 
			
	return G
