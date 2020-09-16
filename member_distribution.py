import numpy as np
sys.path.append('/home/eli/lens_codes_v3.7')
from profiles_fit import SIGMA_nfw
from profiles_fit import r200_nfw
from astropy.cosmology import LambdaCDM
from multipoles_shear import multipole_shear
from scipy import integrate

def momentos(dx,dy,w):
     
     Q11  = np.sum((dx**2)*w)/np.sum(w)
     Q22  = np.sum((dy**2)*w)/np.sum(w)
     Q12  = np.sum((dx*dy)*w)/np.sum(w)
     E1 = (Q11-Q22)/(Q11+Q22)
     E2 = (2.*Q12)/(Q11+Q22)
     e = np.sqrt(E1**2 + E2**2)
     theta = np.arctan2(E2,E1)/2.
     return e,theta
     
def D_miss(M200,e,z,Nmembers,niter=1000.):

     M200     = 2.e14
     e        = 0.2
     z        = 0.25
     
     Nmembers = 20
     niter    = 100
     q        = (1.-e)/(1.+e)
     c_ang    = 0.
     ############  MAKING A GRID
     
     # a  = np.logspace(np.log10(0.01),np.log10(5.),10)
     a  = np.arange(-1.001,1.3,0.02)
     # a  = np.append(a,-1.*a)
     
     x,y = np.meshgrid(a,a)
     
     x = x.flatten()
     y = y.flatten()
     
     r = np.sqrt(x**2 + y**2)
     
     
     theta  = np.arctan2(y,x)
     j   = argsort(r)
     r   = r[j]
     theta  = theta[j]
     x,y = x[j],y[j]
     index = np.arange(len(r))
     
     # COMPUTE COMPONENTS
         
     fi = theta - np.deg2rad(c_ang)
     
     R = (r**2)*np.sqrt(q*(np.cos(fi))**2 + (np.sin(fi))**2 / q)
     
     
     out = multipole_shear(R,M200=M200,z=z,ellip=e)
     S = out['S0']
     Sn = S/np.sum(S)
     
     ang = np.array([])

     plt.scatter(x,y,c=np.log10(Sn),cmap = 'inferno',alpha=0.1)
     plt.axis([-1,1,-1,1])
     for j in range(niter):
          ri = np.random.choice(index,Nmembers,replace = False, p = Sn)
          ang = np.append(ang,momentos(x[ri],y[ri],np.ones(20))[1])
          m = np.tan(ang[-1])
          # plt.plot(x,m*x,'C2',alpha=0.3)
     
     s = np.std(ang)
     arg = lambda x: np.exp((-1.*x**2)/s**2)*np.cos(2.*x)
     D = integrate.quad(arg, -np.pi/2., np.pi/2.)[0]
     
     return D
     
def M200(Lambda,z):
     
     M0    = 2.21e14
     alpha = 1.33
     M200m = M0*((Lambda/40.)**alpha)
     
     from colossus.cosmology import cosmology
     from colossus.halo import mass_defs
     from colossus.halo import concentration
     
     c200m = concentration.concentration(M200m, '200m', z, model = 'duffy08')
     M200c, R200c, c200c = mass_defs.changeMassDefinition(M200m, c200m, z, '200m', '200c')
     
     return M200c

