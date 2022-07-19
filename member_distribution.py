import numpy as np
from profiles_fit import SIGMA_nfw
from profiles_fit import r200_nfw
from astropy.cosmology import LambdaCDM
from multipoles_shear import multipole_shear
from scipy import integrate

def projected_coodinates(x,y,z,xc,yc,zc):

     ra_center = np.arctan(xc/yc)
     dec_center = np.arcsin(zc/np.sqrt(xc**2 + yc**2 + zc**2))
     
     e1x =   np.cos(ra_center)
     e1y =   -np.sin(ra_center)
     
     e2x = -np.sin(dec_center) * np.sin(ra_center)
     e2y = -np.sin(dec_center) * np.cos(ra_center)
     e2z =  np.cos(dec_center)
     
     # Projected coordinates
     xp = e1x*x + e1y*y
     yp = e2x*x + e2y*y + e2z*z
     
     return xp,yp


def compute_axis(x,y,z,xp,yp,w = None,wp = None):
     
     '''
     
     Compute 3D and projected 2D axis
     according to the moment of inertia
     (INPUT)
     x,y,z 3D coordinates arrays of len(N)
           wher N is the total number of particles

     xp,yp 2D coordinates arrays of len(N)
           wher N is the total number of particles
           
     w     len(N) array with weights, if not define
           this will be defined as a unity array
           
     (OUTPUT)
     v,w,v2d,w2d Eigenvectors and eingenvalues in 3D and projected
     '''
     
     if not np.all(w):
          w = np.ones(len(x))

     if not np.all(wp):
          wp = np.ones(len(xp))

     # COMPUTE 3D Tensor

     T3D = np.zeros((3,3))

     T3D[0,0] = np.sum(w*x**2)
     T3D[0,1] = np.sum(w*x*y)
     T3D[0,2] = np.sum(w*x*z)

     T3D[1,0] = np.sum(w*y*x)
     T3D[1,1] = np.sum(w*y**2)
     T3D[1,2] = np.sum(w*y*z)

     T3D[2,0] = np.sum(w*z*x)
     T3D[2,1] = np.sum(w*z*y)
     T3D[2,2] = np.sum(w*z**2)

     w3d,v3d =np.linalg.eig(T3D)

     j = np.flip(np.argsort(w3d))
     w3d = w3d[j] # Ordered eingenvalues
     v3d = v3d[:,j] # Ordered eingenvectors
     
     # -----------------------------------------------
     # COMPUTE projected quantities
          
     T2D = np.zeros((2,2))
     
     T2D[0,0] = np.sum(wp*xp**2)
     T2D[0,1] = np.sum(wp*xp*yp)
     T2D[1,0] = np.sum(wp*xp*yp)
     T2D[1,1] = np.sum(wp*yp**2)
     
     w2d,v2d =np.linalg.eig(T2D)
     
     j = np.flip(np.argsort(w2d))
     w2d = w2d[j] # Ordered eingenvalues
     v2d = v2d[:,j] # Ordered eingenvectors
     
     return v3d,w3d,v2d,w2d

def momentos(dx,dy,w):
     
     Q11  = np.sum((dx**2)*w)/np.sum(w)
     Q22  = np.sum((dy**2)*w)/np.sum(w)
     Q12  = np.sum((dx*dy)*w)/np.sum(w)
     E1 = (Q11-Q22)/(Q11+Q22)
     E2 = (2.*Q12)/(Q11+Q22)
     e = np.sqrt(E1**2 + E2**2)
     theta = np.arctan2(E2,E1)/2.
     return e,theta

def NFW_ellip_plot(M200,z,e0,e1,sample):
     ############  MAKING A GRID
     
     q0        = (1.-e0)/(1.+e0)
     q1        = (1.-e1)/(1.+e1)
     
     a1  = np.arange(-5.001,5.3,0.07)
     a0  = np.arange(-1.55,1.5,0.03)
     
     x1,y1 = np.meshgrid(a1,a1)
     x0,y0 = np.meshgrid(a0,a0)
     
     x1 = x1.flatten()
     y1 = y1.flatten()

     x0 = x0.flatten()
     y0 = y0.flatten()
     
     r0 = np.sqrt(x0**2 + y0**2)
     r1 = np.sqrt(x1**2 + y1**2)

     theta0  = np.arctan2(y0,x0)
     theta1  = np.arctan2(y1,x1)
     
     j0   = argsort(r0)
     r0   = r0[j0]
     x0,y0 = x0[j0],y0[j0]
     theta0  = theta0[j0]
     
     j1   = argsort(r1)
     r1   = r1[j1]
     x1,y1 = x1[j1],y1[j1]
     theta1  = theta1[j1]
     
     # COMPUTE COMPONENTS

     R0 = (r0**2)*np.sqrt(q0*(np.cos(theta0))**2 + (np.sin(theta0))**2 / q0)
     R1 = (r1**2)*np.sqrt(q1*(np.cos(theta1))**2 + (np.sin(theta1))**2 / q1)
     
     out = multipole_shear(R1,M200=M200,z=z,ellip=e1)
     S1 = out['Gt0']
     Sn1 = np.log10(S1/np.sum(S1))
     mr1 = R1 < 5.0

     out = multipole_shear(R0,M200=M200,z=z,ellip=e0)
     S0 = out['Gt0']
     Sn0 = np.log10(S0/np.sum(S1))
     mr0 = R0 < 1.0
    
     fig = plt.figure(figsize=(5,5))
     ax = plt.gca()
     ax.set_facecolor('k')
     plt.title(sample)
     plt.scatter(x1[mr1],y1[mr1],c=Sn1[mr1],cmap = 'magma',alpha=1,s = 20,vmin=-4.4,vmax=-2.5)
     plt.scatter(x0[mr0],y0[mr0],c=Sn0[mr0],cmap = 'magma',alpha=1,s = 20,vmin=-4.4,vmax=-2.5)
     plt.xlabel('x [Mpc]')
     plt.ylabel('y [Mpc]')
     plt.axis([-3,3,-3,3])
     plt.savefig("/home/eli/Documentos/Astronomia/posdoc/halo-elongation/halo_"+sample+'.png')

     
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
     a  = np.arange(-5.001,5.3,0.1)
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
     
     R = r*np.sqrt(q*(np.cos(fi))**2 + (np.sin(fi))**2 / q)
     
     
     out = multipole_shear(R,M200=M200,z=z,ellip=e)
     S = out['Gt0']
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

