import sys
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import LambdaCDM
from astropy.wcs import WCS
from maria_func import *
from fit_profiles_curvefit import *
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from astropy.constants import G,c,M_sun,pc
from scipy import stats
import astropy.units as u
from astropy.coordinates import SkyCoord
from models_profiles import Gamma
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample

# For map
wcs = WCS(naxis=2)
wcs.wcs.crpix = [0., 0.]
wcs.wcs.cdelt = [1./3600., 1./3600.]
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]    

#parameters
cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)


parser = argparse.ArgumentParser()
parser.add_argument('-sample', action='store', dest='sample',default='pru')
parser.add_argument('-lens_cat', action='store', dest='lcat',default='/mnt/simulations/MICE/HALO_Props_MICE.fits')
parser.add_argument('-lM_min', action='store', dest='lM_min', default=14.)
parser.add_argument('-lM_max', action='store', dest='lM_max', default=15.5)
parser.add_argument('-z_min', action='store', dest='z_min', default=0.1)
parser.add_argument('-z_max', action='store', dest='z_max', default=0.4)
parser.add_argument('-q_min', action='store', dest='q_min', default=0.)
parser.add_argument('-q_max', action='store', dest='q_max', default=1.)
parser.add_argument('-T_min', action='store', dest='T_min', default=0.)
parser.add_argument('-T_max', action='store', dest='T_max', default=1.)
parser.add_argument('-rs_min', action='store', dest='rs_min', default=0.)
parser.add_argument('-rs_max', action='store', dest='rs_max', default=1.)
parser.add_argument('-misalign', action='store', dest='misalign', default='False')
parser.add_argument('-miscen', action='store', dest='miscen', default='False')
parser.add_argument('-relax', action='store', dest='relax', default='False')
parser.add_argument('-domap', action='store', dest='domap', default='False')
parser.add_argument('-addnoise', action='store', dest='addnoise', default='False')
parser.add_argument('-RIN', action='store', dest='RIN', default=100.)
parser.add_argument('-ROUT', action='store', dest='ROUT', default=10000.)
parser.add_argument('-nbins', action='store', dest='nbins', default=40)
parser.add_argument('-ncores', action='store', dest='ncores', default=10)
parser.add_argument('-h_cosmo', action='store', dest='h_cosmo', default=1.)
parser.add_argument('-ides_list', action='store', dest='idlist', default=None)
parser.add_argument('-resNFW_max', action='store', dest='resNFW_max', default=100.)
parser.add_argument('-R5s_max', action='store', dest='R5s_max', default=100.)
parser.add_argument('-R5s_min', action='store', dest='R5s_min', default=0.)
parser.add_argument('-soff', action='store', dest='soff', default=0.3)
parser.add_argument('-nback', action='store', dest='nback', default=30)
args = parser.parse_args()

sample     = args.sample
idlist     = args.idlist
lcat       = args.lcat
lM_min     = float(args.lM_min)
lM_max     = float(args.lM_max) 
z_min      = float(args.z_min) 
z_max      = float(args.z_max) 
q_min      = float(args.q_min) 
q_max      = float(args.q_max) 
T_min      = float(args.T_min) 
T_max      = float(args.T_max) 
rs_min     = float(args.rs_min) 
rs_max     = float(args.rs_max) 
RIN        = float(args.RIN)
ROUT       = float(args.ROUT)
ndots      = int(args.nbins)
ncores     = int(args.ncores)
hcosmo     = float(args.h_cosmo)
resNFW_max = float(args.resNFW_max)
R5s_max = float(args.R5s_max)
R5s_min = float(args.R5s_min)
soff = float(args.soff)
nback = float(args.nback)

if args.relax == 'True':
    relax = True
elif args.relax == 'False':
    relax = False

if args.domap == 'True':
    domap = True
elif args.domap == 'False':
    domap = False

if args.misalign == 'False':
    misalign = False
else:
    misalign = int(args.misalign)

if args.miscen == 'True':
    miscen = True
elif args.miscen == 'False':
    miscen = False

if args.addnoise == 'True':
    addnoise = True
elif args.addnoise == 'False':
    addnoise = False


'''
lcat = 'HALO_Props_MICE.fits'
sample='pru'
lM_min=14.0
lM_max=14.5
z_min = 0.1
z_max = 0.2
q_min = 0.
q_max = 1.
rs_min = 0
rs_max = 1
RIN = 100.
ROUT = 10000.
ndots= 40
ncores = 32
hcosmo = 1.0 
vmice = '2'
rmin = 0.
rmax = 1000.
idlist = None
relax = False
domap = False
R5s_min = 0
R5s_max = 100.
resNFW_max = 100.
misalign = False
miscen = False
addnoise = False
nback = 30.
T_min = 0.
T_max = 1.
'''


folder = '/mnt/simulations/MICE/'
S      = fits.open(folder+'MICE_sources_HSN_withextra.fits')[1].data

if nback < 30.:
    nselec = int(nback*5157*3600.)
    j      = np.random.choice(np.array(len(S)),nselec)
    S  = S[j]

print('BACKGROUND GALAXY DENSINTY',len(S)/(5157*3600))

def partial_map(RA0,DEC0,Z,angles,
                RIN,ROUT,ndots,h,
                addnoise,roff,phi_off):

        
        lsize = int(np.sqrt(ndots))
        ndots = int(ndots)
        
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        dl  = cosmo.angular_diameter_distance(Z).value
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
        
        delta = (ROUT*1.5)/(3600*KPCSCALE)
        rarray = [[-1.*(ROUT*1.e-3),(ROUT*1.e-3)],[-1.*(ROUT*1.e-3),(ROUT*1.e-3)]]
        
        t0 = time.time()
        # mask = (S.ra < (RA0+delta))&(S.ra > (RA0-delta))&(S.dec > (DEC0-delta))&(S.dec < (DEC0+delta))&(S.z_v > (Z+0.1))
        mask = (S.ra_gal < (RA0+delta))&(S.ra_gal > (RA0-delta))&(S.dec_gal > (DEC0-delta))&(S.dec_gal < (DEC0+delta))&(S.z_cgal_v > (Z+0.1))
                       
        catdata = S[mask]

        ds  = cosmo.angular_diameter_distance(catdata.z_cgal_v).value
        dls = cosmo.angular_diameter_distance_z1z2(Z, catdata.z_cgal_v).value
                
        
        BETA_array = dls/ds
        
        Dl = dl*1.e6*pc
        sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)

        # Add miscentring
        c1 = SkyCoord(RA0*u.degree, DEC0*u.degree)
        roff_deg = roff/(KPCSCALE*3600.)
        c2 = c1.directional_offset_by(phi_off*u.degree,roff_deg*u.degree)


        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.ra_gal),
                                        np.deg2rad(catdata.dec_gal),
                                        np.deg2rad(c2.ra.value),
                                        np.deg2rad(c2.dec.value))


        theta2 = (2.*np.pi - theta) +np.pi/2.
        theta_ra = theta2
        theta_ra[theta2 > 2.*np.pi] = theta2[theta2 > 2.*np.pi] - 2.*np.pi
                       
        e1     = catdata.gamma1
        e2     = -1.*catdata.gamma2
        
        # Add shape noise due to intrisic galaxy shapes        
        if addnoise:
            es1 = -1.*catdata.eps1
            es2 = catdata.eps2
            e1 += es1
            e2 += es2
       
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
        # '''
        k  = catdata.kappa*sigma_c
                
        wcs.wcs.crval = [RA0,DEC0]
        dx, dy = wcs.wcs_world2pix(catdata.ra_gal,catdata.dec_gal, 0)

        dx = dx*KPCSCALE*1.e-3
        dy = dy*KPCSCALE*1.e-3
          
        del(e1)
        del(e2)
        
        r = np.rad2deg(rads)*3600*KPCSCALE
        
        del(rads)
        
        
        Ntot = len(catdata)
        del(catdata)    
        
        
        t = np.tile(theta_ra,(3,1)).T
        an = np.tile(angles,(len(theta_ra),1))
        at     = t - an
        
        GTsum = np.zeros((ndots,3))
        GXsum = np.zeros((ndots,3))
        Ksum  = np.zeros((ndots,3))
        Ninbin  = np.zeros((ndots,3))
        
        
        for j in range(3):
                x_t  = (dx*np.cos(an[:,j]) + dy*np.sin(an[:,j])) 
                y_t = (-1.*dx*np.sin(an[:,j]) + dy*np.cos(an[:,j])) 
                m   =  (abs(x_t) < (ROUT*1.e-3))*(abs(y_t) < (ROUT*1.e-3)) 
                out = stats.binned_statistic_2d(x_t[m], y_t[m], et[m], 
                                               'count',bins=lsize, range = rarray)
                Ninbin[:,j] = out.statistic.flatten()
                out = stats.binned_statistic_2d(x_t[m], y_t[m], et[m], 
                                               'sum',bins=lsize, range = rarray)
                GTsum[:,j] = out.statistic.flatten()
                out = stats.binned_statistic_2d(x_t[m], y_t[m], ex[m], 
                                               'sum',bins=lsize, range = rarray)
                GXsum[:,j] = out.statistic.flatten()
                out = stats.binned_statistic_2d(x_t[m], y_t[m], k[m], 
                                               'sum',bins=lsize, range = rarray)
                Ksum[:,j] = out.statistic.flatten()

        Ntot = m.sum()
   
        output = {'GTsum':GTsum,'GXsum':GXsum,
                   'Ksum': Ksum, 'N_inbin':Ninbin,
                   'Ntot': Ntot}
        
        return output

def partial_map_unpack(minput):
	return partial_map(*minput)

def partial_profile(RA0,DEC0,Z,angles,
                    RIN,ROUT,ndots,h,
                    addnoise,roff,phi_off):

        ndots = int(ndots)

        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        dl  = cosmo.angular_diameter_distance(Z).value
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
        
        delta = ROUT/(3600*KPCSCALE)
        
        mask = (S.ra_gal < (RA0+delta))&(S.ra_gal > (RA0-delta))&(S.dec_gal > (DEC0-delta))&(S.dec_gal < (DEC0+delta))&(S.z_cgal_v > (Z+0.1))
        catdata = S[mask]

        ds  = cosmo.angular_diameter_distance(catdata.z_cgal_v).value
        dls = cosmo.angular_diameter_distance_z1z2(Z, catdata.z_cgal_v).value
                
        
        BETA_array = dls/ds
        
        Dl = dl*1.e6*pc
        sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)

        # Add miscentring
        c1 = SkyCoord(RA0*u.degree, DEC0*u.degree)
        roff_deg = roff/(KPCSCALE*3600.)
        c2 = c1.directional_offset_by(phi_off*u.degree,roff_deg*u.degree)


        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.ra_gal),
                                        np.deg2rad(catdata.dec_gal),
                                        np.deg2rad(c2.ra.value),
                                        np.deg2rad(c2.dec.value))
        
        theta2 = (2.*np.pi - theta) +np.pi/2.
        theta_ra = theta2
        theta_ra[theta2 > 2.*np.pi] = theta2[theta2 > 2.*np.pi] - 2.*np.pi
                       
        e1     = catdata.gamma1
        e2     = -1.*catdata.gamma2

        # Add shape noise due to intrisic galaxy shapes        
        if addnoise:
            es1 = -1.*catdata.eps1
            es2 = catdata.eps2
            e1 += es1
            e2 += es2
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
        # '''
        k  = catdata.kappa*sigma_c
          
        del(e1)
        del(e2)
        
        r = np.rad2deg(rads)*3600*KPCSCALE
        del(rads)
        
        
        Ntot = len(catdata)
        # del(catdata)    
        
        bines = np.round(np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1),0)
        dig = np.digitize(r,bines)
        
        t = np.tile(theta_ra,(3,1)).T
        an = np.tile(angles,(len(theta_ra),1))
        at     = t - an
        
        
        SIGMAwsum    = []
        DSIGMAwsum_T = []
        DSIGMAwsum_X = []
        N_inbin     = []
        
        N_inbin_par = np.zeros((ndots,3))
        N_inbin_per = np.zeros((ndots,3))
              
        SIGMAcos_wsum = np.zeros((ndots,3))
        SIGMApar_wsum  = np.zeros((ndots,3))
        SIGMAper_wsum  = np.zeros((ndots,3))
        GAMMATcos_wsum = np.zeros((ndots,3))
        GAMMAXsin_wsum = np.zeros((ndots,3))
                
        COS2_2theta = np.zeros((ndots,3))
        SIN2_2theta = np.zeros((ndots,3))
                       
        for nbin in range(ndots):
                mbin = dig == nbin+1              
                
                SIGMAwsum    = np.append(SIGMAwsum,k[mbin].sum())
                DSIGMAwsum_T = np.append(DSIGMAwsum_T,et[mbin].sum())
                DSIGMAwsum_X = np.append(DSIGMAwsum_X,ex[mbin].sum())

                SIGMAcos_wsum[nbin,:]  = np.sum((np.tile(k[mbin],(3,1))*np.cos(2.*at[mbin]).T),axis=1)
                
                GAMMATcos_wsum[nbin,:] = np.sum((np.tile(et[mbin],(3,1))*np.cos(2.*at[mbin]).T),axis=1)
                GAMMAXsin_wsum[nbin,:] = np.sum((np.tile(ex[mbin],(3,1))*np.sin(2.*at[mbin]).T),axis=1)
                
                COS2_2theta[nbin,:] = np.sum((np.cos(2.*at[mbin]).T)**2,axis=1)
                SIN2_2theta[nbin,:] = np.sum((np.sin(2.*at[mbin]).T)**2,axis=1)
                               
                N_inbin = np.append(N_inbin,len(et[mbin]))
                
                index = np.arange(mbin.sum())
                
                mper = ((at[mbin] > np.pi/4.) * (at[mbin] < 3.*np.pi/4.))+((at[mbin] > 5.*np.pi/4.) * (at[mbin] < 7.*np.pi/4.))

                SIGMApar_wsum[nbin,:] = np.sum((np.tile(k[mbin],(3,1))*(~mper).T),axis=1)
                SIGMAper_wsum[nbin,:] = np.sum((np.tile(k[mbin],(3,1))*mper.T),axis=1)
                
                N_inbin_par[nbin,:] = np.sum(~mper,axis=0)
                N_inbin_per[nbin,:] = np.sum(mper,axis=0)
        
        output = {'SIGMAwsum':SIGMAwsum,'DSIGMAwsum_T':DSIGMAwsum_T,
                   'DSIGMAwsum_X':DSIGMAwsum_X,'SIGMAcos_wsum': SIGMAcos_wsum,
                   'SIGMApar_wsum':SIGMApar_wsum,'SIGMAper_wsum': SIGMAper_wsum,
                   'GAMMATcos_wsum': GAMMATcos_wsum, 'GAMMAXsin_wsum': GAMMAXsin_wsum,
                   'COS2_2theta_wsum':COS2_2theta,'SIN2_2theta_wsum':SIN2_2theta,
                   'N_inbin_par':N_inbin_par,'N_inbin_per':N_inbin_per,
                   'N_inbin':N_inbin,'Ntot':Ntot}
        
        return output

def partial_profile_unpack(minput):
	return partial_profile(*minput)

def cov_matrix(array):
        
        K = len(array)
        Kmean = np.average(array,axis=0)
        bins = array.shape[1]
        
        COV = np.zeros((bins,bins))
        
        for k in range(K):
            dif = (array[k]- Kmean)
            COV += np.outer(dif,dif)        
        
        COV *= (K-1)/K
        return COV
        
def main(lcat, sample='pru',
         lM_min=14.,lM_max=14.2,
         z_min = 0.1, z_max = 0.4,
         q_min = 0., q_max = 1.0,
         T_min = 0., T_max = 1.0,
         rs_min = 0., rs_max = 1.0,
         resNFW_max = 100., relax=False,
         R5s_min = 0., R5s_max = 100.,
         domap = False, RIN = 400., ROUT =5000.,
         ndots= 40, ncores=10, 
         idlist= None, hcosmo=1.0, 
         addnoise = False, misalign = False, 
         miscen = False, soff = 0.3):

        '''
        
        INPUT
        ---------------------------------------------------------
        sample         (str) sample name
        lM_min         (float) lower limit for log(Mass) - >=
        lM_max         (float) higher limit for log(Mass) - <
        z_min          (float) lower limit for z - >=
        z_max          (float) higher limit for z - <
        q_min          (float) lower limit for q - >=
        q_max          (float) higher limit for q - <
        T_min          (float) lower limit for T - >=
        T_max          (float) higher limit for T - <
        rs_min         (float) lower limit r_scale = r_c/r_max
        rs_max         (float) higher limit r_scale = r_c/r_max
        R5s_min         (float) lower limit R5 scaled
        R5s_max         (float) higher limit R5 scaled
        resNFW_max     (float) higher limit for resNFW_S
        relax          (bool)  Select only relaxed halos
        domap          (bool) Instead of computing a profile it 
                       will compute a map with 2D bins ndots lsize
        RIN            (float) Inner bin radius of profile
        ROUT           (float) Outer bin radius of profile
        ndots          (int) Number of bins of the profile
        ncores         (int) to run in parallel, number of cores
        h              (float) H0 = 100.*h
        misaling       (bool or float) add misalignment with 
                                       a normal distribution of misalign deg 
                                       standard deviation
        miscen         (bool) add a miscentring for the 25percent of the halos
        addnoise       (bool) add shape noise
        soff         (float) dispersion of Rayleigh distribution for miscenter
        '''

        cosmo = LambdaCDM(H0=100*hcosmo, Om0=0.25, Ode0=0.75)
        tini = time.time()
        
        print('Lens catalog ',lcat)
        print('Sample ',sample)
        print('Selecting halos with:')
        
        
        if idlist:
                print('From id list '+idlist)
        else:
                print(lM_min,' <= log(M) < ',lM_max)
                print(z_min,' <= z < ',z_max)
                print(q_min,' <= q < ',q_max)
                print(rs_min,' <= rs < ',rs_max)
                # print(R5s_min,' <= R5s < ',R5s_max)
                print('resNFW_S < ',resNFW_max)
                print('h ',hcosmo)
                print('misalign '+str(misalign))
        
        if addnoise:
            print('ADDING SHAPE NOISE')
        
        #reading cats
                
        L = fits.open(lcat)[1].data               

        '''
        # To try old centre
        
        ra = np.rad2deg(np.arctan(L.xc/L.yc))
        ra[L.yc==0] = 90.
        dec = np.rad2deg(np.arcsin(L.zc/sqrt(L.xc**2 + L.yc**2 + L.zc**2)))

        L.ra_rc = ra
        L.dec_rc = dec
        '''
        ra = L.ra_rc
        dec = L.dec_rc
        
        try:
            Eratio = (2.*L.K/abs(L.U))
        except:
            Eratio = (2.*L.ekin/abs(L.epot))
                               
        if idlist:
                ides = np.loadtxt(idlist).astype(int)
                mlenses = np.in1d(L.unique_halo_id,ides)
        else:
                
                rs      = L.offset
                # T       = (1. - L.q**2)/(1. - L.s**2)
                
                mmass   = (L.lgm >= lM_min)*(L.lgm < lM_max)
                mz      = (L.redshift >= z_min)*(L.redshift < z_max)
                # mq      = (L.q2d >= q_min)*(L.q2d < q_max)
                # mT      = (T >= T_min)*(T < T_max)
                # mrs     = (rs >= rs_min)*(rs < rs_max)
                # mres    = L.resNFW_S < resNFW_max
                # mr5s    = (L.R5scale >= R5s_min)*(L.R5scale < R5s_max)
                mlenses = mmass*mz#*mq*mT*mrs*mres
        
        # SELECT RELAXED HALOS
        if relax:
            print('Only relaxed halos are considered')
            sample = sample+'_relaxed'
            mlenses = mlenses*(L.offset < 0.1)*(Eratio < 1.35)
                
        Nlenses = mlenses.sum()

        if Nlenses < ncores:
                ncores = Nlenses
        
        print('Nlenses',Nlenses)
        print('CORRIENDO EN ',ncores,' CORES')
        
        L = L[mlenses]
                
        #Computing SMA axis        
        
        # theta  = np.array([np.zeros(sum(mlenses)),np.arctan(L.a2Dy/L.a2Dx),np.arctan(L.a2Dry/L.a2Drx)]).T                
        theta  = np.array([np.zeros(sum(mlenses)),np.zeros(sum(mlenses)),np.zeros(sum(mlenses))]).T                
        
        # Introduce misalignment
        if misalign:
            toff = np.random.normal(0,np.deg2rad(misalign),mlenses.sum())
            theta += np.vstack((toff,toff,toff)).T
            sample = sample+'_mis'+str(misalign)
        
        # Define K masks   
        ncen = 50
        X    = np.array([L.ra_rc,L.dec_rc]).T
        
        km = kmeans_sample(X, ncen, maxiter=100, tol=1.0e-5)
        labels = km.find_nearest(X)
        kmask = np.zeros((ncen+1,len(X)))
        kmask[0] = np.ones(len(X)).astype(bool)
        
        for j in np.arange(1,ncen+1):
            kmask[j] = ~(labels == j-1)

        
        '''
        for j in np.arange(1,ncen+1):
            kmask[j] = ~(labels == j-1)


        
        kmask = np.zeros((ncen+1,len(ra)))
        kmask[0] = np.ones(len(ra)).astype(bool)
        
        ramin  = np.min(ra)
        cdec   = np.sin(np.deg2rad(dec))
        decmin = np.min(cdec)
        dra  = ((np.max(ra)+1.e-5)  - ramin)/10.
        ddec = ((np.max(cdec)+1.e-5) - decmin)/10.
        
        c    = 1
        
        for a in range(10): 
                for d in range(10): 
                        mra  = (ra  >= ramin + a*dra)*(ra < ramin + (a+1)*dra) 
                        mdec = (cdec >= decmin + d*ddec)*(cdec < decmin + (d+1)*ddec) 
                        # plt.plot(ra[(mra*mdec)],dec[(mra*mdec)],'C'+str(c+1)+',')
                        kmask[c] = ~(mra*mdec)
                        c += 1
        '''
                
        # Introduce miscentring
        
        ind_rand0 = np.arange(Nlenses)
        np.random.shuffle(ind_rand0)
        
        roff = np.zeros(Nlenses)
        phi_off = np.zeros(Nlenses)
        
        if miscen:
            nshift = int(Nlenses*0.25)
            x = np.random.uniform(0,5,10000)
            peso = Gamma(x,soff)/sum(Gamma(x,soff))
            roff[ind_rand0[:nshift]] = np.random.choice(x,nshift,p=peso)*1.e3
            phi_off[ind_rand0[:nshift]] = np.random.uniform(0.,360.,nshift)
            sample = sample+'_miscen'
                                
        # SPLIT LENSING CAT
        
        lbins = int(round(Nlenses/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nlenses)]
        Lsplit = np.split(L,slices)
        Tsplit = np.split(theta,slices)        
        Ksplit = np.split(kmask.T,slices)
        Rsplit = np.split(roff,slices)
        PHIsplit = np.split(phi_off,slices)
        
        if domap:

            print('Maps have ',ndots,'pixels')
            print('up to ',ROUT,'kpc')
            
            os.system('mkdir ../maps')
            output_file = 'maps/map_'+sample+'.fits'

            # Defining 2D bins
            
            lsize = int(np.sqrt(ndots))
            
            ndots = int(lsize**2)
            
            Rmpc = 1.e-3*ROUT
            
            xb = np.linspace(-1.*Rmpc+(Rmpc/lsize),Rmpc-(Rmpc/lsize),lsize)
            
            ybin, xbin = np.meshgrid(xb,xb)
            
            ybin = ybin.flatten()
            xbin = xbin.flatten()

            # WHERE THE SUMS ARE GOING TO BE SAVED
            GTsum = np.zeros((ncen+1,ndots,3))
            GXsum = np.zeros((ncen+1,ndots,3))
            Ksum  = np.zeros((ncen+1,ndots,3))        
            
            Ninbin = np.zeros((ncen+1,ndots,3))
            
            # FUNCTION TO RUN IN PARALLEL
            partial = partial_map_unpack
            

        else:

            print('Profile has ',ndots,'bins')
            print('from ',RIN,'kpc to ',ROUT,'kpc')
            os.system('mkdir ../profiles')
            output_file = 'profiles/profile_'+sample+'.fits'

            # Defining radial bins
            bines = np.round(np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1),0)
            R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3

            # WHERE THE SUMS ARE GOING TO BE SAVED
            
            Ninbin = np.zeros((ncen+1,ndots))
            
            SIGMAwsum    = np.zeros((ncen+1,ndots)) 
            DSIGMAwsum_T = np.zeros((ncen+1,ndots)) 
            DSIGMAwsum_X = np.zeros((ncen+1,ndots))
                
            SIGMAcos_wsum = np.zeros((ncen+1,ndots,3))
            SIGMApar_wsum = np.zeros((ncen+1,ndots,3))
            SIGMAper_wsum = np.zeros((ncen+1,ndots,3))
            
            GAMMATcos_wsum = np.zeros((ncen+1,ndots,3))
            GAMMAXsin_wsum = np.zeros((ncen+1,ndots,3))
    
            COS2_2theta_wsum = np.zeros((ncen+1,ndots,3))
            SIN2_2theta_wsum = np.zeros((ncen+1,ndots,3))
            
            Ninbin_par = np.zeros((ncen+1,ndots,3))
            Ninbin_per = np.zeros((ncen+1,ndots,3))
            
            # FUNCTION TO RUN IN PARALLEL
            partial = partial_profile_unpack
            

        print('Saved in ../'+output_file)
                                   
        
        
        Ntot         = np.array([])
        tslice       = np.array([])
        
        for l in range(len(Lsplit)):
                
                print('RUN ',l+1,' OF ',len(Lsplit))
                
                t1 = time.time()
                
                num = len(Lsplit[l])
                
                rin  = RIN*np.ones(num)
                rout = ROUT*np.ones(num)
                nd   = ndots*np.ones(num)
                h_array   = hcosmo*np.ones(num)
                addnoise_array   = np.array([addnoise]*np.ones(num))
                
                if num == 1:
                        entrada = [Lsplit[l].ra_rc[0], Lsplit[l].dec_rc[0],
                                   Lsplit[l].redshift[0],Tsplit[l][0],
                                   RIN,ROUT,ndots,hcosmo,
                                   addnoise,Rsplit[l][0],PHIsplit[l][0]]
                        
                        salida = [partial(entrada)]
                else:          
                        entrada = np.array([Lsplit[l].ra_rc,Lsplit[l].dec_rc,
                                        Lsplit[l].redshift,Tsplit[l].tolist(),
                                        rin,rout,nd,h_array,
                                        addnoise_array,Rsplit[l],PHIsplit[l]]).T
                        
                        pool = Pool(processes=(num))
                        salida = np.array(pool.map(partial, entrada))
                        pool.terminate()
                                
                for j in range(len(salida)):
                        
                        profilesums = salida[j]
                        Ntot         = np.append(Ntot,profilesums['Ntot'])
                        
                        if domap:
                            
                            km      = np.tile(Ksplit[l][j],(3,ndots,1)).T
                            Ninbin += np.tile(profilesums['N_inbin'],(ncen+1,1,1))*km

                            GTsum += np.tile(profilesums['GTsum'],(ncen+1,1,1))*km
                            GXsum += np.tile(profilesums['GXsum'],(ncen+1,1,1))*km
                            Ksum  += np.tile(profilesums['Ksum'],(ncen+1,1,1))*km
                            
                        else:

                            km      = np.tile(Ksplit[l][j],(3,ndots,1)).T

                            Ninbin += np.tile(profilesums['N_inbin'],(ncen+1,1))*km[:,:,0]
                                                
                            SIGMAwsum    += np.tile(profilesums['SIGMAwsum'],(ncen+1,1))*km[:,:,0]
                            DSIGMAwsum_T += np.tile(profilesums['DSIGMAwsum_T'],(ncen+1,1))*km[:,:,0]
                            DSIGMAwsum_X += np.tile(profilesums['DSIGMAwsum_X'],(ncen+1,1))*km[:,:,0]
                            
                            SIGMAcos_wsum  += np.tile(profilesums['SIGMAcos_wsum'],(ncen+1,1,1))*km
                            SIGMApar_wsum  += np.tile(profilesums['SIGMApar_wsum'],(ncen+1,1,1))*km
                            SIGMAper_wsum  += np.tile(profilesums['SIGMAper_wsum'],(ncen+1,1,1))*km

                            GAMMATcos_wsum += np.tile(profilesums['GAMMATcos_wsum'],(ncen+1,1,1))*km
                            GAMMAXsin_wsum += np.tile(profilesums['GAMMAXsin_wsum'],(ncen+1,1,1))*km
    
                            COS2_2theta_wsum += np.tile(profilesums['COS2_2theta_wsum'],(ncen+1,1,1))*km
                            SIN2_2theta_wsum += np.tile(profilesums['SIN2_2theta_wsum'],(ncen+1,1,1))*km

                            Ninbin_par += np.tile(profilesums['N_inbin_par'],(ncen+1,1,1))*km
                            Ninbin_per += np.tile(profilesums['N_inbin_per'],(ncen+1,1,1))*km
                        
                
                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice,ts)
                print('TIME SLICE')
                print(ts)
                print('Estimated ramaining time')
                print(np.mean(tslice)*(len(Lsplit)-(l+1)))

        # AVERAGE LENS PARAMETERS AND SAVE IT IN HEADER
        
        zmean        = np.average(L.redshift,weights=Ntot)
        lM_mean      = np.log10(np.average(10**L.lgm,weights=Ntot))
        # c200_mean    = np.average(L.cNFW_S,weights=Ntot)
        # lM200_mean   = np.log10(np.average(10**L.lgMNFW_S,weights=Ntot))
        
        # q2d_mean     = np.average(L.q2d,weights=Ntot)
        # q2dr_mean    = np.average(L.q2dr,weights=Ntot)
        # q3d_mean     = np.average(L.q,weights=Ntot)
        # q3dr_mean    = np.average(L.qr,weights=Ntot)
        # s3d_mean     = np.average(L.s,weights=Ntot)
        # s3dr_mean    = np.average(L.sr,weights=Ntot)

        h = fits.Header()
        h.append(('N_LENSES',np.int(Nlenses)))
        h.append(('Lens cat',lcat))
        h.append(('MICE version sources 2.0'))
        h.append(('rs_min',np.round(rs_min,1)))
        h.append(('rs_max',np.round(rs_max,1)))
        h.append(('lM_min',np.round(lM_min,2)))
        h.append(('lM_max',np.round(lM_max,2)))
        h.append(('z_min',np.round(z_min,2)))
        h.append(('z_max',np.round(z_max,2)))
        h.append(('q_min',np.round(q_min,2)))
        h.append(('q_max',np.round(q_max,2)))
        h.append(('T_min',np.round(T_min,2)))
        h.append(('T_max',np.round(T_max,2)))
        h.append(('resNFW_max',np.round(resNFW_max,2)))
        h.append(('hcosmo',np.round(hcosmo,4)))
        h.append(('lM_mean',np.round(lM_mean,4)))
        # h.append(('lM200_mean',np.round(lM200_mean,4)))
        # h.append(('c200_mean',np.round(c200_mean,4)))
        h.append(('z_mean',np.round(zmean,4)))
        # h.append(('q2d_mean',np.round(q2d_mean,4)))
        # h.append(('q2dr_mean',np.round(q2dr_mean,4)))
        # h.append(('q3d_mean',np.round(q3d_mean,4)))
        # h.append(('q3dr_mean',np.round(q3dr_mean,4)))
        # h.append(('s3d_mean',np.round(s3d_mean,4)))        
        # h.append(('s3dr_mean',np.round(s3dr_mean,4))) 
        if miscen:
            h.append(('soff',np.round(soff,4)))
        if domap:
            
            # COMPUTING PROFILE        
                
            GT  = (GTsum/Ninbin)
            GX  = (GXsum/Ninbin)
            K   = (Ksum/Ninbin)
                    
            # COMPUTE COVARIANCE
            
            COV_Gtc = cov_matrix(GT[1:,:,0])
            COV_Gt  = cov_matrix(GT[1:,:,1])
            COV_Gtr = cov_matrix(GT[1:,:,2])
            
            COV_Gxc = cov_matrix(GX[1:,:,0])
            COV_Gx  = cov_matrix(GX[1:,:,1])
            COV_Gxr = cov_matrix(GX[1:,:,2])
    
            COV_Kc = cov_matrix(GX[1:,:,0])
            COV_K  = cov_matrix(GX[1:,:,1])
            COV_Kr = cov_matrix(GX[1:,:,2])
            
            table_pro = [fits.Column(name='xmpc', format='E', array=xbin),
                    fits.Column(name='ympc', format='E', array=ybin),
                    fits.Column(name='GT_control', format='E', array=GT[0,:,0]),
                    fits.Column(name='GT', format='E', array=GT[0,:,1]),
                    fits.Column(name='GT_reduced', format='E', array=GT[0,:,2]),
                    fits.Column(name='GX_control', format='E', array=GX[0,:,0]),
                    fits.Column(name='GX', format='E', array=GX[0,:,1]),
                    fits.Column(name='GX_reduced', format='E', array=GX[0,:,2]),
                    fits.Column(name='K_control', format='E', array=K[0,:,0]),
                    fits.Column(name='K', format='E', array=K[0,:,1]),
                    fits.Column(name='K_reduced', format='E', array=K[0,:,2])]
                    
                        
            table_cov = [fits.Column(name='COV_GT_control', format='E', array=COV_Gtc.flatten()),
                        fits.Column(name='COV_GT', format='E', array=COV_Gt.flatten()),
                        fits.Column(name='COV_GT_reduced', format='E', array=COV_Gtr.flatten()),
                        fits.Column(name='COV_GX_control', format='E', array=COV_Gxc.flatten()),
                        fits.Column(name='COV_GX', format='E', array=COV_Gx.flatten()),
                        fits.Column(name='COV_GX_reduced', format='E', array=COV_Gxr.flatten()),
                        fits.Column(name='COV_K_control', format='E', array=COV_Kc.flatten()),
                        fits.Column(name='COV_K', format='E', array=COV_K.flatten()),
                        fits.Column(name='COV_K_reduced', format='E', array=COV_Kr.flatten())]
    
    
        else:
            # COMPUTING PROFILE        
            Ninbin[DSIGMAwsum_T == 0] = 1.
                    
            Sigma     = (SIGMAwsum/Ninbin)
            DSigma_T  = (DSIGMAwsum_T/Ninbin)
            DSigma_X  = (DSIGMAwsum_X/Ninbin)
            
            SIGMA_cos  = (SIGMAcos_wsum/COS2_2theta_wsum).transpose(1,2,0)
            SIGMA_par  = (SIGMApar_wsum/Ninbin_par).transpose(1,2,0)
            SIGMA_per  = (SIGMAper_wsum/Ninbin_per).transpose(1,2,0)
            
            GAMMA_Tcos = (GAMMATcos_wsum/COS2_2theta_wsum).transpose(1,2,0)
            GAMMA_Xsin = (GAMMAXsin_wsum/SIN2_2theta_wsum).transpose(1,2,0)
            
            # COMPUTE COVARIANCE
    
            COV_S   = cov_matrix(Sigma[1:,:])
            COV_DSt  = cov_matrix(DSigma_T[1:,:])
            COV_DSx  = cov_matrix(DSigma_X[1:,:])
            
            COV_Scc = cov_matrix(SIGMA_cos[:,0,1:].T)
            COV_Sc  = cov_matrix(SIGMA_cos[:,1,1:].T)
            COV_Scr = cov_matrix(SIGMA_cos[:,2,1:].T)

            COV_Spar_c = cov_matrix(SIGMA_par[:,0,1:].T)
            COV_Spar   = cov_matrix(SIGMA_par[:,1,1:].T)
            COV_Spar_r = cov_matrix(SIGMA_par[:,2,1:].T)

            COV_Sper_c = cov_matrix(SIGMA_per[:,0,1:].T)
            COV_Sper   = cov_matrix(SIGMA_per[:,1,1:].T)
            COV_Sper_r = cov_matrix(SIGMA_per[:,2,1:].T)

            COV_Stc = cov_matrix(SIGMA_cos[:,0,1:].T)
            COV_St  = cov_matrix(SIGMA_cos[:,1,1:].T)
            COV_Str = cov_matrix(SIGMA_cos[:,2,1:].T)

            COV_Gtc = cov_matrix(GAMMA_Tcos[:,0,1:].T)
            COV_Gt  = cov_matrix(GAMMA_Tcos[:,1,1:].T)
            COV_Gtr = cov_matrix(GAMMA_Tcos[:,2,1:].T)
            
            COV_Gxc = cov_matrix(GAMMA_Xsin[:,0,1:].T)
            COV_Gx  = cov_matrix(GAMMA_Xsin[:,1,1:].T)
            COV_Gxr = cov_matrix(GAMMA_Xsin[:,2,1:].T)

            # WRITING OUTPUT FITS FILE
            
            table_pro = [fits.Column(name='Rp', format='E', array=R),
                    fits.Column(name='Sigma', format='E', array=Sigma[0]),
                    fits.Column(name='DSigma_T', format='E', array=DSigma_T[0]),
                    fits.Column(name='DSigma_X', format='E', array=DSigma_X[0]),
                    fits.Column(name='SIGMA_cos_control', format='E', array=SIGMA_cos[:,0,0]),
                    fits.Column(name='SIGMA_cos', format='E', array=SIGMA_cos[:,1,0]),
                    fits.Column(name='SIGMA_cos_reduced', format='E', array=SIGMA_cos[:,2,0]),
                    fits.Column(name='SIGMA_par_control', format='E', array=SIGMA_par[:,0,0]),
                    fits.Column(name='SIGMA_par', format='E', array=SIGMA_par[:,1,0]),
                    fits.Column(name='SIGMA_par_reduced', format='E', array=SIGMA_par[:,2,0]),
                    fits.Column(name='SIGMA_per_control', format='E', array=SIGMA_per[:,0,0]),
                    fits.Column(name='SIGMA_per', format='E', array=SIGMA_per[:,1,0]),
                    fits.Column(name='SIGMA_per_reduced', format='E', array=SIGMA_per[:,2,0]),
                    fits.Column(name='GAMMA_Tcos_control', format='E', array=GAMMA_Tcos[:,0,0]),
                    fits.Column(name='GAMMA_Tcos', format='E', array=GAMMA_Tcos[:,1,0]),
                    fits.Column(name='GAMMA_Tcos_reduced', format='E', array=GAMMA_Tcos[:,2,0]),
                    fits.Column(name='GAMMA_Xsin_control', format='E', array=GAMMA_Xsin[:,0,0]),
                    fits.Column(name='GAMMA_Xsin', format='E', array=GAMMA_Xsin[:,1,0]),
                    fits.Column(name='GAMMA_Xsin_reduced', format='E', array=GAMMA_Xsin[:,2,0]),
                    fits.Column(name='Ninbin', format='E', array=Ninbin[0])]
                    
                        
            table_cov = [fits.Column(name='COV_DST', format='E', array=COV_DSt.flatten()),
                        fits.Column(name='COV_S', format='E', array=COV_S.flatten()),
                        fits.Column(name='COV_DSX', format='E', array=COV_DSx.flatten()),
                        fits.Column(name='COV_Scos_control', format='E', array=COV_Scc.flatten()),
                        fits.Column(name='COV_Scos', format='E', array=COV_Sc.flatten()),
                        fits.Column(name='COV_Scos_reduced', format='E', array=COV_Scr.flatten()),
                        fits.Column(name='COV_Spar_control', format='E', array=COV_Spar_c.flatten()),
                        fits.Column(name='COV_Spar', format='E', array=COV_Spar.flatten()),
                        fits.Column(name='COV_Spar_reduced', format='E', array=COV_Spar_r.flatten()),
                        fits.Column(name='COV_Sper_control', format='E', array=COV_Sper_c.flatten()),
                        fits.Column(name='COV_Sper', format='E', array=COV_Sper.flatten()),
                        fits.Column(name='COV_Sper_reduced', format='E', array=COV_Sper_r.flatten()),
                        fits.Column(name='COV_GT_control', format='E', array=COV_Gtc.flatten()),
                        fits.Column(name='COV_GT', format='E', array=COV_Gt.flatten()),
                        fits.Column(name='COV_GT_reduced', format='E', array=COV_Gtr.flatten()),
                        fits.Column(name='COV_GX_control', format='E', array=COV_Gxc.flatten()),
                        fits.Column(name='COV_GX', format='E', array=COV_Gx.flatten()),
                        fits.Column(name='COV_GX_reduced', format='E', array=COV_Gxr.flatten())]
        
        tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_pro))
        tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_cov))
        
        
        primary_hdu = fits.PrimaryHDU(header=h)
        
        hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])
        
        hdul.writeto('../'+output_file,overwrite=True)
                
        tfin = time.time()
        
        print('TOTAL TIME ',(tfin-tini)/60.)
        


main(lcat,sample,lM_min,lM_max,z_min,z_max,q_min,q_max,T_min,T_max,rs_min,rs_max,resNFW_max,relax,R5s_min,R5s_max,domap,RIN,ROUT,ndots,ncores,idlist,hcosmo,addnoise,misalign,miscen,soff)
