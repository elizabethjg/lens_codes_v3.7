import sys
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import time
import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM
from maria_func import *
from fit_profiles_curvefit import *
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from astropy.constants import G,c,M_sun,pc
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
from medianas import separate_medianas

#parameters
cvel = c.value;   # Speed of light (m.s-1)
G    = G.value;   # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

w3 = fits.open('/home/elizabeth/CFHTLens/CFHTLens_W3.fits')[1].data
w1 = fits.open('/home/elizabeth/CFHTLens/CFHTLens_W1.fits')[1].data
w2 = fits.open('/home/elizabeth/CFHTLens/CFHTLens_W2.fits')[1].data
# w2 = fits.open('/home/elizabeth/KiDS-450/KiDS_DR3.1_G9_ugri_shear.fits')[1].data

m1 = (w1.ODDS >= 0.5)*(w1.Z_B > 0.2)*(w1.Z_B < 1.2)*(w1.weight > 0)*(w1.fitclass == 0)*(w1.MASK <= 1)
m2 = (w2.ODDS >= 0.5)*(w2.Z_B > 0.2)*(w2.Z_B < 1.2)*(w2.weight > 0)*(w2.fitclass == 0)*(w2.MASK <= 1)
m3 = (w3.ODDS >= 0.5)*(w3.Z_B > 0.2)*(w3.Z_B < 1.2)*(w3.weight > 0)*(w3.fitclass == 0)*(w3.MASK <= 1)

w1_sources = w1[m1]
w2_sources = w2[m2]
w3_sources = w3[m3]

'''
sample = 'pru'
pcat = '_photo_z_2nd_run_mag_i_best'
z_min = 0.0
z_max = 0.4
Lratio_min = 0.0
Lratio_max = 1.
RIN = 100.
ROUT =10000.
ndots= 15
ncores=32
odds_min=0.5
hcosmo=1.
fields='all'
'''

def partial_profile(RA0,DEC0,Z,field,
                    RIN,ROUT,ndots,h,nboot=100):

        if field == 1:
            S = w1_sources
        if field == 2:
            S = w2_sources
        if field == 3:
            S = w3_sources

        cosmo = LambdaCDM(H0=100*h, Om0=0.3, Ode0=0.7)
        ndots = int(ndots)
        
        dl  = cosmo.angular_diameter_distance(Z).value
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
        
        
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        # delta = (ROUT/(3600*KPCSCALE)) + 0.1
        delta = (15.*ROUT)/(3600*KPCSCALE)

        
        mask_region = (abs(S.RAJ2000 -RA0) < delta)&(abs(S.DECJ2000 - DEC0) < delta)
               
        mask = mask_region*(S.Z_B > (Z + 0.3))
        # mask = mask_region*(abs(S.Z_B - Z) < 0.3)
        
        catdata = S[mask]

        z2 = catdata.Z_B
        z1 = np.ones(len(z2))*Z
        
        z1[catdata.Z_B < Z] = catdata.Z_B[catdata.Z_B < Z]
        z2[catdata.Z_B < Z] = (np.ones(len(z2))*Z)[catdata.Z_B < Z]
        

        ds  = cosmo.angular_diameter_distance(catdata.Z_B).value
        dls = cosmo.angular_diameter_distance_z1z2(z1, z2).value
                
        BETA_array = dls/ds
        
        
        Dl = dl*1.e6*pc
        sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)



        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.RAJ2000),
                                        np.deg2rad(catdata.DECJ2000),
                                        np.deg2rad(RA0),
                                        np.deg2rad(DEC0))
               
        #Correct polar angle for e1, e2
        theta = theta+np.pi/2.
        
        e1 = catdata.e1
        try:
            e2 = catdata.e2-catdata.c2
        except:
            e2 = catdata.e2
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
        #get cross ellipticities
        ex = (e1*np.sin(2*theta)-e2*np.cos(2*theta))*sigma_c
        
        del(e1)
        del(e2)
        
        r=np.rad2deg(rads)*3600*KPCSCALE
        del(rads)
        
        peso = catdata.weight
        peso = peso/(sigma_c**2) 
        m    = catdata.m
        
        Ntot = len(catdata)
        # del(catdata)    
        
        
        dig = np.digitize(r,bines)
        
        DSIGMAwsum_T = []
        DSIGMAwsum_X = []
        WEIGHTsum    = []
        Mwsum        = []
        BOOTwsum_T   = np.zeros((nboot,ndots))
        BOOTwsum_X   = np.zeros((nboot,ndots))
        BOOTwsum     = np.zeros((nboot,ndots))
        NGAL         = []
        
        
        for nbin in range(ndots):
                mbin = dig == nbin+1              
                
                DSIGMAwsum_T = np.append(DSIGMAwsum_T,(et[mbin]*peso[mbin]).sum())
                DSIGMAwsum_X = np.append(DSIGMAwsum_X,(ex[mbin]*peso[mbin]).sum())
                WEIGHTsum    = np.append(WEIGHTsum,(peso[mbin]).sum())
                Mwsum        = np.append(Mwsum,((1.+m[mbin])*peso[mbin]).sum())
                NGAL         = np.append(NGAL,mbin.sum())
                
                index = np.arange(mbin.sum())
                if mbin.sum() == 0:
                        continue
                else:
                        with NumpyRNGContext(1):
                                bootresult = bootstrap(index, nboot)
                        INDEX=bootresult.astype(int)
                        BOOTwsum_T[:,nbin] = np.sum(np.array(et[mbin]*peso[mbin])[INDEX],axis=1)
                        BOOTwsum_X[:,nbin] = np.sum(np.array(ex[mbin]*peso[mbin])[INDEX],axis=1)
                        BOOTwsum[:,nbin]   = np.sum(np.array(peso[mbin])[INDEX],axis=1)
                        
                # if nbin == ndots-1:
                    
                    # delta = (ROUT/(3600*KPCSCALE))
                    # mask_region = (abs(catdata.RAJ2000 -RA0) > delta)&(abs(catdata.DECJ2000 - DEC0) > delta)
                    # print('delta ',delta)
                    # print(max(abs(catdata.RAJ2000 -RA0)[mask_region]))
                    # print(max(abs(catdata.DECJ2000 - DEC0)[mask_region]))
                    # print(max(abs(catdata.RAJ2000 -RA0)[mask_region])*(3600*KPCSCALE))
                    # print(max(abs(catdata.DECJ2000 - DEC0)[mask_region])*(3600*KPCSCALE))
        
        output = {'DSIGMAwsum_T':DSIGMAwsum_T,'DSIGMAwsum_X':DSIGMAwsum_X,
                   'WEIGHTsum':WEIGHTsum, 'Mwsum':Mwsum, 
                   'BOOTwsum_T':BOOTwsum_T, 'BOOTwsum_X':BOOTwsum_X, 'BOOTwsum':BOOTwsum, 
                   'Ntot':Ntot,'NGAL':NGAL}
        
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

def kmask_func(fields,ncen=100):

        pcat = '_zspec'

        L1 = np.loadtxt('../parespaus/Pares-PAUS_W1-Photo_z_calibrate'+pcat).T
        field = np.ones(len(L1[1]))*1                             
        L1 = np.vstack((L1,field))                                
                                                                  
        L2 = np.loadtxt('../parespaus/Pares-PAUS_W2-Photo_z_calibrate'+pcat).T
        field = np.ones(len(L2[1]))*2                             
        L2 = np.vstack((L2,field))                                
                                          
        L3 = np.loadtxt('../parespaus/Pares-PAUS_W3-Photo_z_calibrate'+pcat).T
        field = np.ones(len(L3[1]))*3
        L3 = np.vstack((L3,field))
        
        
        if fields == 'all':
            L = np.vstack((L1.T,L2.T,L3.T)).T
        elif fields == 'w3':
            L = L3
        elif fields == 'w1':
            L = L1
        elif fields == 'w2':
            L = L2
        elif fields == 'w1w2':
            L = np.vstack((L1.T,L2.T)).T
        elif fields == 'w1w3':
            L = np.vstack((L1.T,L3.T)).T
        elif fields == 'w2w3':
            L = np.vstack((L2.T,L3.T)).T
        
        # Define K masks
        
        X = np.array([L[4],L[5]]).T

        try:
            km = kmeans_sample(X, ncen, maxiter=100, tol=1.0e-5)
        except:
            print('WARNING!!! USING 40 CEN FOR COV MATRIX')
            km = kmeans_sample(X, 40, maxiter=100, tol=1.0e-5)
        return km


def main(sample,pcat,
        z_min = 0.0, z_max = 0.6,
        Lratio_min = 0.0, Lratio_max = 1.,
        Mtot_min = -24., Mtot_max = -19.0,
        colorcut = False,
        odds_min=0.5, RIN = 100., ROUT =5000.,
        ndots= 15,ncores=10,hcosmo=1.,fields='all'):

        '''
        
        INPUT
        ---------------------------------------------------------
        sample         (str) sample name
        z_min          (float) lower limit for z - >=
        z_max          (float) higher limit for z - <
        Lratio_min     (float) lower limit for L2/L1 - >=
        Lratio_max     (float) higher limit for L2/l1 - <
        Mtot_min       (float) lower limit for Mtot- >=
        Mtot_max       (float) higher limit for Mtot - <
        colorcut       (str) red or blue
        odds_min       (float) cut in odds
        RIN            (float) Inner bin radius of profile
        ROUT           (float) Outer bin radius of profile
        ndots          (int) Number of bins of the profile
        ncores         (int) to run in parallel, number of cores
        hcosmo         (float) H0 = 100.*h
        fields         (str) fields considered
                        'all', 'w1w2', 'w3'
        '''

        cosmo = LambdaCDM(H0=100*hcosmo, Om0=0.3, Ode0=0.7)
        tini = time.time()
        
        print('Selecting pairs with:')
        print(z_min,' <= z < ',z_max)
        print(Lratio_min,' <= L2/L1 < ',Lratio_max)
        print(Mtot_min,' <= M_tot < ',Mtot_max)
        print('Background galaxies with:')
        # print('ODDS > ',odds_min)
        print('Profile has ',ndots,'bins')
        print('from ',RIN,'kpc to ',ROUT,'kpc')
        print('h = ',hcosmo)
              
        # Defining radial bins
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        #reading cats
        
        
        if fields == 'all':
            L1 = np.loadtxt('../pareconk/Pares-PAUS_W1-'+pcat).T
            field = np.ones(len(L1[1]))*1                       
            L1 = np.vstack((L1,field))                          
                                                                
            L2 = np.loadtxt('../pareconk/Pares-PAUS_W2-'+pcat).T
            field = np.ones(len(L2[1]))*2                       
            L2 = np.vstack((L2,field))                          
                                                                
            L3 = np.loadtxt('../pareconk/Pares-PAUS_W3-'+pcat).T
            field = np.ones(len(L3[1]))*3                       
            L3 = np.vstack((L3,field))                          
                                                                
            L = np.vstack((L1.T,L2.T,L3.T)).T                   
                                                                
        elif fields == 'w3':                                    
                                                                
            L3 = np.loadtxt('../pareconk/Pares-PAUS_W3-'+pcat).T
            field = np.ones(len(L3[1]))*3                       
            L3 = np.vstack((L3,field))                          
                                                                
            L = L3                                              
                                                                
        elif fields == 'w1':                                    
                                                                
            L1 = np.loadtxt('../pareconk/Pares-PAUS_W1-'+pcat).T
            field = np.ones(len(L1[1]))*1                       
            L1 = np.vstack((L1,field))                          
                                                                
            L = L1                                              
        elif fields == 'w2':                                    
                                                                
            L2 = np.loadtxt('../pareconk/Pares-PAUS_W2-'+pcat).T
            field = np.ones(len(L2[1]))*2                       
            L2 = np.vstack((L2,field))                          
                                                                
            L = L2                                              
                                                                
        elif fields == 'w1w2':                                  
                                                                
            L1 = np.loadtxt('../pareconk/Pares-PAUS_W1-'+pcat).T
            field = np.ones(len(L1[1]))*1                       
            L1 = np.vstack((L1,field))                          
                                                                
            L2 = np.loadtxt('../pareconk/Pares-PAUS_W2-'+pcat).T
            field = np.ones(len(L2[1]))*2                       
            L2 = np.vstack((L2,field))                          
                                                                
            L = np.vstack((L1.T,L2.T)).T                        
        elif fields == 'w1w3':                                  
                                                                
            L1 = np.loadtxt('../pareconk/Pares-PAUS_W1-'+pcat).T
            field = np.ones(len(L1[1]))*1                       
            L1 = np.vstack((L1,field))                          
                                                                                              
            L3 = np.loadtxt('../pareconk/Pares-PAUS_W3-'+pcat).T
            field = np.ones(len(L3[1]))*3                       
            L3 = np.vstack((L3,field))                          
                                                                
            L = np.vstack((L1.T,L3.T)).T                        
        elif fields == 'w2w3':                                  
                                                                
                                                                
            L2 = np.loadtxt('../pareconk/Pares-PAUS_W2-'+pcat).T
            field = np.ones(len(L2[1]))*2                       
            L2 = np.vstack((L2,field))                          
                                                                
            L3 = np.loadtxt('../pareconk/Pares-PAUS_W3-'+pcat).T
            field = np.ones(len(L3[1]))*3
            L3 = np.vstack((L3,field))           
            
            L = np.vstack((L2.T,L3.T)).T
        
        Lratio = 10.**(-0.4*(L[-2]-L[8]))
        
        # from astropy.cosmology import WMAP9 as cosmo_lum       
        M1 = L[8]-5.*np.log10(np.array(cosmo.luminosity_distance(L[3]))*1.e6)+5
        M2 = L[13]-5.*np.log10(np.array(cosmo.luminosity_distance(L[3]))*1.e6)+5
        Mtot = -2.5*np.log10(10**(-0.4*M1)+10**(-0.4*M2))
        M1i = L[7]-5.*np.log10(np.array(cosmo.luminosity_distance(L[3]))*1.e6)+5
        M2i = L[12]-5.*np.log10(np.array(cosmo.luminosity_distance(L[3]))*1.e6)+5
        Mtoti = -2.5*np.log10(10**(-0.4*M1i)+10**(-0.4*M2i))

        mz      = (L[3] >= z_min)*(L[3] < z_max)
        mtot    = (Mtot >= Mtot_min)*(Mtot < Mtot_max)
        mratio  = (Lratio >= Lratio_min)*(Lratio < Lratio_max)
        mlenses = mz*mratio*mtot    
            
        if colorcut == 'red':
            print('Selecting red pairs')
            mcolor = separate_medianas(Mtot,Mtot-Mtoti,plot=False)[-1]
            mlenses = mlenses*mcolor
        if colorcut == 'blue':
            print('Selecting blue pairs')
            mcolor = ~separate_medianas(Mtot,Mtot-Mtoti,plot=False)[-1]        
            mlenses = mlenses*mcolor

        Nlenses = mlenses.sum()
        
        Mmean  = -2.5*np.log10(np.mean(10**(-0.4*Mtot[mlenses])))
        Mmeani = -2.5*np.log10(np.mean(10**(-0.4*Mtoti[mlenses])))

        if Nlenses < ncores:
                ncores = Nlenses

        sample = sample+'_'+fields+'_'
        outfile = '../profiles_pk/profile_'+sample+pcat+'.fits'
        print('Saving in file ',outfile)        
        print('Nlenses',Nlenses)
        print('CORRIENDO EN ',ncores,' CORES')

        # Define K masks
        L = L[:,mlenses]        
        X = np.array([L[4],L[5]]).T
        
        ncen = 100
        km = kmask_func(fields,ncen)
        labels = km.find_nearest(X)
        kmask = np.zeros((ncen+1,len(X)))
        kmask[0] = np.ones(len(X)).astype(bool)
        
        for j in np.arange(1,ncen+1):
            kmask[j] = ~(labels == j-1)


        # par_gal_id,ra_par,dec_par,z_par,ra_1,dec_1,z_1,i_auto_1,mr_1,ra_2,dec_2,z_2,i_auto_2,mr_2

        mr1 = L[8]
        mr2 = L[13]
        mi1 = L[7]
        mi2 = L[12]
        RA  = L[4]
        DEC = L[5]
        z   = L[3]
        
        
        # SPLIT LENSING CAT
        
        lbins = int(round(Nlenses/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nlenses)]
        Lsplit = np.split(L.T,slices)
        Ksplit = np.split(kmask.T,slices)
                
        # WHERE THE SUMS ARE GOING TO BE SAVED
        
        DSIGMAwsum_T = np.zeros((ncen+1,ndots)) 
        DSIGMAwsum_X = np.zeros((ncen+1,ndots))
        WEIGHTsum    = np.zeros((ncen+1,ndots))
        NGALsum      = np.zeros(ndots)
        Mwsum        = np.zeros((ncen+1,ndots))
        
        BOOTwsum_T   = np.zeros((100,ndots))
        BOOTwsum_X   = np.zeros((100,ndots))
        BOOTwsum     = np.zeros((100,ndots))
        Ntot         = []
        tslice       = np.array([])
        
        for l in range(len(Lsplit)):
                
                print('RUN ',l+1,' OF ',len(Lsplit))
                
                t1 = time.time()
                
                num = len(Lsplit[l])
                
                rin  = RIN*np.ones(num)
                rout = ROUT*np.ones(num)
                nd   = ndots*np.ones(num)
                h_a  = hcosmo*np.ones(num)
                
                if num == 1:
                        entrada = [Lsplit[l].T[1][0], Lsplit[l].T[2][0],
                                   Lsplit[l].T[3][0],Lsplit[l].T[-1][0],
                                   RIN,ROUT,ndots,hcosmo]
                        
                        salida = [partial_profile_unpack(entrada)]
                else:          
                        entrada = np.array([Lsplit[l].T[1], Lsplit[l].T[2],
                                   Lsplit[l].T[3],Lsplit[l].T[-1],
                                        rin,rout,nd,h_a]).T
                        
                        pool = Pool(processes=(num))
                        salida = np.array(pool.map(partial_profile_unpack, entrada))
                        pool.terminate()
                
                for j in range(len(salida)):
                    
                    profilesums = salida[j]

                    km      = np.tile(Ksplit[l][j],(ndots,1)).T

                    DSIGMAwsum_T += np.tile(profilesums['DSIGMAwsum_T'],(ncen+1,1))*km
                    DSIGMAwsum_X += np.tile(profilesums['DSIGMAwsum_X'],(ncen+1,1))*km
                    WEIGHTsum    += np.tile(profilesums['WEIGHTsum'],(ncen+1,1))*km
                    Mwsum        += np.tile(profilesums['Mwsum'],(ncen+1,1))*km

                    NGALsum      += profilesums['NGAL']
                    BOOTwsum_T   += profilesums['BOOTwsum_T']
                    BOOTwsum_X   += profilesums['BOOTwsum_X']
                    BOOTwsum     += profilesums['BOOTwsum']
                    Ntot         = np.append(Ntot,profilesums['Ntot'])                    
                
                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice,ts)
                print('TIME SLICE')
                print(ts)
                print('Estimated ramaining time')
                print((np.mean(tslice)*(len(Lsplit)-(l+1))))
        
        # COMPUTING PROFILE        
                
        Mcorr     = Mwsum/WEIGHTsum
        DSigma_T  = (DSIGMAwsum_T/WEIGHTsum)/Mcorr
        DSigma_X  = (DSIGMAwsum_X/WEIGHTsum)/Mcorr
        eDSigma_T =  np.std((BOOTwsum_T/BOOTwsum),axis=0)/Mcorr[0]
        eDSigma_X =  np.std((BOOTwsum_X/BOOTwsum),axis=0)/Mcorr[0]
        
        COV_St  = cov_matrix(DSigma_T[1:,:])
        COV_Sx  = cov_matrix(DSigma_X[1:,:])

        
        # AVERAGE LENS PARAMETERS
        
        zmean        = np.average(z,weights=Ntot)
        
 
        # WRITING OUTPUT FITS FILE
        
        table_pro  = [fits.Column(name='Rp', format='D', array=R),
                      fits.Column(name='DSigma_T', format='D', array=DSigma_T[0]),
                      fits.Column(name='error_DSigma_T', format='D', array=eDSigma_T),
                      fits.Column(name='DSigma_X', format='D', array=DSigma_X[0]),
                      fits.Column(name='error_DSigma_X', format='D', array=eDSigma_X),
                      fits.Column(name='NGAL', format='D', array=NGALsum),
                      fits.Column(name='NGAL_w', format='D', array=WEIGHTsum[0])]

        table_cov = [fits.Column(name='COV_ST', format='E', array=COV_St.flatten()),
                     fits.Column(name='COV_SX', format='E', array=COV_Sx.flatten())]
        
        h = fits.Header()
        h.append(('N_LENSES',np.int(Nlenses)))
        h.append(('RIN',np.round(RIN,4)))
        h.append(('ROUT',np.round(ROUT,4)))
        h.append(('z_min',np.round(z_min,4)))
        h.append(('z_max',np.round(z_max,4)))
        h.append(('z_mean',np.round(zmean,4)))
        h.append(('M_min',np.round(Mtot_min,4)))
        h.append(('M_max',np.round(Mtot_max,4)))
        h.append(('M_mean_r',np.round(Mmean,4)))
        h.append(('M_mean_i',np.round(Mmeani,4)))
        h.append(('Lratio_min',np.round(Lratio_min,4)))
        h.append(('Lratio_max',np.round(Lratio_max,4)))
        h.append(('hcosmo',np.round(hcosmo,4)))

        tbhdu_pro = fits.BinTableHDU.from_columns(fits.ColDefs(table_pro))
        tbhdu_cov = fits.BinTableHDU.from_columns(fits.ColDefs(table_cov))
        
        
        primary_hdu = fits.PrimaryHDU(header=h)
        
        hdul = fits.HDUList([primary_hdu, tbhdu_pro, tbhdu_cov])

        hdul.writeto(outfile,overwrite=True)
                
        tfin = time.time()
        
        print('TOTAL TIME ',(tfin-tini)/60.)
        


if __name__ == '__main__':
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-sample', action='store', dest='sample',default='pru')
        parser.add_argument('-z_min', action='store', dest='z_min', default=0.0)
        parser.add_argument('-z_max', action='store', dest='z_max', default=1.0)
        parser.add_argument('-Lratio_min', action='store', dest='Lratio_min', default=0.0)
        parser.add_argument('-Lratio_max', action='store', dest='Lratio_max', default=1.0)
        parser.add_argument('-Mtot_min', action='store', dest='Mtot_min', default=-24.0)
        parser.add_argument('-Mtot_max', action='store', dest='Mtot_max', default=-19.0)
        parser.add_argument('-colorcut', action='store', dest='colorcut', default='False')
        parser.add_argument('-ODDS_min', action='store', dest='ODDS_min', default=0.5)
        parser.add_argument('-RIN', action='store', dest='RIN', default=100.)
        parser.add_argument('-ROUT', action='store', dest='ROUT', default=5000.)
        parser.add_argument('-nbins', action='store', dest='nbins', default=10)
        parser.add_argument('-ncores', action='store', dest='ncores', default=10)
        parser.add_argument('-h_cosmo', action='store', dest='h_cosmo', default=1.)
        parser.add_argument('-pcat', action='store', dest='pcat', default='_photo_z_2nd_run_mag_i')
        parser.add_argument('-fields', action='store', dest='fields', default='all')
        args = parser.parse_args()
        
        sample     = args.sample
        z_min      = float(args.z_min) 
        z_max      = float(args.z_max) 
        Lratio_min = float(args.Lratio_min) 
        Lratio_max = float(args.Lratio_max) 
        Mtot_min   = float(args.Mtot_min) 
        Mtot_max   = float(args.Mtot_max) 
        colorcut   = args.colorcut
        ODDS_min   = float(args.ODDS_min)
        RIN        = float(args.RIN)
        ROUT       = float(args.ROUT)
        nbins      = int(args.nbins)
        ncores     = int(args.ncores)
        h          = float(args.h_cosmo)
        fields     = args.fields
        
        main(sample,args.pcat,z_min,z_max,Lratio_min,Lratio_max,Mtot_min,Mtot_max,colorcut,ODDS_min,RIN,ROUT,nbins,ncores,h,fields)

